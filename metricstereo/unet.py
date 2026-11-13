import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops

    _XFORMERS = True
except ImportError:
    _XFORMERS = False

MODEL_CHANNELS = 320
CHANNEL_MULT = (1, 2, 4, 4)
NUM_RES_BLOCKS = 2
ATTENTION_RESOLUTIONS = (4, 2, 1)
HEAD_CHANNELS = 64
CONTEXT_DIM = 1024
CAMERA_DIM = 16


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x, num_frames=1):
        return super().forward(x).type(x.dtype)


class SyncGroupNorm32(nn.GroupNorm):
    """SyncVN: GroupNorm whose mean and variance are pooled over the views of a scene (batch = scenes x views)."""

    def forward(self, x, num_frames=1):
        if num_frames <= 1:
            return super().forward(x).type(x.dtype)
        nf, c, g = x.shape[0], x.shape[1], self.num_groups
        xg = x.float().reshape(nf // num_frames, num_frames, g, c // g, *x.shape[2:])
        dims = [1, 3] + list(range(4, xg.dim()))
        mean, var = xg.mean(dim=dims, keepdim=True), xg.var(dim=dims, keepdim=True, unbiased=False)
        out = ((xg - mean) / torch.sqrt(var + self.eps)).reshape(x.shape)
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out.type(x.dtype)


def normalization(channels, sync=False):
    return (SyncGroupNorm32 if sync else GroupNorm32)(32, channels)


class Attention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads, self.dim_head = heads, dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(0.0))

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = x if context is None else context
        k, v = self.to_k(context), self.to_v(context)
        b = q.shape[0]
        q, k, v = (
            t.unsqueeze(3).reshape(b, t.shape[1], self.heads, self.dim_head).permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head).contiguous()
            for t in (q, k, v)
        )
        try:
            if not _XFORMERS:
                raise NotImplementedError
            out = xformers.ops.memory_efficient_attention(q, k, v)
        except NotImplementedError:  # no xFormers, or no xFormers kernel for this GPU (e.g. Blackwell)
            out = F.scaled_dot_product_attention(q, k, v)
        out = (out.unsqueeze(0).reshape(b, self.heads, out.shape[1], self.dim_head).permute(0, 2, 1, 3)
               .reshape(b, out.shape[1], self.heads * self.dim_head))
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.attn1 = Attention(dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(dim)
        self.attn2 = Attention(dim, context_dim=CONTEXT_DIM, heads=n_heads, dim_head=d_head)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, x, context, num_frames):
        x = rearrange(x, "(b f) l c -> b (f l) c", f=num_frames).contiguous()
        x = self.attn1(self.norm1(x)) + x
        x = rearrange(x, "b (f l) c -> (b f) l c", f=num_frames).contiguous()
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SpatialTransformer(nn.Module):
    def __init__(self, channels, n_heads, d_head):
        super().__init__()
        inner = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(channels, inner)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(inner, n_heads, d_head)])
        self.proj_out = nn.Linear(channels, inner)

    def forward(self, x, context, num_frames):
        b, c, h, w = x.shape
        x_in = x
        x = rearrange(self.norm(x), "b c h w -> b (h w) c").contiguous()
        x = self.proj_in(x)
        x = self.transformer_blocks[0](x, context, num_frames)
        x = self.proj_out(x)
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous() + x_in


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, sync=False):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(normalization(channels, sync), nn.SiLU(),
                                       nn.Conv2d(channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(normalization(out_channels, sync), nn.SiLU(), nn.Dropout(p=0.0),
                                        nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip_connection = nn.Identity() if out_channels == channels else nn.Conv2d(channels, out_channels, 1)

    def forward(self, x, emb, num_frames=1):
        h = self.in_layers[2](self.in_layers[1](self.in_layers[0](x, num_frames)))
        emb_out = self.emb_layers(emb).type(h.dtype)[..., None, None]
        h = self.out_layers[1:](self.out_layers[0](h + emb_out, num_frames))
        return self.skip_connection(x) + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Sequential(nn.Sequential):
    def forward(self, x, emb, context, num_frames):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb, num_frames)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, num_frames)
            else:
                x = layer(x)
        return x


class TwoViewUNet(nn.Module):
    def __init__(self, in_channels, out_channels, sync_view_norm=False):
        super().__init__()
        ch = MODEL_CHANNELS
        emb_dim = ch * 4
        self.time_embed = nn.Sequential(nn.Linear(ch, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.camera_embed = nn.Sequential(nn.Linear(CAMERA_DIM, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

        def attn(c):
            return SpatialTransformer(c, c // HEAD_CHANNELS, HEAD_CHANNELS)

        self.input_blocks = nn.ModuleList([Sequential(nn.Conv2d(in_channels, ch, 3, padding=1))])
        chans, ds = [ch], 1
        for level, mult in enumerate(CHANNEL_MULT):
            for _ in range(NUM_RES_BLOCKS):
                layers = [ResBlock(ch, emb_dim, mult * MODEL_CHANNELS, sync_view_norm)]
                ch = mult * MODEL_CHANNELS
                if ds in ATTENTION_RESOLUTIONS:
                    layers.append(attn(ch))
                self.input_blocks.append(Sequential(*layers))
                chans.append(ch)
            if level != len(CHANNEL_MULT) - 1:
                self.input_blocks.append(Sequential(Downsample(ch)))
                chans.append(ch)
                ds *= 2
        self.middle_block = Sequential(ResBlock(ch, emb_dim, sync=sync_view_norm), attn(ch),
                                       ResBlock(ch, emb_dim, sync=sync_view_norm))
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(CHANNEL_MULT))[::-1]:
            for i in range(NUM_RES_BLOCKS + 1):
                layers = [ResBlock(ch + chans.pop(), emb_dim, MODEL_CHANNELS * mult, sync_view_norm)]
                ch = MODEL_CHANNELS * mult
                if ds in ATTENTION_RESOLUTIONS:
                    layers.append(attn(ch))
                if level and i == NUM_RES_BLOCKS:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(Sequential(*layers))
        self.out = nn.Sequential(normalization(ch), nn.SiLU(), nn.Conv2d(MODEL_CHANNELS, out_channels, 3, padding=1))

    def forward(self, x, timesteps, context, camera=None, num_frames=1):
        emb = self.time_embed(timestep_embedding(timesteps, MODEL_CHANNELS).to(x.dtype))
        if camera is not None:
            emb = emb + self.camera_embed(camera)
        h, hs = x.type(torch.bfloat16), []
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames)
            hs.append(h)
        h = self.middle_block(h, emb, context, num_frames)
        for module in self.output_blocks:
            h = module(torch.cat([h, hs.pop()], dim=1), emb, context, num_frames)
        return self.out(h.type(x.dtype))
