import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    import xformers.ops

    _XFORMERS = True
except ImportError:
    _XFORMERS = False

CH, CH_MULT, NUM_RES_BLOCKS, Z_CHANNELS = 128, (1, 2, 4, 4), 2, 4
LORA_RANK, LORA_ALPHA = 8, 1.0


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(c):
    return nn.GroupNorm(num_groups=32, num_channels=c, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.norm = Normalize(c)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(c, c, 1) for _ in range(4))

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, hh, ww = q.shape
        q, k, v = (rearrange(t, "b c h w -> b (h w) c") for t in (q, k, v))
        try:
            if not _XFORMERS:
                raise NotImplementedError
            qx, kx, vx = (t.unsqueeze(3).reshape(b, t.shape[1], 1, c).permute(0, 2, 1, 3).reshape(b, t.shape[1], c)
                          .contiguous() for t in (q, k, v))
            out = xformers.ops.memory_efficient_attention(qx, kx, vx)
            out = out.unsqueeze(0).reshape(b, 1, out.shape[1], c).permute(0, 2, 1, 3).reshape(b, out.shape[1], c)
        except NotImplementedError:  # no xFormers, or no xFormers kernel for this GPU (e.g. Blackwell)
            out = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
        return x + self.proj_out(rearrange(out, "b (h w) c -> b c h w", h=hh, w=ww))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, CH, 3, padding=1)
        in_mult = (1,) + CH_MULT
        self.down = nn.ModuleList()
        for i, mult in enumerate(CH_MULT):
            level = nn.Module()
            level.block = nn.ModuleList()
            block_in = CH * in_mult[i]
            for _ in range(NUM_RES_BLOCKS):
                level.block.append(ResnetBlock(block_in, CH * mult))
                block_in = CH * mult
            if i != len(CH_MULT) - 1:
                level.downsample = nn.Module()
                level.downsample.conv = nn.Conv2d(block_in, block_in, 3, stride=2, padding=0)
            self.down.append(level)
        self.mid = nn.Module()
        self.mid.block_1, self.mid.attn_1, self.mid.block_2 = (ResnetBlock(block_in, block_in), AttnBlock(block_in),
                                                               ResnetBlock(block_in, block_in))
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * Z_CHANNELS, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i, level in enumerate(self.down):
            for block in level.block:
                h = block(h)
            if i != len(self.down) - 1:
                h = level.downsample.conv(F.pad(h, (0, 1, 0, 1), mode="constant", value=0))
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(nonlinearity(self.norm_out(h)))


class LoRAConv2d(nn.Module):

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_down = nn.Conv2d(base.in_channels, LORA_RANK, kernel_size=base.kernel_size, stride=base.stride,
                                   padding=base.padding, dilation=base.dilation, bias=False)
        self.lora_up = nn.Conv2d(LORA_RANK, base.out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.lora_down(x)) * (LORA_ALPHA / LORA_RANK)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        block_in = CH * CH_MULT[-1]
        self.conv_in = nn.Conv2d(Z_CHANNELS, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1, self.mid.attn_1, self.mid.block_2 = (ResnetBlock(block_in, block_in), AttnBlock(block_in),
                                                               ResnetBlock(block_in, block_in))
        self.up = nn.ModuleList()
        for i in reversed(range(len(CH_MULT))):
            level = nn.Module()
            level.block = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS + 1):
                level.block.append(ResnetBlock(block_in, CH * CH_MULT[i]))
                block_in = CH * CH_MULT[i]
            if i != 0:
                level.upsample = nn.Module()
                level.upsample.conv = nn.Conv2d(block_in, block_in, 3, padding=1)
            self.up.insert(0, level)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)


def add_lora(module):
    wrapped = []
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (1, 1):
                continue
            lora = LoRAConv2d(child)
            setattr(module, name, lora)
            wrapped.append(lora)
        else:
            wrapped.extend(add_lora(child))
    return wrapped


class CrossViewAttention(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.norm = Normalize(c)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(c, c, 1) for _ in range(4))

    @staticmethod
    def _attend(q, k, v):
        b, c, h, w = q.shape
        q, k, v = (t.flatten(2).transpose(1, 2).unsqueeze(1) for t in (q, k, v))
        return F.scaled_dot_product_attention(q, k, v).squeeze(1).transpose(1, 2).reshape(b, c, h, w)

    def forward(self, h_l, h_r):
        n_l, n_r = self.norm(h_l), self.norm(h_r)
        delta_l = self.proj_out(self._attend(self.q(n_l), self.k(n_r), self.v(n_r)))
        delta_r = self.proj_out(self._attend(self.q(n_r), self.k(n_l), self.v(n_l)))
        return h_l + delta_l, h_r + delta_r


class StereoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(2 * Z_CHANNELS, 2 * Z_CHANNELS, 1)
        self.post_quant_conv = nn.Conv2d(Z_CHANNELS, Z_CHANNELS, 1)
        self.lora = add_lora(self.decoder)
        self.mixer = CrossViewAttention(CH * CH_MULT[-1])

    def encode(self, x):
        return torch.chunk(self.quant_conv(self.encoder(x)), 2, dim=1)[0]

    def decode_stereo(self, z_l, z_r):
        d = self.decoder
        h_l, h_r = d.conv_in(self.post_quant_conv(z_l)), d.conv_in(self.post_quant_conv(z_r))
        h_l = d.mid.block_2(d.mid.attn_1(d.mid.block_1(h_l)))
        h_r = d.mid.block_2(d.mid.attn_1(d.mid.block_1(h_r)))
        h_l, h_r = self.mixer(h_l, h_r)
        for i in reversed(range(len(CH_MULT))):
            for block in d.up[i].block:
                h_l, h_r = block(h_l), block(h_r)
            if i != 0:
                h_l = d.up[i].upsample.conv(F.interpolate(h_l, scale_factor=2.0, mode="nearest"))
                h_r = d.up[i].upsample.conv(F.interpolate(h_r, scale_factor=2.0, mode="nearest"))
        h_l = d.conv_out(nonlinearity(d.norm_out(h_l)))
        h_r = d.conv_out(nonlinearity(d.norm_out(h_r)))
        return h_l, h_r
