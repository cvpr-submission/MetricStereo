import numpy as np
import torch

from .config import CFG_END, CFG_START, LATENT_CHANNELS


def cfg_at_step(i, n):
    frac = i / (n - 1)
    return CFG_END + (CFG_START - CFG_END) * np.cos(frac * np.pi / 2) ** 2


def logsnr_timesteps(alphas_cumprod, steps):
    ac = np.clip(alphas_cumprod.detach().float().cpu().numpy(), 1e-8, 1.0 - 1e-8)
    lam = 0.5 * np.log(ac / (1.0 - ac))
    grid = np.arange(len(ac))
    ts = np.interp(np.linspace(lam[1], lam[-1], steps), lam[::-1], grid[::-1])
    ts = np.unique(np.round(ts).astype(np.int64))
    return ts[ts >= 1]


class Sampler:
    def __init__(self, unet, alphas_cumprod, sqrt_ac, sqrt_1m_ac):
        self.unet = unet
        self.ac, self.sqrt_ac, self.sqrt_1m_ac = alphas_cumprod, sqrt_ac, sqrt_1m_ac

    def _x0(self, x, t, context, camera, cond, cfg):
        c = LATENT_CHANNELS
        out = self.unet(x, t, context, camera=camera, num_frames=x.shape[0])
        if cfg == 1.0:
            return self.sqrt_ac[t].view(-1, 1, 1, 1) * x[:, :c] - self.sqrt_1m_ac[t].view(-1, 1, 1, 1) * out
        uncond_x = x.clone()
        uncond_x[:, -1] = 0.0
        uncond_x[:, c:-1] = 0.0
        noise = torch.randn(uncond_x[cond, :c].shape, device=x.device)
        uncond_x[cond, :c] = (self.sqrt_ac[t[cond]].view(-1, 1, 1, 1) * uncond_x[cond, :c]
                              + self.sqrt_1m_ac[t[cond]].view(-1, 1, 1, 1) * noise).to(uncond_x.dtype)
        out_uncond = self.unet(uncond_x, t, torch.zeros_like(context), camera=None, num_frames=1)
        v = out_uncond + cfg * (out - out_uncond)
        return self.sqrt_ac[t].view(-1, 1, 1, 1) * x[:, :c] - self.sqrt_1m_ac[t].view(-1, 1, 1, 1) * v

    @torch.no_grad()
    def sample(self, x, cond, context, camera, steps):
        c = LATENT_CHANNELS
        ts = np.flip(logsnr_timesteps(self.ac, steps))
        targets = list(ts[1:]) + [0]
        n = int(ts.shape[0])
        img, clean = x.clone(), x.clone()
        x0_prev = lam_prev = None
        for i, (t_idx, s_idx) in enumerate(zip(ts, targets)):
            t = torch.full((x.shape[0],), t_idx, device=x.device, dtype=torch.long)
            a_t, s_t = self.ac[t_idx].sqrt(), (1 - self.ac[t_idx]).sqrt()
            a_s, s_s = self.ac[s_idx].sqrt(), (1 - self.ac[s_idx]).sqrt()
            lam_t, lam_s = torch.log(a_t / s_t).item(), torch.log(a_s / s_s).item()
            h = lam_s - lam_t
            x0 = self._x0(img, t, context, camera, cond, cfg_at_step(i, n))
            last = i == n - 1
            if i == 0 or last:
                d = x0
            else:
                r = (lam_t - lam_prev) / h
                d = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev
            new = d if last else (s_s / s_t).to(x.device) * img[:, :c] - (a_s * np.expm1(-h)).to(x.device) * d
            img = img.clone()
            img[:, :c] = new
            img[cond] = clean[cond]
            x0_prev, lam_prev = x0, lam_t
        return img[:, :c]
