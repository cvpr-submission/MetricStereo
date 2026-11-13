import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from .camera import plucker, stereo_poses
from .clip_image import CLIPImageToken
from .config import FOV_RAD, LATENT_CHANNELS, PLUCKER_CHANNELS, SCALE_FACTOR, SIZE, STEPS, SYNC_VIEW_NORM, WEIGHTS
from .sampler import Sampler
from .unet import TwoViewUNet
from .vae import StereoVAE


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


class MetricStereo:
    def __init__(self, device="cuda", weights=WEIGHTS, sync_view_norm=SYNC_VIEW_NORM):
        self.device = device
        sd = load_file(str(weights))
        self.unet = TwoViewUNet(LATENT_CHANNELS + PLUCKER_CHANNELS + 1, LATENT_CHANNELS, sync_view_norm)
        self.unet.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("unet.")}, strict=True)
        self.vae = StereoVAE()
        self.vae.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("vae.")}, strict=True)
        self.unet.to(device).eval()
        self.vae.to(device).eval()
        self.clip = CLIPImageToken().to(device).eval()
        self.sampler = Sampler(self.unet, *(sd[f"schedule.{k}"].to(device)
                                            for k in ("alphas_cumprod", "sqrt_alphas_cumprod",
                                                      "sqrt_one_minus_alphas_cumprod")))

    def _to_tensor(self, image):
        image = cv2.resize(image, dsize=(SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device).to(torch.bfloat16)
        return (t / 255 - 0.5) * 2

    @torch.inference_mode()
    def generate(self, left_rgb, baseline_m, seed):
        torch.manual_seed(seed)
        left = self._to_tensor(left_rgb)
        w2c = stereo_poses(baseline_m, self.device)
        poses = torch.stack([p.flatten() for p in w2c])
        rays = plucker(poses.view(-1, 4, 4), FOV_RAD, SIZE // 8)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_left = self.vae.encode(left) * SCALE_FACTOR
            x = torch.cat([z_left, torch.randn(z_left.size(), device=self.device)], dim=0)
            cond = torch.tensor([1, 0], dtype=torch.bfloat16, device=self.device).view(-1) == 1
            mask = torch.ones_like(x)[:, :1, ...] * cond.view(-1, 1, 1, 1)
            x = torch.cat([x, rays, mask], dim=1)
            context = self.clip(left.to(self.device, dtype=torch.float32)).mean(dim=0, keepdim=True).repeat(2, 1, 1)
            z = self.sampler.sample(x, cond, context, poses, STEPS) / SCALE_FACTOR
            _, right = self.vae.decode_stereo(z[0:1], z[1:2])
        right = torch.clamp((right + 1.0) / 2.0, min=0.0, max=1.0)
        right = 255.0 * right.permute(0, 2, 3, 1).to(torch.float).cpu().numpy()
        return right[0].astype(np.uint8)
