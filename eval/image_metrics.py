import sys
from collections import OrderedDict
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from met3r import MEt3R
from PIL import Image
from torchvision import transforms

ISQOE_ROOT = Path(__file__).resolve().parent / "third_party" / "ml-isqoe"
ISQOE_CKPT = ISQOE_ROOT / "checkpoint" / "isqoe_1_1.ckpt"
MET3R_IMG_SIZE = 256
ISQOE_SIZE = (720, 1280)

if str(ISQOE_ROOT) not in sys.path:
    sys.path.insert(0, str(ISQOE_ROOT))
from isqoe.lightning_model import LightningPerceptualModel  # noqa: E402
from isqoe.pretrained_models.feature_extractor import DINOv2  # noqa: E402


def to_tensor(img: Image.Image, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1).to(device)


class ImageMetrics:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.met3r = MEt3R(img_size=MET3R_IMG_SIZE, use_norm=True, backbone="mast3r", feature_backbone="dino16",
                           feature_backbone_weights="mhamilton723/FeatUp:main", upsampler="featup",
                           distance="cosine", freeze=True).to(device).eval()
        isqoe = LightningPerceptualModel(
            feature_extractor=DINOv2(xatten="concat", xatten_layers=[2, 5, 8, 11]),
            dataset_dir=str(ISQOE_ROOT / "dataset" / "scope"), use_lora=True, lora_r=8, lora_alpha=32,
            lora_dropout=0.1)
        state = torch.load(ISQOE_CKPT, map_location="cpu", weights_only=True)
        isqoe.load_state_dict(state if isinstance(state, OrderedDict) else state["state_dict"], strict=True)
        self.isqoe = isqoe.to(device).requires_grad_(False).eval()
        self.isqoe_tf = transforms.Compose([
            transforms.Resize(ISQOE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor()])

    @torch.no_grad()
    def __call__(self, left: Image.Image, right: Image.Image, gt_right: Image.Image = None) -> dict:
        torch.backends.cuda.matmul.allow_tf32 = True
        out = {"isqoe": float(self.isqoe.stereo_perceptual_model(
            self.isqoe_tf(left)[None].to(self.device), self.isqoe_tf(right)[None].to(self.device)).item())}
        if right.size != left.size:
            right = right.resize(left.size, Image.BICUBIC)
        l_t, r_t = to_tensor(left, self.device), to_tensor(right, self.device)
        views = [F.interpolate(x[None], size=(MET3R_IMG_SIZE, MET3R_IMG_SIZE), mode="bilinear",
                               align_corners=False)[0] for x in (l_t, r_t)]
        score, *_ = self.met3r(images=torch.stack(views)[None], return_overlap_mask=False, return_score_map=False)
        out["met3r"] = float(score.item())
        if gt_right is not None:
            gt_t = to_tensor(gt_right.resize(right.size, Image.BICUBIC), self.device)
            out["lpips"] = float(self.lpips(r_t[None], gt_t[None]).item())
        return out
