import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

FS_ROOT = Path(__file__).resolve().parent / "third_party" / "FoundationStereo"
FS_CKPT = FS_ROOT / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
FS_SHORT_SIDE = 768
FS_VALID_ITERS = 32

if str(FS_ROOT) not in sys.path:
    sys.path.append(str(FS_ROOT))
from core.foundation_stereo import FoundationStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


class FSDisparity:
    def __init__(self, ckpt: Path = FS_CKPT, device: str = "cuda") -> None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cfg = OmegaConf.load(ckpt.parent / "cfg.yaml")
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        cfg["valid_iters"] = FS_VALID_ITERS
        self.model = FoundationStereo(cfg)
        self.model.load_state_dict(torch.load(ckpt, weights_only=False, map_location="cpu")["model"])
        self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        torch.backends.cuda.matmul.allow_tf32 = False
        h, w = left.shape[:2]
        scale = FS_SHORT_SIDE / min(h, w)
        size = (int(w * scale), int(h * scale))
        left = cv2.resize(left, size, interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, size, interpolation=cv2.INTER_LINEAR)
        t_left, t_right = (torch.as_tensor(x).to(self.device).float()[None].permute(0, 3, 1, 2) for x in (left, right))
        padder = InputPadder(t_left.shape, divis_by=32, force_square=False)
        t_left, t_right = (x.contiguous() for x in padder.pad(t_left, t_right))
        with torch.cuda.amp.autocast(True):
            disp = self.model.forward(t_left, t_right, iters=FS_VALID_ITERS, test_mode=True)
        disp = padder.unpad(disp.float()).cpu().numpy().reshape(size[1], size[0])
        if scale != 1.0:
            disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        return disp.astype(np.float32)
