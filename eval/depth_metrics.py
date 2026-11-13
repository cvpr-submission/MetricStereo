from typing import Optional

import cv2
import numpy as np
import torch

from .sawa_h import sawa_h

MIN_DEPTH = 1e-3
MIN_VALID_PIXELS = 100


def disparity_to_depth(disp: np.ndarray) -> np.ndarray:
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 0)
    depth[valid] = 1.0 / disp[valid]
    return depth


def _match_size(pred_disp: np.ndarray, shape: tuple) -> np.ndarray:
    if pred_disp.shape == shape:
        return pred_disp
    return cv2.resize(pred_disp, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def delta1_absrel(pred_disp: np.ndarray, gt_depth: np.ndarray, max_depth: float,
                  device: str = "cuda") -> Optional[dict]:
    # MoGe disparity_affine_invariant
    torch.backends.cuda.matmul.allow_tf32 = False  # met3r turns it on
    pred_disp = _match_size(pred_disp.astype(np.float32), gt_depth.shape)
    mask_np = np.isfinite(gt_depth) & (gt_depth > MIN_DEPTH) & (gt_depth < max_depth)
    if mask_np.sum() < MIN_VALID_PIXELS:
        return None
    gt = torch.from_numpy(gt_depth).to(device)
    pred = torch.from_numpy(pred_disp).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    x, y = pred[mask], 1 / gt[mask]
    a, b = torch.linalg.lstsq(torch.stack([x, torch.ones_like(x)], dim=-1), y[:, None])[0].squeeze(-1).unbind(-1)
    pred_depth = 1 / (pred * a + b).clamp_min(1 / gt[mask].max().item())
    p, g = pred_depth[mask], gt[mask]
    return {"rel": (torch.abs(p - g) / (g + 1e-6)).mean().item(),
            "delta1": (torch.maximum(g / p, p / g) < 1.25).float().mean().item()}


def sawa_h_score(pred_disp: np.ndarray, gt_depth: np.ndarray, gen: torch.Generator,
                 device: str = "cuda") -> Optional[float]:
    torch.backends.cuda.matmul.allow_tf32 = False
    pred_depth = disparity_to_depth(_match_size(pred_disp.astype(np.float32), gt_depth.shape))
    mask = np.isfinite(gt_depth) & (gt_depth > 0)
    if mask.sum() < MIN_VALID_PIXELS:
        return None
    return sawa_h(torch.from_numpy(pred_depth).to(device), torch.from_numpy(gt_depth).to(device),
                  torch.from_numpy(mask).to(device), gen)
