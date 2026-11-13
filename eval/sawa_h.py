"""SAWA-H (EvalMDE, arXiv 2510.19814). WKDR pairs use a seeded generator."""

from math import floor
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

W_WKDR, W_DELTA_DISP_AF, W_DELTA_DEPTH_AF, W_BOUNDARY, W_REL_NORMAL = 3.65, 0.18, 0.01, 0.20, 1.94
WKDR_PAIRS = int(1e7)
REL_NORMAL_SCALES = (1, 2, 4, 8)
REL_NORMAL_PAIRS = int(1e6)
REL_NORMAL_RADIUS, REL_NORMAL_MIN_RADIUS = 32, 3
FOV_DEG = 60.0


def wkdr_error(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, gen: torch.Generator) -> float:
    p, t = pred[valid], gt[valid]
    n = p.shape[0]
    i = torch.randint(0, n, (WKDR_PAIRS,), device=p.device, generator=gen)
    j = torch.randint(0, n, (WKDR_PAIRS,), device=p.device, generator=gen)
    return 1.0 - float(((p[i] < p[j]) == (t[i] < t[j])).float().mean().item())


def delta0125_acc(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    ratio = torch.maximum(gt / pred.clamp_min(1e-10), pred / gt.clamp_min(1e-10))
    ratio[~valid] = 99999.0
    return float((ratio < (1.25 ** 0.125)).float().sum().item() / valid.float().sum().item())


def align_depth_lstsq(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = pred[valid].cpu().numpy().reshape(-1, 1)
    g = gt[valid].cpu().numpy().reshape(-1, 1)
    scale, shift = np.linalg.lstsq(np.concatenate([p, np.ones_like(p)], axis=-1), g, rcond=None)[0].ravel()
    return (pred * float(scale) + float(shift)).clamp_min(eps)


def align_disparity_affine(pred_disp: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor,
                           eps: float = 1e-4) -> torch.Tensor:
    x = pred_disp[valid].double()
    y = (1.0 / gt[valid]).double()
    ab = torch.linalg.lstsq(torch.stack([x, torch.ones_like(x)], dim=-1), y[:, None])[0].squeeze(-1)
    return 1.0 / (pred_disp * float(ab[0].item()) + float(ab[1].item())).clamp_min(eps)


def _edge_ratios(depth: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    disp = 1.0 / torch.clamp(torch.where(valid, depth, torch.full_like(depth, float("nan"))), min=1e-6)
    return (disp[:, :-1] / disp[:, 1:], disp[:-1, :] / disp[1:, :],
            disp[:, 1:] / disp[:, :-1], disp[1:, :] / disp[:-1, :])


def boundary_f1_error(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    thresholds = torch.linspace(1.05, 1.25, 10, device=pred.device).double()
    t = thresholds.view(-1, 1, 1)
    recalls, precisions = [], []
    for pr, gr in zip(_edge_ratios(pred, valid), _edge_ratios(gt, valid)):
        pred_pos = pr.double().unsqueeze(0) > t
        gt_pos = gr.double().unsqueeze(0) > t
        tp = (pred_pos & gt_pos).sum(dim=(1, 2))
        recalls.append(tp / torch.clamp(gt_pos.sum(dim=(1, 2)), min=1))
        precisions.append(tp / torch.clamp(pred_pos.sum(dim=(1, 2)), min=1))
    recall, precision = 0.25 * sum(recalls), 0.25 * sum(precisions)
    f1 = 2 * recall * precision / torch.clamp(recall + precision, min=1e-12)
    return float(1.0 - (f1 * thresholds / thresholds.sum()).sum().item())


def _depth_to_xyz(depth: torch.Tensor, fov_deg: float) -> torch.Tensor:
    h, w = depth.shape
    fx = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    return torch.stack([(u - cx) / fx * depth, (v - cy) / fx * depth, depth], dim=-1)


def _uv_grid(h: int, w: int, **kw) -> torch.Tensor:
    v, u = torch.meshgrid(torch.arange(h, **kw), torch.arange(w, **kw), indexing="ij")
    return torch.stack([u, v], dim=-1)


def _pad(x: torch.Tensor, sc: int) -> torch.Tensor:
    h, w, c_shape = x.shape[0], x.shape[1], x.shape[2:]
    x = x.reshape(h, w, -1).permute(2, 0, 1)
    x = F.pad(x, (0, (sc - w % sc) % sc, 0, (sc - h % sc) % sc), value=0)
    return x.permute(1, 2, 0).reshape((x.shape[-2], x.shape[-1]) + c_shape)


def _patchify(x: torch.Tensor, sc: int) -> torch.Tensor:
    h, w, c_shape = x.shape[0], x.shape[1], x.shape[2:]
    x = x.reshape(h // sc, sc, w // sc, sc, -1).permute(1, 3, 0, 2, 4)
    return x.reshape((sc, sc, h // sc, w // sc) + c_shape)


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    sc, h, w, c_shape = x.shape[0], x.shape[2], x.shape[3], x.shape[4:]
    x = x.reshape(sc * sc, h, w, -1)
    return torch.gather(x, 0, idx[None, :, :, None].expand(1, h, w, x.shape[-1])).reshape((h, w) + c_shape)


def _downsample(sc: int, valid: torch.Tensor, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    kw = dict(device=valid.device, dtype=torch.float)
    h, w = valid.shape
    uv = _patchify(_pad(_uv_grid(h, w, **kw), sc), sc)
    centre = _uv_grid(uv.shape[2], uv.shape[3], **kw) * sc + 0.5 * (sc - 1)
    valid_p = _patchify(_pad(valid, sc), sc)
    dist = (uv - centre[None, None]).norm(dim=-1)
    dist[~valid_p] = torch.inf
    idx = torch.argmin(dist.reshape(-1, dist.shape[-2], dist.shape[-1]), dim=0)
    return [_gather(valid_p, idx)] + [_gather(_patchify(_pad(t, sc), sc), idx) for t in tensors]


def _triangle_normals(xyz: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    a, b, c = xyz[:-2, :-2], xyz[2:, :-2], xyz[:-2, 2:]
    n = torch.linalg.cross(F.normalize(b - a, dim=-1), F.normalize(c - a, dim=-1), dim=-1)
    n_norm = torch.norm(n, dim=-1)
    normal = n / n_norm.clamp(min=1e-5).unsqueeze(-1)
    ok = (n_norm > 1e-5) & valid[:-2, :-2] & valid[2:, :-2] & valid[:-2, 2:]
    return normal, ok


def _angle(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
    return torch.acos((F.normalize(n1, dim=-1) * F.normalize(n2, dim=-1)).sum(dim=-1).clamp(-1, 1))


def _pair_pixels(h: int, w: int, num: int, radius: int, device) -> Tuple[torch.Tensor, ...]:
    radius = min(radius, max(h, w))
    out = torch.empty((4, num), dtype=torch.long, device=device)
    n = 0
    engine = torch.quasirandom.SobolEngine(4)
    while n < num:
        s = engine.draw(floor(num * 1.1)).to(device)
        s[:, 0] *= h
        s[:, 1] *= w
        s[:, 2:] = s[:, 2:] * (radius * 2) - radius
        pts = torch.floor(torch.cat([s[:, :2], s[:, :2] + s[:, 2:]], dim=1))
        ok = (pts[:, [0, 2]] < h).all(-1) & (pts[:, [1, 3]] < w).all(-1) & (pts >= 0).all(-1)
        pts = pts[ok]
        m = min(len(pts), num - n)
        out[:, n:n + m] = pts[:m].T.long()
        n += m
    return tuple(out)


def _rel_normal_at_scale(gt_xyz, gt_valid, pred_xyz, pred_valid, radius: int) -> Optional[float]:
    gt_n, gt_nv = _triangle_normals(gt_xyz, gt_valid)
    pred_n, pred_nv = _triangle_normals(pred_xyz, pred_valid)
    i1, j1, i2, j2 = _pair_pixels(gt_n.shape[0], gt_n.shape[1], REL_NORMAL_PAIRS, radius, gt_xyz.device)
    err = torch.abs(_angle(gt_n[i1, j1], gt_n[i2, j2]) - _angle(pred_n[i1, j1], pred_n[i2, j2]))
    gt_ok = gt_nv[i1, j1] & gt_nv[i2, j2]
    pred_ok = pred_nv[i1, j1] & pred_nv[i2, j2]
    err = torch.where(gt_ok & ~pred_ok, torch.pi, err)[gt_ok]
    return float(err.mean().item()) if err.numel() else None


def rel_normal(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> float:
    gt_xyz, pred_xyz = _depth_to_xyz(gt, FOV_DEG), _depth_to_xyz(pred, FOV_DEG)
    errs = []
    for sc in REL_NORMAL_SCALES:
        gv, gx, pv, px = _downsample(sc, valid, [gt_xyz, valid, pred_xyz])
        e = _rel_normal_at_scale(gx, gv, px, pv, radius=max(REL_NORMAL_RADIUS // sc, REL_NORMAL_MIN_RADIUS))
        if e is not None:
            errs.append(e)
    return float(np.mean(errs)) if errs else 0.0


@torch.no_grad()
def sawa_h(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor, gen: torch.Generator) -> float:
    valid = mask & torch.isfinite(pred_depth) & (pred_depth > 0) & torch.isfinite(gt_depth) & (gt_depth > 0)
    if valid.sum() == 0:
        raise ValueError("no valid pixels")
    rel_n = rel_normal(pred_depth, gt_depth, valid)
    wkdr = wkdr_error(pred_depth, gt_depth, valid, gen)
    delta_disp = 1.0 - delta0125_acc(align_disparity_affine(1.0 / pred_depth.clamp_min(1e-4), gt_depth, valid),
                                     gt_depth, valid)
    delta_depth = 1.0 - delta0125_acc(align_depth_lstsq(pred_depth, gt_depth, valid), gt_depth, valid)
    boundary = boundary_f1_error(pred_depth, gt_depth, valid)
    return (W_WKDR * wkdr + W_DELTA_DISP_AF * delta_disp + W_DELTA_DEPTH_AF * delta_depth
            + W_BOUNDARY * boundary + W_REL_NORMAL * rel_n)
