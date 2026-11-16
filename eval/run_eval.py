import json
from pathlib import Path
from statistics import median

import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

from .depth_metrics import delta1_absrel, disparity_to_depth, sawa_h_score
from .foundation_stereo import FSDisparity
from .frames import frame_size, read_rgb, resize
from .image_metrics import ImageMetrics

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_ROOT = REPO_ROOT / "outputs"
BENCH_ROOT = REPO_ROOT / "data" / "benchmarks"
OUT_ROOT = REPO_ROOT / "outputs" / "eval_common_s768"
CACHE_ROOT = REPO_ROOT / "cache" / "fs_pseudo_gt_s768"

STEREO_DATASETS = ["booster", "layeredflow", "middlebury", "spring"]
# stereo benchmarks with public ground-truth disparity; on the others (test splits without public ground truth) the
# reference is FoundationStereo on the real pair in the same frame
GT_DISPARITY_DATASETS = ["middlebury"]
DEPTH_DATASETS = ["kitti", "sintel", "eth3d", "tum", "nyu"]
MAIN_DATASETS = STEREO_DATASETS + ["kitti", "sintel"]
SHORT_SIDE = 768
MAX_DEPTH = {"kitti": 72.0, "sintel": 72.0, "eth3d": 72.0, "tum": 10.0, "nyu": 10.0}
STEREO_MAX_DEPTH = 1e4
METRICS = ["delta1", "rel", "sawa_h", "lpips", "met3r", "isqoe"]
SAWA_SEED = 0
DEVICE = "cuda"


def list_samples(gen_dir: Path) -> list:
    dirs = sorted((d for d in gen_dir.iterdir() if d.is_dir() and d.name.isdigit() and (d / "001.png").exists()),
                  key=lambda d: int(d.name))
    return [(f"{int(d.name):04d}", d / "001.png") for d in dirs]


def cached_disparity(path: Path, fs: FSDisparity, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, fs(left, right))
    return np.load(path)


def gt_disparity_in_frame(path: Path, size: tuple) -> np.ndarray:
    """Ground-truth disparity resized to the frame (nearest neighbor) and rescaled to frame pixels, invalid -> NaN."""
    g = np.load(path).astype(np.float32)
    g[~np.isfinite(g) | (g <= 0)] = np.nan
    return cv2.resize(g, size, interpolation=cv2.INTER_NEAREST) * (size[0] / g.shape[1])


def evaluate_dataset(ds: str, fs: FSDisparity, image_metrics: ImageMetrics, gen_root: Path, bench_root: Path,
                     out_root: Path, cache_root: Path) -> dict:
    stereo = ds in STEREO_DATASETS
    bench, out = bench_root / ds, out_root / ds
    gen = torch.Generator(device=DEVICE).manual_seed(SAWA_SEED)
    per_sample = {}
    for name, gen_path in tqdm(list_samples(gen_root / ds), desc=ds):
        left_path = bench / "images" / f"{name}.png"
        if not left_path.exists():
            print(f"WARN: no benchmark left image for {ds}/{name}")
            continue
        left = read_rgb(left_path)
        right_gt = read_rgb(bench / "right" / f"{name}.png") if stereo else None
        size = frame_size(*left.shape[:2], SHORT_SIDE)
        left_c = resize(left, size)

        pred_disp = cached_disparity(out / "pred_disp" / f"{name}.npy", fs, left_c, resize(read_rgb(gen_path), size))
        if ds in GT_DISPARITY_DATASETS:
            gt_depth = disparity_to_depth(gt_disparity_in_frame(bench / "gt_disparity" / f"{name}.npy", size))
        elif stereo:
            gt_depth = disparity_to_depth(cached_disparity(cache_root / ds / f"{name}.npy", fs, left_c,
                                                           resize(right_gt, size)))
        else:
            gt_depth = np.load(bench / "gt_depth" / f"{name}.npy").astype(np.float32)
        entry = delta1_absrel(pred_disp, gt_depth, STEREO_MAX_DEPTH if stereo else MAX_DEPTH[ds], DEVICE) or {}
        sawa = sawa_h_score(pred_disp, gt_depth, gen, DEVICE)
        if sawa is not None:
            entry["sawa_h"] = sawa

        right_img = Image.open(gen_path).convert("RGB")
        w, h = right_img.size
        left_n = Image.fromarray(resize(left, (w, h)))
        gt_n = Image.fromarray(resize(right_gt, (w, h))) if stereo else None
        entry.update(image_metrics(left_n, right_img, gt_n))
        per_sample[name] = entry

    def agg(metric: str, fn) -> float:
        vals = [e[metric] for e in per_sample.values() if metric in e and not np.isnan(e[metric])]
        return float(fn(vals)) if vals else None

    result = {m: agg(m, np.mean) for m in METRICS}
    if stereo:
        result["rel"] = agg("rel", median)
    result.update(n_samples=len(per_sample), per_sample=per_sample)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate(gen_root: Path = GEN_ROOT, bench_root: Path = BENCH_ROOT, out_root: Path = OUT_ROOT,
             cache_root: Path = CACHE_ROOT, datasets: list = STEREO_DATASETS + DEPTH_DATASETS) -> dict:
    fs, image_metrics = FSDisparity(device=DEVICE), ImageMetrics(device=DEVICE)
    results = {}
    for ds in datasets:
        if not (gen_root / ds).is_dir():
            print(f"WARN: {gen_root / ds} missing - skipping {ds}")
            continue
        r = evaluate_dataset(ds, fs, image_metrics, gen_root, bench_root, out_root, cache_root)
        results[ds] = {m: r[m] for m in METRICS} | {"n_samples": r["n_samples"]}
    for avg_name, sets in [("avg6", MAIN_DATASETS), ("stereo", STEREO_DATASETS), ("depth", DEPTH_DATASETS)]:
        results[avg_name] = {}
        for m in METRICS:
            vals = [results[ds][m] for ds in sets if ds in results and results[ds][m] is not None]
            results[avg_name][m] = float(np.mean(vals)) if vals else None
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"{'dataset':<12}" + "".join(f"{m:>9}" for m in METRICS))
    for ds, r in results.items():
        print(f"{ds:<12}" + "".join(f"{r[m]:9.4f}" if r[m] is not None else f"{'-':>9}" for m in METRICS))
    return results


if __name__ == "__main__":
    evaluate()
