#!/usr/bin/env python3
"""Per-scene baseline search on the four stereo benchmarks (the main protocol of the paper, after StereoSpace).

For every scene, the baseline is scaled by k = 2^e around the benchmark's median ground-truth baseline:
  coarse  e in {-2, -1.5, ..., 2} on up to two search images per scene,
  fine    e = best_coarse +- 0.25,
and the e whose SGBM disparity on (left, generated right) best matches the SGBM disparity of the real pair (RMSE,
mean over the scene's search images) is kept. Every image of the scene is then generated at that e and evaluated
with eval.run_eval (common protocol).

  python -m scripts.scene_search                         # all four benchmarks (~22 h on one A100)
  python -m scripts.scene_search --datasets middlebury   # one benchmark
  python -m scripts.scene_search --stage coarse          # stop after a stage (coarse | fine | final)

Outputs: outputs/scene_search/{gen/e<e>/<ds>/<idx>.png, scores.json, selection_*.json, final/<ds>/<idx>/001.png,
eval/summary.json}. Resumable: generated images and scores are reused.
"""
import argparse
import csv
import json
import math
import statistics
from multiprocessing import get_context

import cv2
import numpy as np
from PIL import Image

from eval.frames import frame_size, resize
from metricstereo.config import ROOT, SEED

STEREO = ["booster", "layeredflow", "middlebury", "spring"]
DATA = ROOT / "data" / "benchmarks"
WORK = ROOT / "outputs" / "scene_search"
COARSE = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
FINE_DELTA = 0.25
SEARCH_PER_SCENE = 2
SHORT_SIDE = 768
DISP_FRAC = 0.3
MIN_VALID = 500


def etag(e):
    return f"{e:+.2f}"


def even_pick(items, m):
    n = len(items)
    return list(items) if n <= m else [items[int((i + 0.5) * n / m)] for i in range(m)]


def make_plan(datasets):
    plan = {}
    for ds in datasets:
        with open(DATA / ds / "manifest.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        scenes = {}
        for r in rows:
            scenes.setdefault(r["scene"], []).append(f"{int(r['idx']):04d}")
        plan[ds] = {"median": statistics.median(float(r["baseline_m"]) for r in rows if r.get("baseline_m")),
                    "scenes": {s: {"all": v, "search": even_pick(v, SEARCH_PER_SCENE)} for s, v in scenes.items()}}
    return plan


def gen_path(ds, stem, e):
    return WORK / "gen" / f"e{etag(e)}" / ds / f"{stem}.png"


def generate(model, plan, jobs):
    from metricstereo.pipeline import read_rgb
    jobs = [j for j in jobs if not gen_path(*j).exists()]
    for n, (ds, stem, e) in enumerate(jobs):
        b = float(f"{2.0 ** e * plan[ds]['median']:.6f}")
        out = gen_path(ds, stem, e)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(model.generate(read_rgb(DATA / ds / "images" / f"{stem}.png"), b, SEED + int(stem))).save(out)
        if (n + 1) % 50 == 0 or n + 1 == len(jobs):
            print(f"  generated {n + 1}/{len(jobs)}", flush=True)


def sgbm(left_bgr, right_bgr):
    gl, gr = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    nd = max(16, int(math.ceil(DISP_FRAC * gl.shape[1] / 16.0)) * 16)
    bs = 5
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                              disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                              mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = m.compute(gl, gr).astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def frame_left(ds, stem):
    left = cv2.imread(str(DATA / ds / "images" / f"{stem}.png"), cv2.IMREAD_COLOR)
    size = frame_size(*left.shape[:2], SHORT_SIDE)
    return resize(left, size), size


def real_disp(job):
    ds, stem = job
    path = WORK / "sgbm_real" / ds / f"{stem}.npy"
    if not path.exists():
        left, size = frame_left(ds, stem)
        right = cv2.imread(str(DATA / ds / "right" / f"{stem}.png"), cv2.IMREAD_COLOR)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, sgbm(left, resize(right, size)).astype(np.float16))


def score_job(job):
    ds, stem, e = job
    cv2.setNumThreads(2)
    d_real = np.load(WORK / "sgbm_real" / ds / f"{stem}.npy").astype(np.float32)
    left, size = frame_left(ds, stem)
    d_gen = sgbm(left, resize(cv2.imread(str(gen_path(ds, stem, e)), cv2.IMREAD_COLOR), size))
    valid = np.isfinite(d_real) & np.isfinite(d_gen)
    rmse = float(np.sqrt(np.mean((d_gen[valid] - d_real[valid]) ** 2))) if valid.sum() >= MIN_VALID else None
    return ds, stem, etag(e), rmse


def score(plan, jobs, workers):
    path = WORK / "scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}
    jobs = [j for j in jobs if etag(j[2]) not in scores.get(j[0], {}).get(j[1], {})]
    with get_context("spawn").Pool(workers) as pool:  # no fork after CUDA init (deadlocks the workers)
        pool.map(real_disp, sorted({(ds, s) for ds, s, _ in jobs}))
        for ds, stem, et, rmse in pool.imap_unordered(score_job, jobs, chunksize=4):
            scores.setdefault(ds, {}).setdefault(stem, {})[et] = rmse
    path.write_text(json.dumps(scores, indent=1))
    return scores


def select(plan, scores, allowed):
    """Per scene: the e with the lowest mean SGBM RMSE over its search images (ties -> e closest to 0)."""
    sel = {}
    for ds, d in plan.items():
        sel[ds] = {}
        for name, sc in d["scenes"].items():
            curve = {}
            for et in sorted({k for s in sc["search"] for k in scores.get(ds, {}).get(s, {})}, key=float):
                if allowed is not None and et not in allowed:
                    continue
                vals = [scores[ds][s][et] for s in sc["search"] if et in scores[ds].get(s, {})]
                vals = [v for v in vals if v is not None and np.isfinite(v)]
                if len(vals) == len(sc["search"]):
                    curve[et] = float(np.mean(vals))
            best = min(curve, key=lambda et: (curve[et], abs(float(et)))) if curve else etag(0.0)
            sel[ds][name] = {"e": float(best), "k": 2.0 ** float(best), "rmse": curve.get(best)}
    return sel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=STEREO, choices=STEREO)
    p.add_argument("--stage", default="eval", choices=["coarse", "fine", "final", "eval"], help="stop after")
    p.add_argument("--workers", type=int, default=12, help="CPU workers for SGBM")
    a = p.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    plan = make_plan(a.datasets)
    from metricstereo.pipeline import MetricStereo
    model = MetricStereo()

    print("coarse", flush=True)
    jobs = [(ds, s, e) for ds, d in plan.items() for sc in d["scenes"].values() for s in sc["search"] for e in COARSE]
    generate(model, plan, jobs)
    sel = select(plan, score(plan, jobs, a.workers), {etag(e) for e in COARSE})
    (WORK / "selection_coarse.json").write_text(json.dumps(sel, indent=1))
    if a.stage == "coarse":
        return

    print("fine", flush=True)
    jobs = [(ds, s, sel[ds][n]["e"] + dd) for ds, d in plan.items() for n, sc in d["scenes"].items()
            for s in sc["search"] for dd in (-FINE_DELTA, FINE_DELTA)]
    generate(model, plan, jobs)
    scores = score(plan, [(ds, s, e) for ds, d in plan.items() for sc in d["scenes"].values() for s in sc["search"]
                          for e in COARSE] + jobs, a.workers)
    sel = select(plan, scores, None)
    (WORK / "selection_fine.json").write_text(json.dumps(sel, indent=1))
    if a.stage == "fine":
        return

    print("final", flush=True)
    jobs = [(ds, s, sel[ds][n]["e"]) for ds, d in plan.items() for n, sc in d["scenes"].items() for s in sc["all"]]
    generate(model, plan, jobs)
    for ds, s, e in jobs:  # evaluation layout: final/<ds>/<idx>/001.png
        dst = WORK / "final" / ds / s / "001.png"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(gen_path(ds, s, e).resolve())
    for ds, d in plan.items():
        ks = [v["k"] for v in sel[ds].values()]
        print(f"  {ds:12s} scenes {len(ks):4d}  mean k {np.mean(ks):.3f}")
    if a.stage == "final":
        return

    del model
    from eval.run_eval import evaluate
    evaluate(gen_root=WORK / "final", out_root=WORK / "eval", datasets=a.datasets)


if __name__ == "__main__":
    main()
