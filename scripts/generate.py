#!/usr/bin/env python3
import csv
import statistics

from PIL import Image

from metricstereo.config import DEPTH_BASELINE_M, ROOT, SEED
from metricstereo.pipeline import MetricStereo, read_rgb

STEREO = ["booster", "layeredflow", "middlebury", "spring"]
DEPTH = ["kitti", "sintel", "eth3d", "tum", "nyu"]
DATA = ROOT / "data" / "benchmarks"
OUT = ROOT / "outputs"


def baseline(benchmark):
    if benchmark in DEPTH:
        return DEPTH_BASELINE_M
    with open(DATA / benchmark / "manifest.csv") as f:
        median = statistics.median(float(r["baseline_m"]) for r in csv.DictReader(f) if r.get("baseline_m"))
    return float(f"{median:.6f}")


def main():
    model = MetricStereo()
    for benchmark in STEREO + DEPTH:
        images = sorted((DATA / benchmark / "images").iterdir())
        b = baseline(benchmark)
        print(f"{benchmark}: {len(images)} images at baseline {b} m", flush=True)
        for i, path in enumerate(images):
            out = OUT / benchmark / f"{i:04d}" / "001.png"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(model.generate(read_rgb(path), b, SEED + i)).save(out)


if __name__ == "__main__":
    main()
