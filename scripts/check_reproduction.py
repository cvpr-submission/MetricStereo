#!/usr/bin/env python3
import numpy as np
from PIL import Image

from scripts.generate import baseline
from metricstereo.config import ROOT, SEED
from metricstereo.pipeline import MetricStereo, read_rgb

CASES = [0, 1]


def main():
    model = MetricStereo()
    images = sorted((ROOT / "data" / "benchmarks" / "middlebury" / "images").iterdir())
    b = baseline("middlebury")
    ok = True
    for i in CASES:
        out = model.generate(read_rgb(images[i]), b, SEED + i)
        ref = np.asarray(Image.open(ROOT / "tests" / "reference" / f"middlebury_{i:04d}_right.png").convert("RGB"))
        diff = np.abs(out.astype(int) - ref.astype(int))
        same = diff.max() == 0
        ok &= same
        print(f"middlebury #{i}: {'IDENTICAL' if same else 'DIFFERENT'} "
              f"(max |diff| {diff.max()}, {100 * (diff > 0).mean():.3f}% of values differ)")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
