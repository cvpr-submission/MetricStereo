#!/usr/bin/env python3
# Builds data/benchmarks/layeredflow from the official LayeredFlow zip in data/raw/layeredflow/layeredflow.zip.
# Only the test split and the calibration are read (no need to unpack the 50 GB archive). Pairs are rectified
# with the shipped calibration; the baseline comes from the stereo extrinsics.
import csv
import pickle
import zipfile

import cv2
import numpy as np

from metricstereo.config import ROOT

RAW = ROOT / "data" / "raw" / "layeredflow"
OUT = ROOT / "data" / "benchmarks" / "layeredflow"
PREFIX = "public_layeredflow_benchmark/"
DEFAULT_BASELINE_M = 0.142


def load_calib(z, names, calib_id):
    path = f"{PREFIX}calib/{calib_id}/data/calib_dict.pickle"
    if path not in names:
        return None
    d = pickle.loads(z.read(path))
    w, h = d["img_shape"]
    k = d["rectify"]["P1"][:3, :3]

    def maps(cam, r):
        return cv2.initUndistortRectifyMap(d[cam]["K"], d[cam]["d"].flatten(), d["rectify"][r], k, (w, h),
                                           cv2.CV_32FC1)

    return {"k": k, "baseline": abs(float(d["stereo"]["T"][0, 0])), "left": maps("left", "R1"),
            "right": maps("right", "R2")}


def read_bgr(z, name):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)


def main():
    z = zipfile.ZipFile(RAW / "layeredflow.zip")
    names = set(z.namelist())
    scenes = {}
    for n in names:
        parts = n[len(PREFIX):].split("/")
        if n.startswith(PREFIX + "test/") and len(parts) == 3 and parts[2]:
            scenes.setdefault(parts[1], set()).add(parts[2])
    pairs = []
    for scene in sorted(scenes):
        files = scenes[scene]
        calib_file = f"{PREFIX}test/{scene}/calib_id.txt"
        calib_id = z.read(calib_file).decode().strip() if calib_file in names else "0"
        for left in sorted(f for f in files if f.endswith("_0.png")):
            right = left[:-6] + "_1.png"
            if right in files:
                pairs.append((f"{PREFIX}test/{scene}/{left}", f"{PREFIX}test/{scene}/{right}", f"test_{scene}",
                              calib_id))
    for sub in ("images", "right", "intrinsics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    calibs = {}
    with open(OUT / "manifest.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["idx", "scene", "baseline_m"])
        for i, (left_name, right_name, scene, calib_id) in enumerate(pairs):
            if calib_id not in calibs:
                calibs[calib_id] = load_calib(z, names, calib_id)
            c = calibs[calib_id]
            left, right = read_bgr(z, left_name), read_bgr(z, right_name)
            baseline = DEFAULT_BASELINE_M
            if c is not None:
                left = cv2.remap(left, *c["left"], cv2.INTER_LINEAR)
                right = cv2.remap(right, *c["right"], cv2.INTER_LINEAR)
                np.save(OUT / "intrinsics" / f"{i:04d}.npy", c["k"].astype(np.float64))
                baseline = c["baseline"]
            cv2.imwrite(str(OUT / "images" / f"{i:04d}.png"), left)
            cv2.imwrite(str(OUT / "right" / f"{i:04d}.png"), right)
            rows.writerow([i, scene, f"{baseline:.6f}"])
    print(f"layeredflow: {len(pairs)} pairs -> {OUT}")


if __name__ == "__main__":
    main()
