#!/usr/bin/env python3
# Builds data/benchmarks/sintel from the two official MPI Sintel zips in data/raw/sintel/:
#   MPI-Sintel-complete.zip                  (training/final)
#   MPI-Sintel-depth-training-20150305.zip   (training/depth, training/camdata_left)
import csv
import struct
import zipfile

import cv2
import numpy as np

from metricstereo.config import ROOT

RAW = ROOT / "data" / "raw" / "sintel"
OUT = ROOT / "data" / "benchmarks" / "sintel"
TAG = 202021.25


def read_dpt(buf):
    assert struct.unpack("f", buf[:4])[0] == TAG
    w, h = struct.unpack("ii", buf[4:12])
    return np.frombuffer(buf[12:], dtype=np.float32, count=w * h).reshape(h, w)


def read_cam_k(buf):
    assert struct.unpack("f", buf[:4])[0] == TAG
    return np.frombuffer(buf[4:76], dtype="<f8").reshape(3, 3).copy()


def main():
    images = zipfile.ZipFile(RAW / "MPI-Sintel-complete.zip")
    depth = zipfile.ZipFile(RAW / "MPI-Sintel-depth-training-20150305.zip")
    depth_names = set(depth.namelist())
    frames = sorted(n[len("training/final/"):-4] for n in images.namelist()
                    if n.startswith("training/final/") and n.endswith(".png"))
    frames = [f for f in frames if f"training/depth/{f}.dpt" in depth_names]
    for sub in ("images", "gt_depth", "intrinsics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["idx", "frame"])
        for i, frame in enumerate(frames):
            name = f"{i:04d}"
            img = cv2.imdecode(np.frombuffer(images.read(f"training/final/{frame}.png"), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(OUT / "images" / f"{name}.png"), img)
            np.save(OUT / "gt_depth" / f"{name}.npy", read_dpt(depth.read(f"training/depth/{frame}.dpt")))
            np.save(OUT / "intrinsics" / f"{name}.npy", read_cam_k(depth.read(f"training/camdata_left/{frame}.cam")))
            rows.writerow([i, frame])
    print(f"sintel: {len(frames)} frames -> {OUT}")


if __name__ == "__main__":
    main()
