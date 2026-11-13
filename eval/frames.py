from pathlib import Path

import cv2
import numpy as np


def frame_size(h: int, w: int, short_side: int) -> tuple:
    s = min(short_side, min(h, w)) / min(h, w)
    return int(round(w * s)), int(round(h * s))


def resize(img: np.ndarray, size: tuple) -> np.ndarray:
    shrink = size[0] < img.shape[1] or size[1] < img.shape[0]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(bgr[..., ::-1])
