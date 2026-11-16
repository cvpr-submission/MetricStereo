from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "metricstereo.safetensors"

SIZE = 768
STEPS = 25
CFG_START = 3.0  # cosine decay
CFG_END = 1.0
FOV_RAD = 0.9424777960769379
SEED = 0
SYNC_VIEW_NORM = True  # cross-view synchronized normalization (SyncVN); False for per-view-GroupNorm weights
DEPTH_BASELINE_M = 0.10

LATENT_CHANNELS = 4
PLUCKER_CHANNELS = 6
SCALE_FACTOR = 0.18215
