# MetricStereo

Inference code for MetricStereo: left image + baseline (m) -> right view.
Weights: [cvprsubmission/metricstereo-model](https://huggingface.co/cvprsubmission/metricstereo-model)

Settings are fixed to the ones used in the paper: 768x768, DPM-Solver++ 25 steps, CFG 3->1 (cosine), 54 deg FOV,
seed = image index. The U-Net uses cross-view synchronized normalization (SyncVN, `SYNC_VIEW_NORM` in
`metricstereo/config.py`).

## Install

```bash
pip install torch==2.5.1 torchvision==0.20.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/download_weights.sh
```

or `docker build -t metricstereo .`

## Quick start

```python
from PIL import Image
from metricstereo.pipeline import MetricStereo, read_rgb

model = MetricStereo()
right = model.generate(read_rgb("left.png"), baseline_m=0.065, seed=0)
Image.fromarray(right).save("right.png")
```

## Benchmarks

Seven benchmarks are downloaded preprocessed. Sintel and LayeredFlow do not allow redistribution, so they are
downloaded from the official sources and preprocessed locally:

```bash
bash scripts/download_data.sh            # 7 benchmarks, ~14 GB (or e.g. `bash scripts/download_data.sh middlebury`)
bash scripts/prepare_restricted.sh       # Sintel + LayeredFlow from the official sources (downloads ~57 GB)
python -m scripts.check_reproduction     # 2 Middlebury images vs tests/reference/, bit-exact on A100
python -m scripts.generate               # all 9 benchmarks -> outputs/  (~6.5 h, A100)
```

`prepare_restricted.sh` fetches the Sintel training set and depth maps (MPI, 7.2 GB) and the LayeredFlow
benchmark (Google Drive, 50 GB) into `data/raw/`, then:

- **Sintel:** 1064 frames of the `final` pass with their `.dpt` metric depth and `.cam` intrinsics.
- **LayeredFlow:** the 1200 test pairs (300 scenes x 4), rectified with the shipped calibration. The per-pair
  baseline from the stereo extrinsics goes to `manifest.csv`.

Images, right views, depth and baselines are identical to the files used in the paper. Sintel intrinsics are
the per-frame values from the `.cam` files; they are not used by `generate` or `eval` (fixed 54° field of view).
`data/raw/` can be deleted afterwards.

| | Booster | LayeredFlow | Middlebury | Spring | KITTI | Sintel | ETH3D | TUM | NYUv2 |
|---|---|---|---|---|---|---|---|---|---|
| pairs | 191 | 1200 | 45 | 1000 | 161 | 1064 | 454 | 1815 | 654 |
| source | zip 3.9G | official | zip 0.6G | zip 6.2G | zip 0.2G | official | zip 1.4G | zip 0.8G | zip 0.4G |
| license | CC BY-NC 4.0 | no license (all rights reserved) | use + publish | CC BY 4.0 | CC BY-NC-SA 3.0 | MPI PS:License 1.0 | CC BY-NC-SA 4.0 | CC BY 4.0 | none stated |

Layout: `data/benchmarks/<name>/{images,right,gt_depth,intrinsics,manifest.csv}`. Each benchmark keeps its
original license; please cite the original datasets.

## Evaluation

```bash
bash scripts/setup_eval_deps.sh
python -m eval.run_eval                  # -> outputs/eval_common_s768/summary.json  (~7 h, A100)
```

Expected:

```
dataset        delta1      rel   sawa_h    lpips    met3r    isqoe
booster        0.9684   0.0413   0.8378   0.3097   0.0887   0.6319
layeredflow    0.8755   0.0814   1.1245   0.4001   0.1216   0.6565
middlebury     0.8818   0.0857   0.9133   0.3408   0.1005   0.6740
spring         0.7840   0.1191   1.4005   0.1394   0.0942   0.7848
kitti          0.9617   0.0571   1.3437        -   0.0690   0.7214
sintel         0.7019   0.2263   1.6260        -   0.1494   0.8216
eth3d          0.9271   0.0846   1.1209        -   0.0666   0.5522
tum            0.9109   0.0902   1.6667        -   0.0905   0.7824
nyu            0.9642   0.0582   1.4668        -   0.1005   0.7419
```

## Per-scene baseline search (stereo benchmarks)

The main comparison of the paper tunes the baseline of every method per scene against the real right view, following
StereoSpace: a coarse search over 2^-2 ... 2^2 times the benchmark's median baseline, refined by 2^(+-0.25) around the
best value, keeps the value whose SGBM disparity on the generated pair best matches the SGBM disparity of the real
pair; every image of the scene is then generated at that value and scored with the protocol above.

```bash
python -m scripts.scene_search                         # Booster, LayeredFlow, Middlebury, Spring (~22 h, A100)
python -m scripts.scene_search --datasets middlebury   # one benchmark (~1 h)
```

Expected (`outputs/scene_search/eval/summary.json`):

```
dataset        delta1      rel   sawa_h    lpips    met3r    isqoe
booster        0.9674   0.0420   0.8263   0.2578   0.0999   0.6445
layeredflow    0.8798   0.0802   1.1204   0.3395   0.1382   0.6730
middlebury     0.8820   0.0852   0.9041   0.2447   0.0958   0.6638
spring         0.7906   0.1158   1.4121   0.1292   0.0920   0.7846
stereo         0.8799   0.0808   1.0657   0.2428   0.1065   0.6915
```

The expected numbers above were produced on NVIDIA RTX PRO 6000 (Blackwell) GPUs, and on that GPU this code reproduces
them bit-exactly (identical generated images and per-scene selections). Diffusion sampling is not
bit-reproducible across GPU architectures: on an A100 the generated images differ slightly, and the per-scene search on
Middlebury changed each metric by at most 0.004 (LPIPS by 0.0002). `check_reproduction` is bit-exact on A100.
