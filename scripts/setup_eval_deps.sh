#!/usr/bin/env bash
# eval deps: FoundationStereo, iSQoE, MEt3R, LPIPS -> eval/third_party/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/eval/third_party"
mkdir -p "$TP"

FS_COMMIT=6e8806816b533e4d13ddbb95ffa907b797060a62
ISQOE_COMMIT=cb02dbb1b2d5e932c0503a40c8f6535d75075bb9
FS_WEIGHTS_URL="https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"
ISQOE_CKPT_URL="https://ml-site.cdn-apple.com/models/isqoe/isqoe_1_1.ckpt"

pip install -r "${ROOT}/requirements-eval.txt"
pip install --no-build-isolation "git+https://github.com/mhamilton723/FeatUp@6b5a6c0e91f75e69194807128dcbc39c3084a30d"
pip install "git+https://github.com/mohammadasim98/met3r@ee0e1752898559e1a3e85e2e151d3edeb9b55f73"

clone() {
    if [[ ! -d "$2/.git" ]]; then
        git clone "$1" "$2"
    fi
    git -C "$2" checkout -q "$3"
}

clone https://github.com/NVlabs/FoundationStereo "${TP}/FoundationStereo" "$FS_COMMIT"
FS_W="${TP}/FoundationStereo/pretrained_models/23-51-11"
if [[ ! -f "${FS_W}/model_best_bp2.pth" ]]; then
    mkdir -p "${TP}/FoundationStereo/pretrained_models"
    pip install gdown
    gdown --folder "$FS_WEIGHTS_URL" -O "${TP}/FoundationStereo/pretrained_models" \
        || echo "gdown failed: download folder 23-51-11 (cfg.yaml + model_best_bp2.pth) from ${FS_WEIGHTS_URL}" \
                "into ${TP}/FoundationStereo/pretrained_models/" >&2
fi
# md5(model_best_bp2.pth) = fed9cbbb6f139e64520153d1f469f698

# newer DINOv2 stores attn_drop/proj_drop as floats
clone https://github.com/apple/ml-isqoe "${TP}/ml-isqoe" "$ISQOE_COMMIT"
FE="${TP}/ml-isqoe/isqoe/pretrained_models/feature_extractor.py"
sed -i 's/block\.attn\.attn_drop\.p,/getattr(block.attn.attn_drop, "p", block.attn.attn_drop),/; s/block\.attn\.proj_drop\.p,/getattr(block.attn.proj_drop, "p", block.attn.proj_drop),/' "$FE"
if [[ ! -f "${TP}/ml-isqoe/checkpoint/isqoe_1_1.ckpt" ]]; then
    mkdir -p "${TP}/ml-isqoe/checkpoint"
    wget -q -O "${TP}/ml-isqoe/checkpoint/isqoe_1_1.ckpt" "$ISQOE_CKPT_URL"
fi

echo "eval dependencies ready in ${TP}"
