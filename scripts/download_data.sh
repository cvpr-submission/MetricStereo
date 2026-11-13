#!/usr/bin/env bash
# usage: bash scripts/download_data.sh [benchmark ...]   (set HF_TOKEN if the dataset is gated)
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="https://huggingface.co/datasets/cvprsubmission/metricstereo-data/resolve/main"
ALL=(booster middlebury spring kitti eth3d tum nyu)  # sintel, layeredflow: scripts/prepare_restricted.sh
BENCHMARKS=("${@:-${ALL[@]}}")
AUTH=()
[[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")

mkdir -p data/benchmarks data/zips
curl -sL --fail "${AUTH[@]}" -o data/zips/SHA256SUMS "${BASE}/SHA256SUMS"

for b in "${BENCHMARKS[@]}"; do
    if [[ -f "data/benchmarks/${b}/.complete" ]]; then
        echo "${b}: already present"
        continue
    fi
    echo "${b}: downloading"
    curl -L --fail --retry 10 --retry-all-errors -C - "${AUTH[@]}" -o "data/zips/${b}.zip" "${BASE}/${b}.zip"
    (cd data/zips && grep " ${b}.zip\$" SHA256SUMS | sha256sum -c -)
    unzip -q -o "data/zips/${b}.zip" -d data/benchmarks
    touch "data/benchmarks/${b}/.complete"
    rm "data/zips/${b}.zip"
done
echo "benchmarks ready in data/benchmarks/"
