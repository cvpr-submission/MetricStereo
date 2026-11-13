#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://huggingface.co/cvprsubmission/metricstereo-model/resolve/main/metricstereo.safetensors"
SHA256="40d36117e7ab9f5ef27818a94e4cf2ff37e0c703befe7ed6792ae6ced04847e5"
mkdir -p weights
F=weights/metricstereo.safetensors
if [[ ! -f "$F" ]]; then
    curl -L --fail --retry 10 --retry-all-errors -C - -o "$F.part" "$URL"
    echo "${SHA256}  $F.part" | sha256sum -c -
    mv "$F.part" "$F"
fi
echo "${SHA256}  $F" | sha256sum -c -
