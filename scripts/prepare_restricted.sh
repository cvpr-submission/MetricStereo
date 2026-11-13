#!/usr/bin/env bash
# Sintel and LayeredFlow may not be redistributed, so they are not in the benchmark zips. This downloads them from
# the official sources into data/raw/ and builds data/benchmarks/{sintel,layeredflow}.
# By running it you accept the MPI Sintel license (http://sintel.is.tue.mpg.de/license) and the LayeredFlow terms
# (https://layeredflow.cs.princeton.edu). Downloads: Sintel 7.2 GB, LayeredFlow 50 GB (test split + val with GT;
# only the test split is used). The raw zips can be deleted afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {  # fetch <url> <file>
    if [[ ! -f "$2" ]]; then
        curl -L --fail --retry 10 --retry-all-errors -C - -o "$2.part" "$1"
        mv "$2.part" "$2"
    fi
}

mkdir -p data/raw/sintel data/raw/layeredflow
fetch http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip data/raw/sintel/MPI-Sintel-complete.zip
fetch http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip \
    data/raw/sintel/MPI-Sintel-depth-training-20150305.zip
python -m scripts.prepare_sintel

if [[ ! -f data/raw/layeredflow/layeredflow.zip ]]; then
    command -v gdown >/dev/null || pip install gdown
    gdown 1EEFp7AE8ZX75ADztP74Mx7VZ6MOymneN -O data/raw/layeredflow/layeredflow.zip
fi
python -m scripts.prepare_layeredflow
