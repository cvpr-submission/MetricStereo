#   docker build -t metricstereo .
#   docker run --gpus all -v $PWD/weights:/app/weights -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs \
#       metricstereo python -m scripts.check_reproduction
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git wget curl unzip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-eval.txt ./
RUN pip install -r requirements.txt && pip install -r requirements-eval.txt
COPY . .
RUN bash scripts/setup_eval_deps.sh
