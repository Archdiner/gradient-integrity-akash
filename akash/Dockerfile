FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=0

RUN pip install \
        "transformers>=4.40,<4.46" \
        "datasets>=2.18,<3.0" \
        "tokenizers>=0.15,<0.20" \
        "huggingface_hub>=0.22,<1.0" \
        "wandb>=0.17,<0.19" \
        "numpy>=1.26,<2.0" \
        "tqdm>=4.66"

WORKDIR /app

COPY src/__init__.py                         /app/src/__init__.py
COPY src/experiments/__init__.py             /app/src/experiments/__init__.py
COPY src/experiments/akash_fromscratch.py    /app/src/experiments/akash_fromscratch.py
COPY src/defenses/__init__.py                /app/src/defenses/__init__.py
COPY src/defenses/reputation.py              /app/src/defenses/reputation.py

RUN mkdir -p /app/results /app/results/checkpoints

VOLUME ["/app/results"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=300s --retries=3 \
    CMD test -f /app/results/.alive || exit 1

ENTRYPOINT ["python", "-m", "src.experiments.akash_fromscratch"]

CMD ["--model", "small", "--rounds", "200", "--device", "cuda", \
     "--output", "/app/results/run.json"]
