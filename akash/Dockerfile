FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN pip install --no-cache-dir transformers datasets wandb numpy tqdm

WORKDIR /app

COPY src/experiments/akash_fromscratch.py /app/src/experiments/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "src.experiments.akash_fromscratch"]