FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV AUDIO_LAB_MODEL_ROOT=/app/final_models \
    AUDIO_LAB_DEVICE=auto \
    AUDIO_LAB_LOAD_MODE=load \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY training /app/training
COPY final_models /app/final_models

RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

EXPOSE 7860

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
