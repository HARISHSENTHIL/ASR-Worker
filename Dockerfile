FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libgomp1 \
    gcc \
    g++ \
    make \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .

RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir -r requirements.txt

# Optional cleanup
RUN apt-get purge -y --auto-remove git wget python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip3 cache purge

COPY auditchimp/ /app/auditchimp/
RUN mkdir -p /app/models /app/uploads /app/data /app/logs /app/temp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["python3", "-m", "uvicorn", "auditchimp.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
