FROM python:3.10.20-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    MEDICAL_AD_CONFIG_PATH=config/config.yaml

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/storage/tmp /app/storage/annotated /app/storage/logs /app/models \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "medical_ad_ocr_tools.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
