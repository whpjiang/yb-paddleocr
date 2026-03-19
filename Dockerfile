FROM python:3.10.20-slim-bookworm

ARG GITHUB_OWNER=whpjiang
ARG GITHUB_REPO=yb-paddleocr
ARG GIT_REF=main

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

RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's/security.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY . /app

RUN set -eux; \
    fetch_lfs_model() { \
        model_path="$1"; \
        if [ ! -f "$model_path" ]; then \
            return 0; \
        fi; \
        if head -n 1 "$model_path" | grep -q "git-lfs.github.com/spec/v1"; then \
            remote_path="${model_path#/app/}"; \
            remote_url="https://media.githubusercontent.com/media/${GITHUB_OWNER}/${GITHUB_REPO}/${GIT_REF}/${remote_path}"; \
            echo "Downloading real LFS model for ${remote_path} from ${remote_url}"; \
            curl -fL "$remote_url" -o "${model_path}.download"; \
            mv "${model_path}.download" "$model_path"; \
        fi; \
    }; \
    fetch_lfs_model /app/models/ocr/PP-OCRv5_mobile_det/inference.pdiparams; \
    fetch_lfs_model /app/models/ocr/PP-OCRv5_mobile_rec/inference.pdiparams; \
    ! head -n 1 /app/models/ocr/PP-OCRv5_mobile_det/inference.pdiparams | grep -q "git-lfs.github.com/spec/v1"; \
    ! head -n 1 /app/models/ocr/PP-OCRv5_mobile_rec/inference.pdiparams | grep -q "git-lfs.github.com/spec/v1"

RUN mkdir -p /app/storage/tmp /app/storage/annotated /app/storage/logs /app/models \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "medical_ad_ocr_tools.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
