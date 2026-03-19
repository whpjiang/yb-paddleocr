from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests

from medical_ad_ocr_tools.core.settings import Settings


def build_request_id(request_id: str | None = None) -> str:
    return request_id or uuid4().hex


def save_upload_bytes(data: bytes, filename: str | None, request_id: str, settings: Settings) -> Path:
    suffix = Path(filename or "").suffix
    if not suffix:
        suffix = ".jpg"
    target = settings.runtime.temp_dir / f"{request_id}{suffix}"
    target.write_bytes(data)
    return target


def download_image(url: str, request_id: str, settings: Settings) -> Path:
    suffix = Path(requests.utils.urlparse(url).path).suffix or ".jpg"
    digest = hashlib.md5(f"{request_id}:{url}".encode("utf-8")).hexdigest()
    target = settings.runtime.temp_dir / f"{digest}{suffix}"
    response = requests.get(url, timeout=settings.runtime.request_timeout_seconds)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image
