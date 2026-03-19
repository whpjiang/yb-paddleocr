from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests

from medical_ad_ocr_tools.core.settings import Settings

logger = logging.getLogger(__name__)


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


def remove_temp_file(path: Path | None, settings: Settings) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
        temp_dir = settings.runtime.temp_dir.resolve()
        if resolved.parent != temp_dir or not resolved.exists():
            return
        resolved.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("remove_temp_file failed path=%s error=%s", path, exc)


def remove_annotated_file(path: Path | None, settings: Settings) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
        output_dir = settings.runtime.output_dir.resolve()
        if resolved.parent != output_dir or not resolved.exists():
            return
        resolved.unlink()
    except Exception as exc:  # noqa: BLE001
        logger.warning("remove_annotated_file failed path=%s error=%s", path, exc)


def cleanup_expired_temp_files(settings: Settings, max_age_hours: int = 24) -> int:
    temp_dir = settings.runtime.temp_dir
    if not temp_dir.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in temp_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("cleanup_expired_temp_files failed path=%s error=%s", path, exc)
    return removed


def cleanup_expired_annotated_files(settings: Settings, max_age_hours: int = 24) -> int:
    output_dir = settings.runtime.output_dir
    if not output_dir.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("cleanup_expired_annotated_files failed path=%s error=%s", path, exc)
    return removed
