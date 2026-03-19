from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import oss2

from medical_ad_ocr_tools.core.settings import get_settings


def upload_file(local_path: Path, request_id: str) -> tuple[str | None, str | None]:
    settings = get_settings()
    if not settings.oss.enabled:
        return None, None
    if not all([settings.oss.endpoint, settings.oss.bucket_name, settings.oss.access_key_id, settings.oss.access_key_secret]):
        raise ValueError("OSS is enabled but endpoint/bucket/access key configuration is incomplete.")

    endpoint = settings.oss.endpoint or ""
    auth = oss2.Auth(settings.oss.access_key_id or "", settings.oss.access_key_secret or "")
    bucket = oss2.Bucket(auth, endpoint, settings.oss.bucket_name or "")
    key = f"{settings.oss.prefix.strip('/')}/{request_id}.jpg"
    bucket.put_object_from_file(key, str(local_path))

    if settings.oss.public_base_url:
        base = settings.oss.public_base_url.rstrip("/")
        url = f"{base}/{quote(key)}"
    else:
        normalized = endpoint if endpoint.startswith("http") else f"https://{endpoint}"
        url = f"{normalized.rstrip('/')}/{quote(key)}"
    return url, key
