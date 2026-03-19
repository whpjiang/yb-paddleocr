from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException

from medical_ad_ocr_tools.core.models import HealthResponse, ImageSourcePayload, RuleEvaluateRequest
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.image_io import (
    build_request_id,
    cleanup_expired_annotated_files,
    cleanup_expired_temp_files,
    download_image,
    remove_annotated_file,
    remove_temp_file,
)
from medical_ad_ocr_tools.services.ocr_service import get_ocr_engine
from medical_ad_ocr_tools.services.rule_service import evaluate_request, get_rule_config
from medical_ad_ocr_tools.services.workflow import analyze_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()
app = FastAPI(title=settings.app.name, version=settings.app.version)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@app.on_event("startup")
def preload_models() -> None:
    rule_config = get_rule_config()
    removed_temp_files = cleanup_expired_temp_files(settings)
    removed_annotated_files = cleanup_expired_annotated_files(settings)
    logger.info(
        "startup config rules_path=%s suspicious_score=%s phone_only_floor=%s git_commit=%s version=%s temp_cleanup_removed=%s annotated_cleanup_removed=%s",
        settings.rules_path,
        rule_config.suspicious_score,
        rule_config.scores.get("phone_only_floor"),
        _git_commit(),
        settings.app.version,
        removed_temp_files,
        removed_annotated_files,
    )
    get_ocr_engine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app.name, version=settings.app.version)


@app.post("/tools/medical-ad/rule/evaluate")
async def rule_evaluate(payload: RuleEvaluateRequest) -> dict:
    try:
        result = evaluate_request(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Rule evaluation failed: {exc}") from exc
    return result.model_dump()


@app.post("/tools/medical-ad/ocr/analyze")
async def analyze(payload: ImageSourcePayload) -> dict:
    image_path: Path | None = None
    annotated_image_path: Path | None = None
    try:
        request_id = build_request_id(payload.request_id)
        image_path = download_image(str(payload.image_url), request_id, settings)
        image_source = str(payload.image_url)
        result = analyze_image(request_id=request_id, image_path=Path(image_path), image_source=image_source)
        annotated_image_path = Path(result.artifacts.annotated_image_path) if result.artifacts.annotated_image_path else None
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"OCR analyze failed: {exc}") from exc
    finally:
        remove_annotated_file(annotated_image_path, settings)
        remove_temp_file(image_path, settings)
    return result.response.model_dump()
