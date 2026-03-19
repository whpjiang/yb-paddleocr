from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from medical_ad_ocr_tools.core.models import HealthResponse, ImageSourcePayload, RuleEvaluateRequest
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.image_io import build_request_id, download_image
from medical_ad_ocr_tools.services.ocr_service import get_ocr_engine
from medical_ad_ocr_tools.services.rule_service import evaluate_request
from medical_ad_ocr_tools.services.workflow import analyze_image

settings = get_settings()
app = FastAPI(title=settings.app.name, version=settings.app.version)


@app.on_event("startup")
def preload_models() -> None:
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
    try:
        request_id = build_request_id(payload.request_id)
        image_path = download_image(str(payload.image_url), request_id, settings)
        image_source = str(payload.image_url)
        result = analyze_image(request_id=request_id, image_path=Path(image_path), image_source=image_source)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"OCR analyze failed: {exc}") from exc
    return result.response.model_dump()
