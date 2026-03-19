from __future__ import annotations

from pathlib import Path

import cv2

from medical_ad_ocr_tools.core.models import AnalyzeArtifacts, AnalyzeResponse, OCRBlock, OCRRecord, WorkflowResult
from medical_ad_ocr_tools.services.annotator import draw_annotations
from medical_ad_ocr_tools.services.ocr_service import run_ocr
from medical_ad_ocr_tools.services.oss_uploader import upload_file
from medical_ad_ocr_tools.services.rule_service import evaluate_blocks


def _records_to_blocks(records: list[OCRRecord]) -> list[OCRBlock]:
    return [OCRBlock(text=item.text, points=item.points, confidence=item.confidence) for item in records]


def analyze_image(request_id: str, image_path: Path, image_source: str) -> WorkflowResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    ocr_records, avg_confidence, ocr_text = run_ocr(image_path)
    evaluation = evaluate_blocks(blocks=_records_to_blocks(ocr_records), image=image, request_id=request_id)
    evaluation.ocr_text = ocr_text

    annotated_image_path = draw_annotations(image_path=image_path, request_id=request_id, evaluation=evaluation)
    annotated_image_url, annotated_image_oss_key = (None, None)
    if annotated_image_path is not None:
        annotated_image_url, annotated_image_oss_key = upload_file(annotated_image_path, request_id=request_id)

    response = AnalyzeResponse(
        request_id=request_id,
        image_source=image_source,
        ocr_text=evaluation.ocr_text,
        ocr_confidence=avg_confidence,
        ocr_blocks=evaluation.ocr_blocks,
        phones=evaluation.phones,
        wechat_ids=evaluation.wechat_ids,
        qqs=evaluation.qqs,
        hit_keywords=evaluation.hit_keywords,
        hit_rules=evaluation.hit_rules,
        risk_score=evaluation.risk_score,
        risk_level=evaluation.risk_level,
        suspicious=evaluation.suspicious,
        ads=evaluation.ads,
        annotated_image_url=annotated_image_url,
        annotated_image_oss_key=annotated_image_oss_key,
    )
    artifacts = AnalyzeArtifacts(
        source_path=str(image_path),
        source_label=image_source,
        original_image_shape=tuple(image.shape),
        annotated_image_path=str(annotated_image_path) if annotated_image_path else None,
        annotated_image_url=annotated_image_url,
        annotated_image_oss_key=annotated_image_oss_key,
    )
    return WorkflowResult(response=response, artifacts=artifacts)
