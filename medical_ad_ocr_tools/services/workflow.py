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

    ocr_result = run_ocr(image_path, request_id=request_id)
    evaluation = evaluate_blocks(blocks=_records_to_blocks(ocr_result.records), image=image, request_id=request_id)
    evaluation.ocr_text = ocr_result.ocr_text

    annotated_image_path = draw_annotations(image_path=image_path, request_id=request_id, evaluation=evaluation)
    annotated_image_url, annotated_image_oss_key = (None, None)
    if annotated_image_path is not None:
        annotated_image_url, annotated_image_oss_key = upload_file(annotated_image_path, request_id=request_id)

    response = AnalyzeResponse(
        request_id=request_id,
        image_source=image_source,
        ocr_text=evaluation.ocr_text,
        ocr_confidence=ocr_result.avg_confidence,
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
        round1_triggered_focus_retry=ocr_result.round1_triggered_focus_retry,
        focus_retry_reason=ocr_result.focus_retry_reason,
        focus_region=ocr_result.focus_region,
        focus_retry_added_boxes=ocr_result.focus_retry_added_boxes,
        focus_retry_variant=ocr_result.focus_retry_variant,
        focus_retry_semantic_score=ocr_result.focus_retry_semantic_score,
        round1_low_semantic_confidence=ocr_result.round1_low_semantic_confidence,
        selected_by_semantic_score=ocr_result.selected_by_semantic_score,
        focus_region_angle=ocr_result.focus_region_angle,
        focus_region_shape=ocr_result.focus_region_shape,
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
