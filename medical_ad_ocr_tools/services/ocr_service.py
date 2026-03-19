from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from paddleocr import PaddleOCR

from medical_ad_ocr_tools.core.models import FocusRegion, OCRBlock, OCRRecord, OCRRunResult, RuleEvaluationResponse
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.card_detector import detect_card_candidates
from medical_ad_ocr_tools.services.rule_service import evaluate_blocks, get_rule_config


@dataclass
class OCRCandidate:
    name: str
    image: np.ndarray
    offset_x: int
    offset_y: int
    crop_width: int
    crop_height: int
    scale: float
    rotation: int = 0


@dataclass
class FocusRegionCandidate:
    bbox: tuple[int, int, int, int]
    score: float
    block_indices: list[int]


@dataclass
class Round1Analysis:
    evaluation: RuleEvaluationResponse
    box_count: int
    total_text_length: int
    high_risk_keyword_hits: int
    has_complete_contact: bool
    has_partial_contact: bool
    has_card_candidate: bool


@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "lang": settings.ocr.lang,
        "ocr_version": settings.ocr.version,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": settings.ocr.use_textline_orientation,
        "text_det_limit_side_len": settings.ocr.text_det_limit_side_len,
        "text_det_box_thresh": settings.ocr.text_det_box_thresh,
        "text_det_thresh": settings.ocr.text_det_thresh,
        "text_det_unclip_ratio": settings.ocr.text_det_unclip_ratio,
        "text_rec_score_thresh": settings.ocr.text_rec_score_thresh,
    }
    if settings.ocr.text_detection_model_name:
        kwargs["text_detection_model_name"] = settings.ocr.text_detection_model_name
    if settings.ocr.text_recognition_model_name:
        kwargs["text_recognition_model_name"] = settings.ocr.text_recognition_model_name
    if settings.ocr.text_detection_model_dir:
        kwargs["text_detection_model_dir"] = str(settings.ocr.text_detection_model_dir)
    if settings.ocr.text_recognition_model_dir:
        kwargs["text_recognition_model_dir"] = str(settings.ocr.text_recognition_model_dir)
    if settings.ocr.textline_orientation_model_dir:
        kwargs["textline_orientation_model_dir"] = str(settings.ocr.textline_orientation_model_dir)
    return PaddleOCR(**kwargs)


def _make_candidate(
    crop: np.ndarray,
    name: str,
    offset_x: int,
    offset_y: int,
    *,
    scale: float = 1.0,
    rotation: int = 0,
) -> OCRCandidate:
    processed = crop.copy()
    if scale != 1.0:
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if rotation == 90:
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == -90:
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return OCRCandidate(
        name=name,
        image=processed,
        offset_x=offset_x,
        offset_y=offset_y,
        crop_width=crop.shape[1],
        crop_height=crop.shape[0],
        scale=scale,
        rotation=rotation,
    )


def _build_initial_candidates(image: np.ndarray) -> list[OCRCandidate]:
    candidates = [_make_candidate(image, "full", 0, 0)]
    if image.shape[0] > image.shape[1] * 1.2:
        top_crop = image[: int(image.shape[0] * 0.82), :]
        candidates.append(_make_candidate(top_crop, "top_crop", 0, 0))
    return candidates


def _map_points(candidate: OCRCandidate, points: list[list[float]]) -> list[list[int]]:
    mapped: list[list[int]] = []
    for point in points:
        scaled_x = point[0] / candidate.scale
        scaled_y = point[1] / candidate.scale
        if candidate.rotation == 90:
            crop_x, crop_y = scaled_y, candidate.crop_height - scaled_x
        elif candidate.rotation == -90:
            crop_x, crop_y = candidate.crop_width - scaled_y, scaled_x
        else:
            crop_x, crop_y = scaled_x, scaled_y
        mapped.append([int(round(crop_x + candidate.offset_x)), int(round(crop_y + candidate.offset_y))])
    return mapped


def _bbox(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = min(max(0, bbox[0]), width - 1)
    y1 = min(max(0, bbox[1]), height - 1)
    x2 = min(max(x1 + 1, bbox[2]), width)
    y2 = min(max(y1 + 1, bbox[3]), height)
    return x1, y1, x2, y2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    inter_left = max(a[0], b[0])
    inter_top = max(a[1], b[1])
    inter_right = min(a[2], b[2])
    inter_bottom = min(a[3], b[3])
    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0
    inter = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _center_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], threshold: int = 24) -> bool:
    return abs((a[0] + a[2]) - (b[0] + b[2])) <= threshold and abs((a[1] + a[3]) - (b[1] + b[3])) <= threshold


def _normalize_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _similar_text(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 3:
        return False
    overlap = sum(1 for left, right in zip(short, long) if left == right)
    return overlap / max(1, len(short)) >= 0.65


def _prefer_record(candidate: OCRRecord, existing: OCRRecord) -> bool:
    candidate_text = _normalize_text(candidate.text)
    existing_text = _normalize_text(existing.text)
    candidate_score = (len(candidate_text), candidate.confidence)
    existing_score = (len(existing_text), existing.confidence)
    return candidate_score > existing_score


def _deduplicate(records: list[OCRRecord]) -> list[OCRRecord]:
    deduped: list[OCRRecord] = []
    for record in sorted(records, key=lambda item: item.confidence, reverse=True):
        current_bbox = _bbox(record.points)
        current_text = _normalize_text(record.text)
        duplicate_index = -1
        for index, kept in enumerate(deduped):
            kept_bbox = _bbox(kept.points)
            kept_text = _normalize_text(kept.text)
            if current_text and kept_text and current_text == kept_text and (_iou(current_bbox, kept_bbox) >= 0.25 or _center_close(current_bbox, kept_bbox)):
                duplicate_index = index
                break
            if _similar_text(current_text, kept_text) and (_iou(current_bbox, kept_bbox) >= 0.5 or _center_close(current_bbox, kept_bbox)):
                duplicate_index = index
                break
        if duplicate_index >= 0:
            if _prefer_record(record, deduped[duplicate_index]):
                deduped[duplicate_index] = record
            continue
        deduped.append(record)
    return deduped


def _as_points(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [[float(point[0]), float(point[1])] for point in value]


def _iter_result_items(result: Any) -> Iterable[tuple[list[list[float]], str, float]]:
    if result is None:
        return []
    if isinstance(result, dict) or hasattr(result, "keys"):
        keys = result.keys() if hasattr(result, "keys") else ()
        if "dt_polys" in keys and "rec_texts" in keys and "rec_scores" in keys:
            return [
                (_as_points(points), str(text).strip(), float(score))
                for points, text, score in zip(result["dt_polys"], result["rec_texts"], result["rec_scores"])
            ]
    items: list[tuple[list[list[float]], str, float]] = []
    for page in result or []:
        for item in page or []:
            points, rec = item
            items.append((_as_points(points), str(rec[0]).strip(), float(rec[1])))
    return items


def _predict_candidate(engine: PaddleOCR, candidate: OCRCandidate, use_textline_orientation: bool) -> list[OCRRecord]:
    records: list[OCRRecord] = []
    result = engine.predict(
        candidate.image,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=use_textline_orientation,
    )
    pages = result if len(result) != 1 else [result[0]]
    for page in pages:
        for points, text, confidence in _iter_result_items(page):
            if text:
                records.append(OCRRecord(points=_map_points(candidate, points), text=text, confidence=confidence))
    return records


def _records_to_blocks(records: list[OCRRecord]) -> list[OCRBlock]:
    return [OCRBlock(text=item.text, points=item.points, confidence=item.confidence) for item in records]


def _has_partial_contact(evaluation: RuleEvaluationResponse) -> bool:
    if evaluation.phones or evaluation.wechat_ids:
        return False
    phone_hint = re.compile(r"1[3-9][0-9\-\s]{5,}")
    wechat_hint = re.compile(r"(微信|vx|V[Xx]|wechat)", re.IGNORECASE)
    qq_hint = re.compile(r"\b(QQ|qq)\b")
    return any(phone_hint.search(block.text) or wechat_hint.search(block.text) or qq_hint.search(block.text) for block in evaluation.ocr_blocks)


def _build_round1_analysis(image: np.ndarray, records: list[OCRRecord], request_id: str) -> Round1Analysis:
    evaluation = evaluate_blocks(blocks=_records_to_blocks(records), image=image, request_id=request_id)
    return Round1Analysis(
        evaluation=evaluation,
        box_count=len(evaluation.ocr_blocks),
        total_text_length=sum(len(block.text.strip()) for block in evaluation.ocr_blocks),
        high_risk_keyword_hits=len(evaluation.hit_keywords),
        has_complete_contact=bool(evaluation.phones or evaluation.wechat_ids),
        has_partial_contact=_has_partial_contact(evaluation),
        has_card_candidate=bool(detect_card_candidates(image)),
    )


def should_run_focus_retry(round1_analysis: Round1Analysis) -> str:
    settings = get_settings()
    if not settings.ocr.focus_retry_enabled:
        return ""
    evaluation = round1_analysis.evaluation
    if round1_analysis.high_risk_keyword_hits >= settings.ocr.focus_retry_min_keyword_hits and not round1_analysis.has_complete_contact:
        return "high_risk_keywords_missing_contact"
    if (
        round1_analysis.box_count <= 4
        and round1_analysis.total_text_length <= 36
        and round1_analysis.high_risk_keyword_hits >= 1
    ):
        return "sparse_text_with_risk_words"
    if (
        settings.ocr.focus_retry_mid_risk_min <= evaluation.risk_score <= settings.ocr.focus_retry_mid_risk_max
        and round1_analysis.has_card_candidate
    ):
        return "mid_risk_with_card_region"
    return ""


def _cluster_boxes(boxes: list[OCRBlock]) -> list[list[int]]:
    clusters: list[list[int]] = []
    used: set[int] = set()
    bboxes = [_bbox(block.points) for block in boxes]

    for index, box in enumerate(bboxes):
        if index in used:
            continue
        cluster = {index}
        queue = [index]
        while queue:
            current = queue.pop()
            cx1, cy1, cx2, cy2 = bboxes[current]
            current_center_x = (cx1 + cx2) / 2
            current_center_y = (cy1 + cy2) / 2
            current_w = max(20, cx2 - cx1)
            current_h = max(20, cy2 - cy1)
            limit_x = max(70, int(current_w * 2.4))
            limit_y = max(120, int(current_h * 4.8))
            for other_index, other in enumerate(bboxes):
                if other_index in cluster:
                    continue
                ox1, oy1, ox2, oy2 = other
                other_center_x = (ox1 + ox2) / 2
                other_center_y = (oy1 + oy2) / 2
                if abs(other_center_x - current_center_x) <= limit_x and abs(other_center_y - current_center_y) <= limit_y:
                    cluster.add(other_index)
                    queue.append(other_index)
        clusters.append(sorted(cluster))
        used.update(cluster)
    return clusters


def _score_focus_region(
    image: np.ndarray,
    boxes: list[OCRBlock],
    block_indices: list[int],
) -> FocusRegionCandidate | None:
    config = get_rule_config()
    selected = [boxes[index] for index in block_indices]
    merged = (
        min(_bbox(block.points)[0] for block in selected),
        min(_bbox(block.points)[1] for block in selected),
        max(_bbox(block.points)[2] for block in selected),
        max(_bbox(block.points)[3] for block in selected),
    )
    padding_x = max(config.min_region_padding, int((merged[2] - merged[0]) * config.region_padding_x_ratio))
    padding_y = max(config.min_region_padding, int((merged[3] - merged[1]) * config.region_padding_y_ratio))
    bbox = _clip_bbox((merged[0] - padding_x, merged[1] - padding_y, merged[2] + padding_x, merged[3] + padding_y), image.shape)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 48 or height < 48:
        return None

    keyword_hits = sum(len(block.hit_keywords) for block in selected)
    contact_hits = sum(sum(1 for item in block.clue_types if item in {"phone", "wechat", "qq"}) for block in selected)
    if contact_hits == 0:
        contact_hits = sum(1 for block in selected if re.search(r"(微信|vx|V[Xx]|wechat|QQ|qq|手机|电话)", block.text, re.IGNORECASE))
    text_area = sum(max(1, (_bbox(block.points)[2] - _bbox(block.points)[0]) * (_bbox(block.points)[3] - _bbox(block.points)[1])) for block in selected)
    region_area = max(1, width * height)
    density = min(1.0, text_area / region_area * 2.5)
    aspect_ratio = width / max(height, 1)
    shape_score = max(0.0, 1.0 - min(abs(np.log(max(aspect_ratio, 1e-6))) / 1.2, 1.0))
    line_score = min(1.0, len(selected) / 4.0)
    keyword_score = min(1.0, keyword_hits / 3.0)
    contact_score = min(1.0, contact_hits / 3.0)
    score = round(keyword_score * 0.35 + contact_score * 0.25 + density * 0.15 + shape_score * 0.10 + line_score * 0.15, 4)
    return FocusRegionCandidate(bbox=bbox, score=score, block_indices=block_indices)


def select_best_focus_region(image: np.ndarray, boxes: list[OCRBlock], analysis: Round1Analysis) -> FocusRegion | None:
    settings = get_settings()
    candidates: list[FocusRegionCandidate] = []
    for cluster in _cluster_boxes(boxes):
        candidate = _score_focus_region(image, boxes, cluster)
        if candidate is None:
            continue
        if analysis.high_risk_keyword_hits and not any(boxes[index].hit_keywords for index in cluster):
            continue
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    candidates = candidates[: settings.ocr.focus_retry_max_regions]
    if not candidates:
        return None
    best = candidates[0]
    if best.score < settings.ocr.focus_retry_min_region_score:
        return None
    return FocusRegion(x1=best.bbox[0], y1=best.bbox[1], x2=best.bbox[2], y2=best.bbox[3], score=best.score)


def build_focus_retry_candidate(image: np.ndarray, focus_region: FocusRegion) -> OCRCandidate | None:
    settings = get_settings()
    bbox = _clip_bbox((focus_region.x1, focus_region.y1, focus_region.x2, focus_region.y2), image.shape)
    crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if crop.size == 0:
        return None

    processed = crop.copy()
    if settings.ocr.focus_retry_enable_contrast:
        processed = cv2.convertScaleAbs(processed, alpha=1.15, beta=4)
    if settings.ocr.focus_retry_enable_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5.0, -1], [0, -1, 0]], dtype=np.float32)
        processed = cv2.filter2D(processed, -1, kernel)

    rotation = 0
    if settings.ocr.focus_retry_enable_rotate and crop.shape[0] > crop.shape[1] * 1.6:
        rotation = 90
    return _make_candidate(
        processed,
        name="focus_retry",
        offset_x=bbox[0],
        offset_y=bbox[1],
        scale=settings.ocr.focus_retry_scale,
        rotation=rotation,
    )


def merge_round1_and_focus(round1_boxes: list[OCRRecord], focus_boxes: list[OCRRecord]) -> list[OCRRecord]:
    merged = list(round1_boxes)
    for focus_box in focus_boxes:
        focus_bbox = _bbox(focus_box.points)
        duplicate_index = -1
        for index, existing in enumerate(merged):
            existing_bbox = _bbox(existing.points)
            if _iou(focus_bbox, existing_bbox) >= 0.45 or _center_close(focus_bbox, existing_bbox, threshold=30):
                duplicate_index = index
                break
        if duplicate_index >= 0:
            if _prefer_record(focus_box, merged[duplicate_index]):
                merged[duplicate_index] = focus_box
            continue
        merged.append(focus_box)
    return _deduplicate(merged)


def run_ocr(image_path: str | Path, request_id: str = "ocr-analyze") -> OCRRunResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    settings = get_settings()
    engine = get_ocr_engine()

    round1_boxes: list[OCRRecord] = []
    for candidate in _build_initial_candidates(image):
        round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
    round1_boxes = _deduplicate(round1_boxes)

    round1_analysis = _build_round1_analysis(image, round1_boxes, request_id)
    retry_reason = should_run_focus_retry(round1_analysis)
    focus_region = select_best_focus_region(image, round1_analysis.evaluation.ocr_blocks, round1_analysis) if retry_reason else None

    final_boxes = list(round1_boxes)
    focus_added_boxes = 0
    if retry_reason and focus_region is not None:
        retry_candidate = build_focus_retry_candidate(image, focus_region)
        if retry_candidate is not None:
            focus_boxes = _predict_candidate(engine, retry_candidate, settings.ocr.use_textline_orientation)
            merged = merge_round1_and_focus(round1_boxes, focus_boxes)
            focus_added_boxes = max(0, len(merged) - len(round1_boxes))
            final_boxes = merged
        else:
            retry_reason = ""
            focus_region = None
    elif retry_reason:
        retry_reason = ""

    final_boxes.sort(key=lambda item: (min(point[1] for point in item.points), min(point[0] for point in item.points)))
    avg_confidence = round(sum(item.confidence for item in final_boxes) / len(final_boxes), 4) if final_boxes else 0.0
    return OCRRunResult(
        records=final_boxes,
        avg_confidence=avg_confidence,
        ocr_text="\n".join(item.text for item in final_boxes),
        round1_triggered_focus_retry=bool(retry_reason and focus_region is not None),
        focus_retry_reason=retry_reason,
        focus_region=focus_region,
        focus_retry_added_boxes=focus_added_boxes,
    )
