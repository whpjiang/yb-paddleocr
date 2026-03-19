from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np
from paddleocr import PaddleOCR

from medical_ad_ocr_tools.core.models import FocusRegion, OCRBlock, OCRRecord, OCRRunResult, RuleEvaluationResponse
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.card_detector import CardCandidate, detect_card_candidates
from medical_ad_ocr_tools.services.rule_service import evaluate_blocks, get_rule_config

RetryVariant = Literal["normal", "mirrored", "rotate_90", "rotate_180", "rotate_270", "deskew", "perspective"]
logger = logging.getLogger(__name__)


@dataclass
class OCRCandidate:
    name: str
    variant: RetryVariant | Literal["full", "top_crop"]
    image: np.ndarray
    crop_width: int
    crop_height: int
    scale: float
    offset_x: int = 0
    offset_y: int = 0
    mirror: bool = False
    rotation_quadrants: int = 0
    inverse_affine_matrix: list[list[float]] | None = None
    inverse_perspective_matrix: list[list[float]] | None = None


@dataclass
class FocusRegionSelection:
    bbox: tuple[int, int, int, int]
    score: float
    block_indices: list[int]
    angle: float
    shape: str
    card_candidate: CardCandidate | None = None


@dataclass
class Round1Analysis:
    evaluation: RuleEvaluationResponse
    box_count: int
    total_text_length: int
    high_risk_keyword_hits: int
    has_complete_contact: bool
    has_partial_contact: bool
    has_card_candidate: bool
    card_candidates: list[CardCandidate]
    low_semantic_confidence: bool


@dataclass
class RetryCandidateResult:
    candidate: OCRCandidate
    merged_boxes: list[OCRRecord]
    evaluation: RuleEvaluationResponse
    semantic_score: float
    avg_confidence: float


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


def _make_initial_candidate(crop: np.ndarray, name: Literal["full", "top_crop"], offset_x: int, offset_y: int) -> OCRCandidate:
    return OCRCandidate(
        name=name,
        variant=name,
        image=crop.copy(),
        crop_width=crop.shape[1],
        crop_height=crop.shape[0],
        scale=1.0,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _build_initial_candidates(image: np.ndarray) -> list[OCRCandidate]:
    candidates = [_make_initial_candidate(image, "full", 0, 0)]
    if image.shape[0] > image.shape[1] * 1.15:
        top_crop = image[: int(image.shape[0] * 0.82), :]
        candidates.append(_make_initial_candidate(top_crop, "top_crop", 0, 0))
    return candidates


def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = min(max(0, bbox[0]), width - 1)
    y1 = min(max(0, bbox[1]), height - 1)
    x2 = min(max(x1 + 1, bbox[2]), width)
    y2 = min(max(y1 + 1, bbox[3]), height)
    return x1, y1, x2, y2


def _bbox(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


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


def _normalize_rect_angle(angle: float, width: float, height: float) -> float:
    adjusted = angle
    if width < height:
        adjusted -= 90.0
    while adjusted <= -45.0:
        adjusted += 90.0
    while adjusted > 45.0:
        adjusted -= 90.0
    return adjusted


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
    return (len(candidate_text), candidate.confidence) > (len(existing_text), existing.confidence)


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


def _map_points(candidate: OCRCandidate, points: list[list[float]]) -> list[list[int]]:
    src_points = np.array([[point[0] / candidate.scale, point[1] / candidate.scale] for point in points], dtype=np.float32)
    if candidate.rotation_quadrants == 1:
        x = src_points[:, 0].copy()
        y = src_points[:, 1].copy()
        src_points[:, 0] = y
        src_points[:, 1] = candidate.crop_height - x
    elif candidate.rotation_quadrants == 2:
        src_points[:, 0] = candidate.crop_width - src_points[:, 0]
        src_points[:, 1] = candidate.crop_height - src_points[:, 1]
    elif candidate.rotation_quadrants == 3:
        x = src_points[:, 0].copy()
        y = src_points[:, 1].copy()
        src_points[:, 0] = candidate.crop_width - y
        src_points[:, 1] = x
    if candidate.mirror:
        src_points[:, 0] = candidate.crop_width - src_points[:, 0]
    if candidate.inverse_affine_matrix is not None:
        matrix = np.array(candidate.inverse_affine_matrix, dtype=np.float32)
        src_points = cv2.transform(src_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    elif candidate.inverse_perspective_matrix is not None:
        matrix = np.array(candidate.inverse_perspective_matrix, dtype=np.float32)
        src_points = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    else:
        src_points[:, 0] += candidate.offset_x
        src_points[:, 1] += candidate.offset_y
    return [[int(round(x)), int(round(y))] for x, y in src_points]


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


def _noise_hits_for_text(text: str) -> list[str]:
    config = get_rule_config()
    extra_noise = ["刻章", "办证", "贷款", "发票", "搬家", "租房", "招聘"]
    return [word for word in (config.noise_keywords + extra_noise) if word in text]


def _text_has_partial_contact(text: str) -> bool:
    return bool(re.search(r"(微信|vx|V[Xx]|wechat|QQ|qq|手机|电话|1[3-9][0-9]{5,})", text, re.IGNORECASE))


def _round1_low_semantic_confidence(evaluation: RuleEvaluationResponse) -> bool:
    all_text = "\n".join(block.text for block in evaluation.ocr_blocks)
    noise_hits = _noise_hits_for_text(all_text)
    short_or_odd = sum(
        1
        for block in evaluation.ocr_blocks
        if len(block.text.strip()) <= 2 or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", block.text)
    )
    has_only_contact = bool((evaluation.phones or evaluation.wechat_ids or evaluation.qqs) and not evaluation.hit_keywords)
    fallback_only = set(evaluation.hit_rules) <= {"contact.phone", "contact.wechat", "contact.qq", "fallback.phone_only"} and bool(evaluation.hit_rules)
    return (has_only_contact and (bool(noise_hits) or short_or_odd >= max(1, len(evaluation.ocr_blocks) // 2))) or fallback_only


def _build_round1_analysis(image: np.ndarray, records: list[OCRRecord], request_id: str) -> Round1Analysis:
    evaluation = evaluate_blocks(blocks=_records_to_blocks(records), image=image, request_id=request_id)
    cards = detect_card_candidates(image)
    return Round1Analysis(
        evaluation=evaluation,
        box_count=len(evaluation.ocr_blocks),
        total_text_length=sum(len(block.text.strip()) for block in evaluation.ocr_blocks),
        high_risk_keyword_hits=len(evaluation.hit_keywords),
        has_complete_contact=bool(evaluation.phones or evaluation.wechat_ids or evaluation.qqs),
        has_partial_contact=any(_text_has_partial_contact(block.text) for block in evaluation.ocr_blocks),
        has_card_candidate=bool(cards),
        card_candidates=cards,
        low_semantic_confidence=_round1_low_semantic_confidence(evaluation),
    )


def should_run_focus_retry(round1_analysis: Round1Analysis) -> str:
    settings = get_settings()
    if not settings.ocr.focus_retry_enabled:
        return ""
    evaluation = round1_analysis.evaluation
    has_any_risk = round1_analysis.high_risk_keyword_hits >= 1
    if evaluation.risk_score >= settings.ocr.focus_retry_mid_risk_max and round1_analysis.has_complete_contact and round1_analysis.high_risk_keyword_hits >= 2:
        return ""
    if not has_any_risk and not round1_analysis.has_complete_contact and not round1_analysis.has_card_candidate:
        return ""
    if round1_analysis.high_risk_keyword_hits >= settings.ocr.focus_retry_min_keyword_hits and not round1_analysis.has_complete_contact:
        return "high_risk_keywords_missing_contact"
    if round1_analysis.box_count <= 5 and round1_analysis.total_text_length <= 40 and has_any_risk and round1_analysis.has_card_candidate:
        return "sparse_text_with_risk_words"
    if (
        settings.ocr.focus_retry_mid_risk_min <= evaluation.risk_score <= settings.ocr.focus_retry_mid_risk_max
        and has_any_risk
        and round1_analysis.has_card_candidate
    ):
        return "mid_risk_with_card_region"
    if round1_analysis.has_complete_contact and round1_analysis.has_card_candidate and not has_any_risk and round1_analysis.low_semantic_confidence:
        return "contact_only_low_semantic"
    return ""


def _cluster_boxes(boxes: list[OCRBlock]) -> list[list[int]]:
    clusters: list[list[int]] = []
    used: set[int] = set()
    bboxes = [_bbox(block.points) for block in boxes]
    for index in range(len(bboxes)):
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
            limit_x = max(60, int(current_w * 2.6))
            limit_y = max(90, int(current_h * 4.2))
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


def _shape_from_card(card: CardCandidate | None) -> tuple[float, str]:
    if card is None:
        return 0.0, "rectangle"
    points = np.array(card.box, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    angle = _normalize_rect_angle(float(rect[2]), float(rect[1][0]), float(rect[1][1]))
    ordered = points[np.argsort(points[:, 1])]
    top = ordered[:2][np.argsort(ordered[:2, 0])]
    bottom = ordered[2:][np.argsort(ordered[2:, 0])]
    top_w = np.linalg.norm(top[1] - top[0])
    bottom_w = np.linalg.norm(bottom[1] - bottom[0])
    left_h = np.linalg.norm(bottom[0] - top[0])
    right_h = np.linalg.norm(bottom[1] - top[1])
    width_diff = abs(top_w - bottom_w) / max(top_w, bottom_w, 1.0)
    height_diff = abs(left_h - right_h) / max(left_h, right_h, 1.0)
    if width_diff > 0.15 or height_diff > 0.15:
        return angle, "trapezoid"
    return angle, "rectangle"


def _nearby_block_indices(boxes: list[OCRBlock], bbox: tuple[int, int, int, int]) -> list[int]:
    x1, y1, x2, y2 = bbox
    padding_x = max(12, int((x2 - x1) * 0.14))
    padding_y = max(12, int((y2 - y1) * 0.14))
    expanded = (x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y)
    matched: list[int] = []
    for index, block in enumerate(boxes):
        block_bbox = _bbox(block.points)
        center_x = (block_bbox[0] + block_bbox[2]) / 2
        center_y = (block_bbox[1] + block_bbox[3]) / 2
        if (
            expanded[0] <= center_x <= expanded[2]
            and expanded[1] <= center_y <= expanded[3]
        ) or _iou(expanded, block_bbox) >= 0.08:
            matched.append(index)
    return matched


def _estimate_shape_from_boxes(selected: list[OCRBlock]) -> tuple[float, str]:
    point_cloud = np.array([point for block in selected for point in block.points], dtype=np.float32)
    if len(point_cloud) < 4:
        return 0.0, "irregular"
    rect = cv2.minAreaRect(point_cloud)
    angle = _normalize_rect_angle(float(rect[2]), float(rect[1][0]), float(rect[1][1]))
    width, height = rect[1]
    if min(width, height) <= 1:
        return angle, "irregular"
    aspect = max(width, height) / max(1.0, min(width, height))
    return angle, "rectangle" if aspect <= 4.5 else "irregular"


def _region_score(
    image: np.ndarray,
    boxes: list[OCRBlock],
    block_indices: list[int],
    card_candidates: list[CardCandidate],
    *,
    analysis: Round1Analysis | None = None,
    source: str = "text_cluster",
    preset_bbox: tuple[int, int, int, int] | None = None,
    preset_card: CardCandidate | None = None,
) -> FocusRegionSelection | None:
    selected = [boxes[index] for index in block_indices]
    if preset_bbox is not None:
        bbox = _clip_bbox(preset_bbox, image.shape)
    else:
        merged = (
            min(_bbox(block.points)[0] for block in selected),
            min(_bbox(block.points)[1] for block in selected),
            max(_bbox(block.points)[2] for block in selected),
            max(_bbox(block.points)[3] for block in selected),
        )
        width = max(1, merged[2] - merged[0])
        height = max(1, merged[3] - merged[1])
        padding_x = max(10, int(width * 0.08))
        padding_y = max(12, int(height * 0.10))
        bbox = _clip_bbox((merged[0] - padding_x, merged[1] - padding_y, merged[2] + padding_x, merged[3] + padding_y), image.shape)
    if bbox[2] - bbox[0] < 48 or bbox[3] - bbox[1] < 48:
        return None

    overlap_card = preset_card
    overlap_score = 0.0
    if overlap_card is None:
        for card in card_candidates:
            score = _iou(bbox, card.bbox)
            if score > overlap_score:
                overlap_card = card
                overlap_score = score

    keyword_hits = sum(len(boxes[index].hit_keywords) for index in block_indices)
    contact_hits = sum(sum(1 for item in boxes[index].clue_types if item in {"phone", "wechat", "qq"}) for index in block_indices)
    digit_hints = sum(1 for index in block_indices if re.search(r"\d{6,}", boxes[index].text))
    text_area = sum(max(1, (_bbox(boxes[index].points)[2] - _bbox(boxes[index].points)[0]) * (_bbox(boxes[index].points)[3] - _bbox(boxes[index].points)[1])) for index in block_indices)
    region_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    density = min(1.0, text_area / region_area * 2.8)
    compactness = min(1.0, len(block_indices) / 4.0)
    card_bonus = 0.18 if overlap_score >= 0.2 else 0.0
    angle, shape = _shape_from_card(overlap_card) if overlap_card is not None else _estimate_shape_from_boxes(selected)
    shape_score = {"rectangle": 1.0, "trapezoid": 0.9, "irregular": 0.55}.get(shape, 0.55)
    contact_score = min(1.0, (contact_hits + digit_hints) / 3.0)
    keyword_score = min(1.0, keyword_hits / 3.0)
    area_bonus = 0.0
    if overlap_card is not None:
        card_area = max(1, (overlap_card.bbox[2] - overlap_card.bbox[0]) * (overlap_card.bbox[3] - overlap_card.bbox[1]))
        area_bonus = min(0.12, card_area / max(region_area, 1) * 0.04)
    score = keyword_score * 0.34 + contact_score * 0.24 + density * 0.14 + compactness * 0.12 + shape_score * 0.16 + card_bonus + area_bonus
    if analysis is not None and analysis.low_semantic_confidence and source == "card_candidate":
        score += 0.16
    if analysis is not None and analysis.low_semantic_confidence and source == "text_cluster":
        joined_text = "".join(boxes[index].text for index in block_indices)
        if not keyword_hits and _noise_hits_for_text(joined_text):
            score -= 0.18
    if analysis is not None and analysis.has_complete_contact and not keyword_hits and overlap_card is not None and source == "card_candidate":
        score += 0.08
    score = round(max(0.0, min(1.0, score)), 4)
    return FocusRegionSelection(bbox=bbox, score=min(score, 1.0), block_indices=block_indices, angle=angle, shape=shape, card_candidate=overlap_card)


def select_best_focus_region(image: np.ndarray, boxes: list[OCRBlock], analysis: Round1Analysis) -> FocusRegionSelection | None:
    settings = get_settings()
    candidates: list[FocusRegionSelection] = []
    for cluster in _cluster_boxes(boxes):
        candidate = _region_score(image, boxes, cluster, analysis.card_candidates, analysis=analysis, source="text_cluster")
        if candidate is None:
            continue
        if analysis.high_risk_keyword_hits and not any(boxes[index].hit_keywords for index in cluster) and not any(re.search(r"\d{6,}", boxes[index].text) for index in cluster):
            continue
        candidates.append(candidate)

    for card in analysis.card_candidates:
        nearby_indices = _nearby_block_indices(boxes, card.bbox)
        candidate = _region_score(
            image,
            boxes,
            nearby_indices,
            analysis.card_candidates,
            analysis=analysis,
            source="card_candidate",
            preset_bbox=card.bbox,
            preset_card=card,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    if best.score < settings.ocr.focus_retry_min_region_score:
        return None
    return best


def _enhance_crop(crop: np.ndarray) -> np.ndarray:
    settings = get_settings()
    processed = crop.copy()
    if settings.ocr.focus_retry_enable_contrast:
        processed = cv2.convertScaleAbs(processed, alpha=1.16, beta=4)
    if settings.ocr.focus_retry_enable_sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5.1, -1], [0, -1, 0]], dtype=np.float32)
        processed = cv2.filter2D(processed, -1, kernel)
    return processed


def _expanded_focus_bbox(image: np.ndarray, selection: FocusRegionSelection, variant: RetryVariant) -> tuple[int, int, int, int]:
    bbox = selection.bbox
    if variant in {"rotate_180", "rotate_90", "rotate_270"} and selection.card_candidate is not None:
        card_bbox = selection.card_candidate.bbox
        bbox = (
            min(bbox[0], card_bbox[0]),
            min(bbox[1], card_bbox[1]),
            max(bbox[2], card_bbox[2]),
            max(bbox[3], card_bbox[3]),
        )

    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    if variant == "rotate_180":
        padding_x = max(12, int(width * 0.10))
        padding_top = max(8, int(height * 0.04))
        padding_bottom = max(20, int(height * 0.28))
        expanded = (bbox[0] - padding_x, bbox[1] - padding_top, bbox[2] + padding_x, bbox[3] + padding_bottom)
    elif variant in {"rotate_90", "rotate_270"}:
        padding_x = max(12, int(width * 0.14))
        padding_y = max(12, int(height * 0.14))
        expanded = (bbox[0] - padding_x, bbox[1] - padding_y, bbox[2] + padding_x, bbox[3] + padding_y)
    else:
        expanded = bbox
    return _clip_bbox(expanded, image.shape)


def _build_crop_candidate(image: np.ndarray, bbox: tuple[int, int, int, int], variant: RetryVariant) -> OCRCandidate | None:
    settings = get_settings()
    crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if crop.size == 0:
        return None
    processed = _enhance_crop(crop)
    mirror = variant == "mirrored"
    rotation_quadrants = 0
    if mirror:
        processed = cv2.flip(processed, 1)
    elif variant == "rotate_90":
        processed = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
        rotation_quadrants = 1
    elif variant == "rotate_180":
        processed = cv2.rotate(processed, cv2.ROTATE_180)
        rotation_quadrants = 2
    elif variant == "rotate_270":
        processed = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotation_quadrants = 3
    if settings.ocr.focus_retry_scale != 1.0:
        processed = cv2.resize(processed, None, fx=settings.ocr.focus_retry_scale, fy=settings.ocr.focus_retry_scale, interpolation=cv2.INTER_CUBIC)
    return OCRCandidate(
        name=f"focus_{variant}",
        variant=variant,
        image=processed,
        crop_width=crop.shape[1],
        crop_height=crop.shape[0],
        scale=settings.ocr.focus_retry_scale,
        offset_x=bbox[0],
        offset_y=bbox[1],
        mirror=mirror,
        rotation_quadrants=rotation_quadrants,
    )


def _build_deskew_candidate(image: np.ndarray, selection: FocusRegionSelection) -> OCRCandidate | None:
    settings = get_settings()
    bbox = _clip_bbox(selection.bbox, image.shape)
    crop = image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if crop.size == 0:
        return None
    processed = _enhance_crop(crop)
    center = (processed.shape[1] / 2.0, processed.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -float(selection.angle), 1.0)
    rotated = cv2.warpAffine(processed, matrix, (processed.shape[1], processed.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    inverse = cv2.invertAffineTransform(matrix)
    inverse[:, 2] += np.array([bbox[0], bbox[1]], dtype=np.float32)
    if settings.ocr.focus_retry_scale != 1.0:
        rotated = cv2.resize(rotated, None, fx=settings.ocr.focus_retry_scale, fy=settings.ocr.focus_retry_scale, interpolation=cv2.INTER_CUBIC)
    return OCRCandidate(
        name="focus_deskew",
        variant="deskew",
        image=rotated,
        crop_width=crop.shape[1],
        crop_height=crop.shape[0],
        scale=settings.ocr.focus_retry_scale,
        inverse_affine_matrix=inverse.tolist(),
    )


def _build_perspective_candidate(selection: FocusRegionSelection) -> OCRCandidate | None:
    settings = get_settings()
    card = selection.card_candidate
    if card is None:
        return None
    processed = _enhance_crop(card.crop)
    if settings.ocr.focus_retry_scale != 1.0:
        processed = cv2.resize(processed, None, fx=settings.ocr.focus_retry_scale, fy=settings.ocr.focus_retry_scale, interpolation=cv2.INTER_CUBIC)
    return OCRCandidate(
        name="focus_perspective",
        variant="perspective",
        image=processed,
        crop_width=card.crop.shape[1],
        crop_height=card.crop.shape[0],
        scale=settings.ocr.focus_retry_scale,
        inverse_perspective_matrix=card.inverse_perspective_matrix,
    )


def build_focus_retry_candidate(image: np.ndarray, focus_region: FocusRegionSelection, variant: RetryVariant) -> OCRCandidate | None:
    if variant in {"normal", "mirrored", "rotate_90", "rotate_180", "rotate_270"}:
        return _build_crop_candidate(image, _expanded_focus_bbox(image, focus_region, variant), variant)
    if variant == "deskew":
        return _build_deskew_candidate(image, focus_region)
    if variant == "perspective":
        return _build_perspective_candidate(focus_region)
    return None


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


def _variant_order(selection: FocusRegionSelection, analysis: Round1Analysis) -> list[RetryVariant]:
    settings = get_settings()
    variants: list[RetryVariant] = ["normal"]
    looks_mirrored = analysis.low_semantic_confidence and analysis.has_complete_contact and analysis.high_risk_keyword_hits == 0
    looks_inverted = analysis.low_semantic_confidence and analysis.has_complete_contact and selection.shape in {"rectangle", "trapezoid"}
    angled = settings.ocr.focus_retry_deskew_angle_min <= abs(selection.angle) <= settings.ocr.focus_retry_deskew_angle_max
    trapezoid = selection.shape == "trapezoid"
    region_w = selection.bbox[2] - selection.bbox[0]
    region_h = selection.bbox[3] - selection.bbox[1]
    tall_card = region_h / max(region_w, 1) >= 1.45
    sparse_keywords = analysis.high_risk_keyword_hits <= 1
    vertical_text_bias = tall_card and selection.shape in {"rectangle", "trapezoid"}
    looks_vertical_card = vertical_text_bias and analysis.low_semantic_confidence and (analysis.has_complete_contact or analysis.has_partial_contact or sparse_keywords)
    if looks_mirrored and settings.ocr.focus_retry_enable_mirror:
        variants.append("mirrored")
    if looks_inverted:
        variants.append("rotate_180")
    if looks_vertical_card:
        variants.extend(["rotate_90", "rotate_270"])
    if trapezoid and settings.ocr.focus_retry_enable_perspective:
        variants.append("perspective")
    elif angled:
        variants.append("deskew")
    if looks_mirrored and (angled or trapezoid):
        if settings.ocr.focus_retry_enable_mirror and "mirrored" not in variants:
            variants.append("mirrored")
        if trapezoid and settings.ocr.focus_retry_enable_perspective and "perspective" not in variants:
            variants.append("perspective")
        elif angled and "deskew" not in variants:
            variants.append("deskew")
    deduped: list[RetryVariant] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped[:5]


def _semantic_entities(evaluation: RuleEvaluationResponse) -> dict[str, Any]:
    all_text = "\n".join(block.text for block in evaluation.ocr_blocks)
    return {
        "phones": evaluation.phones,
        "wechat_ids": evaluation.wechat_ids,
        "qqs": evaluation.qqs,
        "hit_rules": evaluation.hit_rules,
        "noise_hits": _noise_hits_for_text(all_text),
        "odd_blocks": sum(1 for block in evaluation.ocr_blocks if len(block.text.strip()) <= 2),
        "block_count": len(evaluation.ocr_blocks),
    }


def evaluate_candidate_semantics(boxes: list[OCRRecord], text: str, entities: dict[str, Any], keywords: list[str]) -> float:
    score = 0.0
    medical_keywords = {"医保", "医保卡", "统筹", "报销", "住院报销", "门诊统筹", "保取", "保提"}
    illegal_keywords = {"套现", "取现", "提现", "回收", "高价回收"}
    keyword_set = set(keywords)
    score += min(3.0, len(keyword_set & medical_keywords)) * 18.0
    score += min(3.0, len(keyword_set & illegal_keywords)) * 16.0
    if entities["phones"]:
        score += 16.0
    if entities["wechat_ids"]:
        score += 14.0
    if entities["qqs"]:
        score += 10.0
    if (keyword_set & medical_keywords) and (entities["phones"] or entities["wechat_ids"] or entities["qqs"]):
        score += 18.0
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    score += min(12.0, len(text_lines) * 2.5)
    score += min(8.0, len(text.replace("\n", "").strip()) / 10.0)
    if "fallback.phone_only" in entities["hit_rules"] and not keyword_set:
        score -= 20.0
    score -= len(entities["noise_hits"]) * 16.0
    score -= entities["odd_blocks"] * 3.0
    if re.search(r"[?？]{2,}", text):
        score -= 8.0
    return round(max(0.0, min(100.0, score)), 4)


def _avg_confidence(records: list[OCRRecord]) -> float:
    return round(sum(item.confidence for item in records) / len(records), 4) if records else 0.0


def _evaluate_retry_candidate(
    image: np.ndarray,
    round1_boxes: list[OCRRecord],
    focus_boxes: list[OCRRecord],
    candidate: OCRCandidate,
    request_id: str,
) -> RetryCandidateResult:
    merged = merge_round1_and_focus(round1_boxes, focus_boxes)
    evaluation = evaluate_blocks(blocks=_records_to_blocks(merged), image=image, request_id=request_id)
    text = "\n".join(item.text for item in merged)
    entities = _semantic_entities(evaluation)
    semantic_score = evaluate_candidate_semantics(merged, text, entities, evaluation.hit_keywords)
    return RetryCandidateResult(
        candidate=candidate,
        merged_boxes=merged,
        evaluation=evaluation,
        semantic_score=semantic_score,
        avg_confidence=_avg_confidence(focus_boxes),
    )


def _select_best_retry_result(candidates: list[RetryCandidateResult]) -> RetryCandidateResult | None:
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.semantic_score, item.avg_confidence, len(item.merged_boxes)), reverse=True)
    return candidates[0]


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
    focus_selection = select_best_focus_region(image, round1_analysis.evaluation.ocr_blocks, round1_analysis) if retry_reason else None

    round1_semantic_score = evaluate_candidate_semantics(
        round1_boxes,
        "\n".join(item.text for item in round1_boxes),
        _semantic_entities(round1_analysis.evaluation),
        round1_analysis.evaluation.hit_keywords,
    )

    final_boxes = list(round1_boxes)
    focus_added_boxes = 0
    selected_variant = ""
    selected_semantic_score = 0.0
    selected_by_semantic = False
    variant_order: list[RetryVariant] = _variant_order(focus_selection, round1_analysis) if focus_selection is not None and retry_reason else []
    logger.info(
        "run_ocr request_id=%s retry_reason=%s round1_semantic_score=%.4f round1_low_semantic_confidence=%s focus_region=%s variant_order=%s",
        request_id,
        retry_reason or "",
        round1_semantic_score,
        round1_analysis.low_semantic_confidence,
        None if focus_selection is None else {
            "bbox": focus_selection.bbox,
            "score": focus_selection.score,
            "angle": round(focus_selection.angle, 4),
            "shape": focus_selection.shape,
        },
        variant_order,
    )
    if retry_reason and focus_selection is not None:
        retry_results: list[RetryCandidateResult] = []
        for variant in variant_order:
            candidate = build_focus_retry_candidate(image, focus_selection, variant)
            if candidate is None:
                logger.info("run_ocr request_id=%s variant=%s skipped reason=no_candidate", request_id, variant)
                continue
            focus_boxes = _predict_candidate(engine, candidate, settings.ocr.use_textline_orientation)
            retry_result = _evaluate_retry_candidate(image, round1_boxes, focus_boxes, candidate, request_id)
            logger.info(
                "run_ocr request_id=%s variant=%s focus_boxes=%s semantic_score=%.4f hit_keywords=%s hit_rules=%s",
                request_id,
                variant,
                len(focus_boxes),
                retry_result.semantic_score,
                retry_result.evaluation.hit_keywords,
                retry_result.evaluation.hit_rules,
            )
            retry_results.append(retry_result)

        best_retry = _select_best_retry_result(retry_results)
        if best_retry is not None and (
            best_retry.semantic_score > round1_semantic_score
            or best_retry.evaluation.hit_keywords != round1_analysis.evaluation.hit_keywords
        ):
            final_boxes = best_retry.merged_boxes
            focus_added_boxes = max(0, len(final_boxes) - len(round1_boxes))
            selected_variant = str(best_retry.candidate.variant)
            selected_semantic_score = best_retry.semantic_score
            selected_by_semantic = True
            logger.info(
                "run_ocr request_id=%s selected_by_semantic_score=%s selected_variant=%s selected_semantic_score=%.4f selected_hit_keywords=%s",
                request_id,
                True,
                selected_variant,
                selected_semantic_score,
                best_retry.evaluation.hit_keywords,
            )
        else:
            logger.info(
                "run_ocr request_id=%s selected_by_semantic_score=%s reason=no_better_variant",
                request_id,
                False,
            )
    else:
        logger.info("run_ocr request_id=%s selected_by_semantic_score=%s reason=no_focus_retry", request_id, False)

    final_boxes.sort(key=lambda item: (min(point[1] for point in item.points), min(point[0] for point in item.points)))
    focus_region = (
        FocusRegion(
            x1=focus_selection.bbox[0],
            y1=focus_selection.bbox[1],
            x2=focus_selection.bbox[2],
            y2=focus_selection.bbox[3],
            score=focus_selection.score,
        )
        if focus_selection is not None
        else None
    )
    return OCRRunResult(
        records=final_boxes,
        avg_confidence=_avg_confidence(final_boxes),
        ocr_text="\n".join(item.text for item in final_boxes),
        round1_triggered_focus_retry=bool(retry_reason and focus_selection is not None),
        focus_retry_reason=retry_reason,
        focus_region=focus_region,
        focus_retry_added_boxes=focus_added_boxes,
        focus_retry_variant=selected_variant,
        focus_retry_semantic_score=selected_semantic_score,
        round1_low_semantic_confidence=round1_analysis.low_semantic_confidence,
        selected_by_semantic_score=selected_by_semantic,
        focus_region_angle=round(focus_selection.angle, 4) if focus_selection is not None else 0.0,
        focus_region_shape=focus_selection.shape if focus_selection is not None else "",
    )
