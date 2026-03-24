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
from medical_ad_ocr_tools.services.rule_service import estimate_block_angle, evaluate_blocks, get_rule_config

RetryVariant = Literal["normal", "mirrored", "rotate_90", "rotate_180", "rotate_270", "deskew", "perspective"]
logger = logging.getLogger(__name__)


@dataclass
class OCRCandidate:
    name: str
    variant: RetryVariant | str
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
    source: str = "text_cluster"
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
    oblique_small_text_indices: list[int]


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
        detection_model_dir = Path(settings.ocr.text_detection_model_dir)
        if detection_model_dir.exists():
            kwargs["text_detection_model_dir"] = str(detection_model_dir)
        else:
            logger.warning(
                "OCR text detection model dir missing, falling back to model name: %s",
                detection_model_dir,
            )
    if settings.ocr.text_recognition_model_dir:
        recognition_model_dir = Path(settings.ocr.text_recognition_model_dir)
        if recognition_model_dir.exists():
            kwargs["text_recognition_model_dir"] = str(recognition_model_dir)
        else:
            logger.warning(
                "OCR text recognition model dir missing, falling back to model name: %s",
                recognition_model_dir,
            )
    if settings.ocr.textline_orientation_model_dir:
        textline_orientation_model_dir = Path(settings.ocr.textline_orientation_model_dir)
        if textline_orientation_model_dir.exists():
            kwargs["textline_orientation_model_dir"] = str(textline_orientation_model_dir)
        else:
            logger.warning(
                "OCR textline orientation model dir missing, skipping local model dir: %s",
                textline_orientation_model_dir,
            )
    logger.info(
        "Initializing PaddleOCR with detection_model_dir=%s recognition_model_dir=%s textline_orientation_model_dir=%s",
        kwargs.get("text_detection_model_dir"),
        kwargs.get("text_recognition_model_dir"),
        kwargs.get("textline_orientation_model_dir"),
    )
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


def _make_named_candidate(crop: np.ndarray, name: str, offset_x: int, offset_y: int) -> OCRCandidate:
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


def _build_large_image_retry_candidates(image: np.ndarray) -> list[OCRCandidate]:
    height, width = image.shape[:2]
    if max(height, width) < 1800:
        return []
    tile_w = min(width, max(int(width * 0.62), width // 2))
    tile_h = min(height, max(int(height * 0.62), height // 2))
    if tile_w >= width or tile_h >= height:
        return []
    x_starts = sorted({0, max(0, width - tile_w)})
    y_starts = sorted({0, max(0, height - tile_h)})
    candidates: list[OCRCandidate] = []
    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            crop = image[y : y + tile_h, x : x + tile_w]
            candidates.append(_make_named_candidate(crop, f"tile_{row}_{col}", x, y))
    if height > width * 1.05:
        bottom_crop_y = max(0, height - int(height * 0.72))
        bottom_crop = image[bottom_crop_y:, :]
        candidates.append(_make_named_candidate(bottom_crop, "bottom_crop", 0, bottom_crop_y))
    return candidates


def _build_low_signal_scan_candidates(image: np.ndarray) -> list[OCRCandidate]:
    height, width = image.shape[:2]
    crop_w = max(int(width * 0.68), min(width, 420))
    crop_h = max(int(height * 0.68), min(height, 420))
    crop_w = min(width, crop_w)
    crop_h = min(height, crop_h)
    x_starts = sorted({0, max(0, (width - crop_w) // 2), max(0, width - crop_w)})
    y_starts = sorted({0, max(0, (height - crop_h) // 2), max(0, height - crop_h)})
    candidates: list[OCRCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            bbox = (x, y, x + crop_w, y + crop_h)
            if bbox in seen:
                continue
            seen.add(bbox)
            crop = image[y : y + crop_h, x : x + crop_w]
            candidates.append(_make_named_candidate(crop, f"scan_{row}_{col}", x, y))
    if height > width:
        lower_y = max(0, height - int(height * 0.62))
        lower_crop = image[lower_y:, :]
        candidates.append(_make_named_candidate(lower_crop, "scan_lower", 0, lower_y))
    return candidates[:7]


def _detect_qr_regions(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = cv2.QRCodeDetector()
    regions: list[tuple[int, int, int, int]] = []
    try:
        ok, points = detector.detectMulti(image)
    except cv2.error:
        ok, points = False, None
    if not ok or points is None:
        return regions
    for quad in points:
        pts = np.array(quad, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        regions.append((x, y, x + w, y + h))
    return regions


def _build_qr_probe_candidates(image: np.ndarray) -> list[OCRCandidate]:
    candidates: list[OCRCandidate] = []
    for index, (x1, y1, x2, y2) in enumerate(_detect_qr_regions(image)[:3]):
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        expanded = _clip_bbox(
            (
                x1 - max(50, int(width * 1.4)),
                y1 - max(60, int(height * 1.2)),
                x2 + max(50, int(width * 1.4)),
                y2 + max(90, int(height * 1.8)),
            ),
            image.shape,
        )
        crop = image[expanded[1] : expanded[3], expanded[0] : expanded[2]]
        if crop.size == 0:
            continue
        candidates.append(_make_named_candidate(crop, f"qr_probe_{index}", expanded[0], expanded[1]))
    return candidates


def _build_full_rotation_probe_candidates(image: np.ndarray) -> list[OCRCandidate]:
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    rotated_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return [
        OCRCandidate(
            name="full_rotate_90",
            variant="rotate_90",
            image=rotated_90,
            crop_width=image.shape[1],
            crop_height=image.shape[0],
            scale=1.0,
            rotation_quadrants=1,
        ),
        OCRCandidate(
            name="full_rotate_180",
            variant="rotate_180",
            image=rotated_180,
            crop_width=image.shape[1],
            crop_height=image.shape[0],
            scale=1.0,
            rotation_quadrants=2,
        ),
        OCRCandidate(
            name="full_rotate_270",
            variant="rotate_270",
            image=rotated_270,
            crop_width=image.shape[1],
            crop_height=image.shape[0],
            scale=1.0,
            rotation_quadrants=3,
        ),
    ]


def _build_card_probe_candidates(cards: list[CardCandidate]) -> list[OCRCandidate]:
    settings = get_settings()
    candidates: list[OCRCandidate] = []
    for index, card in enumerate(cards[:3]):
        processed = _enhance_crop(card.crop)
        if settings.ocr.focus_retry_scale != 1.0:
            processed = cv2.resize(processed, None, fx=settings.ocr.focus_retry_scale, fy=settings.ocr.focus_retry_scale, interpolation=cv2.INTER_CUBIC)
        candidates.append(
            OCRCandidate(
                name=f"pre_card_{index}",
                variant=f"pre_card_{index}",
                image=processed,
                crop_width=card.crop.shape[1],
                crop_height=card.crop.shape[0],
                scale=settings.ocr.focus_retry_scale,
                inverse_perspective_matrix=card.inverse_perspective_matrix,
            )
        )
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


def _has_meaningful_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", text))


def _is_punctuation_noise(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned)


def _should_retry_large_image_detection(records: list[OCRRecord], image: np.ndarray) -> bool:
    if max(image.shape[:2]) < 1800:
        return False
    if not records:
        return True
    meaningful = [record for record in records if _has_meaningful_text(record.text)]
    punctuation_only = [record for record in records if _is_punctuation_noise(record.text)]
    avg_conf = _avg_confidence(records)
    if meaningful:
        return False
    if len(records) <= 8 and len(punctuation_only) >= max(3, len(records) - 1):
        return True
    return avg_conf < 0.45 and len(punctuation_only) >= max(1, len(records) // 2)


def _has_phone_like_digits(records: list[OCRRecord]) -> bool:
    return any(re.search(r"\d{7,}", record.text) for record in records)


def _has_config_risk_keywords(records: list[OCRRecord]) -> bool:
    config = get_rule_config()
    text = "\n".join(record.text for record in records)
    return any(keyword in text for keyword in (config.medical_keywords + config.illegal_keywords))


def _has_noise_keywords(records: list[OCRRecord]) -> bool:
    text = "\n".join(record.text for record in records)
    return bool(_noise_hits_for_text(text))


def _should_run_low_signal_scan(records: list[OCRRecord], image: np.ndarray) -> bool:
    if not records:
        return True
    meaningful_count = sum(1 for record in records if _has_meaningful_text(record.text))
    meaningful_len = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", record.text)) for record in records)
    if meaningful_len <= 12 and meaningful_count <= 6:
        return True
    if meaningful_len <= 18 and not _has_phone_like_digits(records) and max(image.shape[:2]) >= 800:
        return True
    return False


def _should_run_rotation_probe(records: list[OCRRecord], image: np.ndarray) -> bool:
    if not records:
        return True
    if _has_config_risk_keywords(records):
        return False
    meaningful_count = sum(1 for record in records if _has_meaningful_text(record.text))
    meaningful_len = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", record.text)) for record in records)
    if meaningful_len <= 24 and meaningful_count <= 10:
        return True
    if _has_phone_like_digits(records) and _has_noise_keywords(records):
        return True
    return max(image.shape[:2]) >= 800 and meaningful_count <= 12


def _meaningful_text_length(blocks: list[OCRBlock]) -> int:
    return sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", block.text)) for block in blocks)


def _low_text_recall(analysis: Round1Analysis) -> bool:
    return _meaningful_text_length(analysis.evaluation.ocr_blocks) <= 8


def _is_oblique_small_text_block(block: OCRBlock, median_center_y: float) -> bool:
    text = re.sub(r"\s+", "", block.text)
    if not text or len(text) > 4:
        return False
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return False
    angle = abs(estimate_block_angle(block))
    if angle < 20.0:
        return False
    x1, y1, x2, y2 = _bbox(block.points)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    has_digit_hint = bool(re.search(r"\d", text))
    lower_than_main_text = (y1 + y2) / 2 >= median_center_y + max(40.0, height * 0.6)
    elongated = max(width, height) / max(1.0, min(width, height)) >= 1.15
    return lower_than_main_text and (has_digit_hint or elongated)


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
    centers_y = [(_bbox(block.points)[1] + _bbox(block.points)[3]) / 2 for block in evaluation.ocr_blocks]
    median_center_y = float(np.median(centers_y)) if centers_y else 0.0
    oblique_small_text_indices = [
        index
        for index, block in enumerate(evaluation.ocr_blocks)
        if _is_oblique_small_text_block(block, median_center_y)
    ]
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
        oblique_small_text_indices=oblique_small_text_indices,
    )


def should_run_focus_retry(round1_analysis: Round1Analysis) -> str:
    settings = get_settings()
    if not settings.ocr.focus_retry_enabled:
        return ""
    evaluation = round1_analysis.evaluation
    has_any_risk = round1_analysis.high_risk_keyword_hits >= 1
    has_oblique_small_text = bool(round1_analysis.oblique_small_text_indices)
    low_text_recall = _low_text_recall(round1_analysis)
    if evaluation.risk_score >= settings.ocr.focus_retry_mid_risk_max and round1_analysis.has_complete_contact and round1_analysis.high_risk_keyword_hits >= 2:
        return ""
    if not has_any_risk and not round1_analysis.has_complete_contact and not round1_analysis.has_card_candidate and not has_oblique_small_text:
        return ""
    if round1_analysis.has_card_candidate and not has_any_risk and not round1_analysis.has_complete_contact and low_text_recall:
        return "card_candidate_low_text_recall"
    if round1_analysis.high_risk_keyword_hits >= settings.ocr.focus_retry_min_keyword_hits and not round1_analysis.has_complete_contact:
        return "high_risk_keywords_missing_contact"
    if has_oblique_small_text and not has_any_risk and not round1_analysis.has_complete_contact:
        return "oblique_small_text"
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


def _expanded_bbox_for_block(block: OCRBlock, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = _bbox(block.points)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    padding_x = max(24, int(width * 1.0))
    padding_y = max(28, int(height * 1.0))
    return _clip_bbox((x1 - padding_x, y1 - padding_y, x2 + padding_x, y2 + padding_y), image_shape)


def _expanded_bbox_for_card_candidate(
    card: CardCandidate,
    image_shape: tuple[int, int, int],
    *,
    aggressive: bool = False,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = card.bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if aggressive:
        padding_left = max(80, int(width * 0.85))
        padding_right = max(60, int(width * 0.35))
        padding_top = max(70, int(height * 0.40))
        padding_bottom = max(120, int(height * 0.95))
    else:
        padding_left = max(32, int(width * 0.18))
        padding_right = max(24, int(width * 0.18))
        padding_top = max(28, int(height * 0.14))
        padding_bottom = max(36, int(height * 0.18))
    return _clip_bbox((x1 - padding_left, y1 - padding_top, x2 + padding_right, y2 + padding_bottom), image_shape)


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
    oblique_small_text_hits = sum(1 for index in block_indices if index in (analysis.oblique_small_text_indices if analysis is not None else []))
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
    if analysis is not None and source == "card_candidate" and preset_card is not None and _low_text_recall(analysis):
        card_area = max(1, (preset_card.bbox[2] - preset_card.bbox[0]) * (preset_card.bbox[3] - preset_card.bbox[1]))
        image_area = max(1, image.shape[0] * image.shape[1])
        card_area_ratio = card_area / image_area
        if card_area_ratio >= 0.015:
            score = max(score, 0.46)
        if shape in {"rectangle", "trapezoid"}:
            score += 0.08
        if abs(angle) >= 8.0:
            score += 0.06
    if analysis is not None and oblique_small_text_hits and source == "oblique_small_text":
        score += 0.32
    elif analysis is not None and oblique_small_text_hits and not keyword_hits and not contact_hits:
        score += min(0.18, oblique_small_text_hits * 0.12)
    score = round(max(0.0, min(1.0, score)), 4)
    return FocusRegionSelection(
        bbox=bbox,
        score=min(score, 1.0),
        block_indices=block_indices,
        angle=angle,
        shape=shape,
        source=source,
        card_candidate=overlap_card,
    )


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
        preset_bbox = _expanded_bbox_for_card_candidate(card, image.shape, aggressive=_low_text_recall(analysis))
        nearby_indices = _nearby_block_indices(boxes, preset_bbox)
        candidate = _region_score(
            image,
            boxes,
            nearby_indices,
            analysis.card_candidates,
            analysis=analysis,
            source="card_candidate",
            preset_bbox=preset_bbox,
            preset_card=card,
        )
        if candidate is not None:
            candidates.append(candidate)

    for block_index in analysis.oblique_small_text_indices:
        expanded_bbox = _expanded_bbox_for_block(boxes[block_index], image.shape)
        nearby_indices = _nearby_block_indices(boxes, expanded_bbox)
        if block_index not in nearby_indices:
            nearby_indices.append(block_index)
        candidate = _region_score(
            image,
            boxes,
            sorted(set(nearby_indices)),
            analysis.card_candidates,
            analysis=analysis,
            source="oblique_small_text",
            preset_bbox=expanded_bbox,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    min_region_score = settings.ocr.focus_retry_min_region_score
    if analysis.has_card_candidate and _low_text_recall(analysis):
        min_region_score = min(min_region_score, 0.32)
    if analysis.oblique_small_text_indices and analysis.high_risk_keyword_hits == 0 and not analysis.has_complete_contact:
        min_region_score = min(min_region_score, 0.4)
    if best.score < min_region_score:
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
    if selection.source == "card_candidate" and selection.card_candidate is not None:
        bbox = _expanded_bbox_for_card_candidate(selection.card_candidate, image.shape, aggressive=True)
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
    oblique_scene = bool(analysis.oblique_small_text_indices) and analysis.high_risk_keyword_hits == 0 and not analysis.has_complete_contact
    card_low_text_scene = selection.source == "card_candidate" and selection.card_candidate is not None and _low_text_recall(analysis)
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
    if card_low_text_scene:
        variants.extend(["rotate_180", "rotate_90", "rotate_270"])
    if oblique_scene:
        variants.extend(["deskew", "rotate_90", "rotate_270"])
    if looks_vertical_card:
        variants.extend(["rotate_90", "rotate_270"])
    if trapezoid and settings.ocr.focus_retry_enable_perspective:
        variants.append("perspective")
    elif angled:
        variants.append("deskew")
    if card_low_text_scene:
        if "deskew" not in variants:
            variants.append("deskew")
        if settings.ocr.focus_retry_enable_perspective and "perspective" not in variants:
            variants.append("perspective")
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

    if _should_run_low_signal_scan(round1_boxes, image):
        scan_candidates = _build_low_signal_scan_candidates(image)
        logger.info(
            "run_ocr request_id=%s low_signal_scan=true initial_boxes=%s scan_candidates=%s",
            request_id,
            len(round1_boxes),
            [candidate.name for candidate in scan_candidates],
        )
        for candidate in scan_candidates:
            round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
        round1_boxes = _deduplicate(round1_boxes)
    else:
        logger.info(
            "run_ocr request_id=%s low_signal_scan=false initial_boxes=%s",
            request_id,
            len(round1_boxes),
        )
    if _should_run_rotation_probe(round1_boxes, image):
        rotation_probe_candidates = _build_full_rotation_probe_candidates(image)
        logger.info(
            "run_ocr request_id=%s rotation_probe=true candidates=%s",
            request_id,
            [candidate.name for candidate in rotation_probe_candidates],
        )
        for candidate in rotation_probe_candidates:
            round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
        round1_boxes = _deduplicate(round1_boxes)
    else:
        logger.info("run_ocr request_id=%s rotation_probe=false", request_id)

    if _should_retry_large_image_detection(round1_boxes, image):
        fallback_candidates = _build_large_image_retry_candidates(image)
        logger.info(
            "run_ocr request_id=%s large_image_detection_retry=true initial_boxes=%s fallback_candidates=%s",
            request_id,
            len(round1_boxes),
            [candidate.name for candidate in fallback_candidates],
        )
        for candidate in fallback_candidates:
            round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
        round1_boxes = _deduplicate(round1_boxes)
    else:
        logger.info(
            "run_ocr request_id=%s large_image_detection_retry=false initial_boxes=%s",
            request_id,
            len(round1_boxes),
        )
    if _should_run_low_signal_scan(round1_boxes, image):
        qr_probe_candidates = _build_qr_probe_candidates(image)
        if qr_probe_candidates:
            logger.info(
                "run_ocr request_id=%s qr_probe=true qr_candidates=%s",
                request_id,
                [candidate.name for candidate in qr_probe_candidates],
            )
            for candidate in qr_probe_candidates:
                round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
            round1_boxes = _deduplicate(round1_boxes)
        else:
            logger.info("run_ocr request_id=%s qr_probe=false reason=no_qr_detected", request_id)
    card_probe_candidates = detect_card_candidates(image)
    if card_probe_candidates and not any(_has_meaningful_text(record.text) for record in round1_boxes):
        probe_candidates = _build_card_probe_candidates(card_probe_candidates)
        logger.info(
            "run_ocr request_id=%s pre_card_probe=true card_candidates=%s probe_candidates=%s",
            request_id,
            len(card_probe_candidates),
            [candidate.name for candidate in probe_candidates],
        )
        for candidate in probe_candidates:
            round1_boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))
        round1_boxes = _deduplicate(round1_boxes)
    else:
        logger.info(
            "run_ocr request_id=%s pre_card_probe=false card_candidates=%s meaningful_text=%s",
            request_id,
            len(card_probe_candidates),
            any(_has_meaningful_text(record.text) for record in round1_boxes),
        )

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
