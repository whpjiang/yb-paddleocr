from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from paddleocr import PaddleOCR

from medical_ad_ocr_tools.core.models import OCRRecord
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.card_detector import CardCandidate, detect_card_candidates


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
    inverse_perspective_matrix: list[list[float]] | None = None


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


def _enhance_image(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5.2, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(enhanced, -1, kernel)


def _make_candidate(
    crop: np.ndarray,
    name: str,
    offset_x: int,
    offset_y: int,
    *,
    scale: float = 1.0,
    enhance: bool = False,
    rotation: int = 0,
    inverse_perspective_matrix: list[list[float]] | None = None,
) -> OCRCandidate:
    processed = _enhance_image(crop) if enhance else crop.copy()
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
        inverse_perspective_matrix=inverse_perspective_matrix,
    )


def _build_initial_candidates(image: np.ndarray) -> list[OCRCandidate]:
    candidates = [_make_candidate(image, "full", 0, 0)]
    if image.shape[0] > image.shape[1] * 1.2:
        top_crop = image[: int(image.shape[0] * 0.82), :]
        candidates.append(_make_candidate(top_crop, "top_crop", 0, 0))
    return candidates


def _clip_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1 = min(max(0, bbox[0]), width - 1)
    y1 = min(max(0, bbox[1]), height - 1)
    x2 = min(max(x1 + 1, bbox[2]), width)
    y2 = min(max(y1 + 1, bbox[3]), height)
    return x1, y1, x2, y2


def _map_points(candidate: OCRCandidate, points: list[list[float]]) -> list[list[int]]:
    transformed_points: list[list[float]] = []
    for point in points:
        scaled_x = point[0] / candidate.scale
        scaled_y = point[1] / candidate.scale
        if candidate.rotation == 90:
            crop_x, crop_y = scaled_y, candidate.crop_height - scaled_x
        elif candidate.rotation == -90:
            crop_x, crop_y = candidate.crop_width - scaled_y, scaled_x
        else:
            crop_x, crop_y = scaled_x, scaled_y
        transformed_points.append([crop_x, crop_y])

    if candidate.inverse_perspective_matrix is not None:
        matrix = np.array(candidate.inverse_perspective_matrix, dtype=np.float32)
        src = np.array(transformed_points, dtype=np.float32).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(src, matrix).reshape(-1, 2)
        return [[int(round(x)), int(round(y))] for x, y in dst]

    return [[int(round(x + candidate.offset_x)), int(round(y + candidate.offset_y))] for x, y in transformed_points]


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


def _normalize_text(text: str) -> str:
    return "".join(char for char in text.lower() if char.isalnum())


def _deduplicate(records: list[OCRRecord]) -> list[OCRRecord]:
    deduped: list[OCRRecord] = []
    for record in sorted(records, key=lambda item: item.confidence, reverse=True):
        current_bbox = _bbox(record.points)
        current_text = _normalize_text(record.text)
        duplicate = False
        for kept in deduped:
            kept_bbox = _bbox(kept.points)
            kept_text = _normalize_text(kept.text)
            center_close = abs((current_bbox[0] + current_bbox[2]) - (kept_bbox[0] + kept_bbox[2])) <= 24 and abs(
                (current_bbox[1] + current_bbox[3]) - (kept_bbox[1] + kept_bbox[3])
            ) <= 24
            if current_text and kept_text and current_text == kept_text and (_iou(current_bbox, kept_bbox) >= 0.25 or center_close):
                duplicate = True
                break
            if _similar_text(current_text, kept_text) and (_iou(current_bbox, kept_bbox) >= 0.5 or center_close):
                duplicate = True
                break
        if not duplicate:
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
            if not text:
                continue
            records.append(
                OCRRecord(
                    points=_map_points(candidate, points),
                    text=text,
                    confidence=confidence,
                )
            )
    return records


def _build_focus_regions(
    image: np.ndarray,
    records: list[OCRRecord],
    card_candidates: list[CardCandidate],
) -> list[tuple[str, tuple[int, int, int, int], list[list[float]] | None]]:
    regions: list[tuple[str, tuple[int, int, int, int], list[list[float]] | None]] = []
    text_boxes = [_bbox(record.points) for record in records]
    used: set[int] = set()

    for index, box in enumerate(text_boxes):
        if index in used:
            continue
        selected = {index}
        x1, y1, x2, y2 = box
        box_w = max(20, x2 - x1)
        box_h = max(20, y2 - y1)
        limit_x = max(80, int(box_w * 2.4))
        limit_y = max(120, int(box_h * 5.0))
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        for other_index, other in enumerate(text_boxes):
            if other_index == index:
                continue
            ox1, oy1, ox2, oy2 = other
            other_center_x = (ox1 + ox2) / 2
            other_center_y = (oy1 + oy2) / 2
            if abs(other_center_x - center_x) <= limit_x and abs(other_center_y - center_y) <= limit_y:
                selected.add(other_index)

        merged = (
            min(text_boxes[item][0] for item in selected),
            min(text_boxes[item][1] for item in selected),
            max(text_boxes[item][2] for item in selected),
            max(text_boxes[item][3] for item in selected),
        )
        padding_x = max(20, int((merged[2] - merged[0]) * 0.22))
        padding_y = max(24, int((merged[3] - merged[1]) * 0.28))
        clipped = _clip_bbox((merged[0] - padding_x, merged[1] - padding_y, merged[2] + padding_x, merged[3] + padding_y), image.shape)
        duplicate = False
        for _, existing, _ in regions:
            if _iou(existing, clipped) >= 0.45:
                duplicate = True
                break
        if not duplicate and (clipped[2] - clipped[0]) >= 48 and (clipped[3] - clipped[1]) >= 48:
            regions.append((f"text_{index}", clipped, None))
            used.update(selected)

    for index, card in enumerate(card_candidates):
        clipped = _clip_bbox(card.bbox, image.shape)
        duplicate = False
        for _, existing, _ in regions:
            if _iou(existing, clipped) >= 0.5:
                duplicate = True
                break
        if not duplicate:
            regions.append((f"card_{index}", clipped, card.inverse_perspective_matrix))

    regions.sort(key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]), reverse=True)
    return regions[:4]


def _build_targeted_candidates(
    image: np.ndarray,
    records: list[OCRRecord],
    card_candidates: list[CardCandidate],
) -> list[OCRCandidate]:
    candidates: list[OCRCandidate] = []

    for name, bbox, _ in _build_focus_regions(image, records, card_candidates):
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        candidates.append(_make_candidate(crop, f"{name}_enhanced", x1, y1, scale=2.6, enhance=True))
        if name.startswith("text_") and (y2 - y1) > (x2 - x1) * 1.25:
            candidates.append(_make_candidate(crop, f"{name}_rot", x1, y1, scale=3.0, enhance=True, rotation=90))

    for index, card in enumerate(card_candidates[:2]):
        candidates.append(
            _make_candidate(
                card.crop,
                f"card_warp_enhanced_{index}",
                0,
                0,
                scale=3.0,
                enhance=True,
                inverse_perspective_matrix=card.inverse_perspective_matrix,
            )
        )
        if card.crop.shape[0] > card.crop.shape[1] * 1.1:
            candidates.append(
                _make_candidate(
                    card.crop,
                    f"card_warp_rot_{index}",
                    0,
                    0,
                    scale=3.2,
                    enhance=True,
                    rotation=90,
                    inverse_perspective_matrix=card.inverse_perspective_matrix,
                )
            )
    return candidates


def run_ocr(image_path: str | Path) -> tuple[list[OCRRecord], float, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    settings = get_settings()
    engine = get_ocr_engine()
    card_candidates = detect_card_candidates(image)

    boxes: list[OCRRecord] = []
    for candidate in _build_initial_candidates(image):
        boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))

    first_pass_boxes = _deduplicate(boxes)
    for candidate in _build_targeted_candidates(image, first_pass_boxes, card_candidates):
        boxes.extend(_predict_candidate(engine, candidate, settings.ocr.use_textline_orientation))

    boxes = _deduplicate(boxes)
    boxes.sort(key=lambda item: (min(point[1] for point in item.points), min(point[0] for point in item.points)))
    avg_confidence = round(sum(item.confidence for item in boxes) / len(boxes), 4) if boxes else 0.0
    return boxes, avg_confidence, "\n".join(item.text for item in boxes)
