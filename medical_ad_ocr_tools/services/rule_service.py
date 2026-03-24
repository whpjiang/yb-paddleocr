from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml

from medical_ad_ocr_tools.core.models import AdRegion, OCRBlock, OCRBlockInput, OCRRecord, RuleConfig, RuleEvaluationResponse, RuleEvaluateRequest
from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.card_detector import CardCandidate, detect_card_candidates

logger = logging.getLogger(__name__)


@dataclass
class BoxEvidence:
    index: int
    block: OCRBlock
    medical_hits: list[str]
    illegal_hits: list[str]
    contact_hits: list[str]
    phones: list[str]
    wechat_ids: list[str]
    qqs: list[str]
    noise_hits: list[str]


def estimate_block_angle(block: OCRBlock) -> float:
    points = np.array(block.points, dtype=np.float32)
    if len(points) < 4:
        return 0.0
    rect = cv2.minAreaRect(points)
    (width, height) = rect[1]
    angle = float(rect[2])
    if width < height:
        angle -= 90.0
    while angle <= -45.0:
        angle += 90.0
    while angle > 45.0:
        angle -= 90.0
    return angle


def extract_patch(image: np.ndarray | None, bbox: tuple[int, int, int, int], padding: int = 6) -> np.ndarray | None:
    if image is None:
        return None
    height, width = image.shape[:2]
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(width, bbox[2] + padding)
    y2 = min(height, bbox[3] + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    return patch if patch.size else None


def estimate_patch_signature(patch: np.ndarray | None) -> dict[str, np.ndarray] | None:
    if patch is None or patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return {
        "hsv_mean": hsv.reshape(-1, 3).mean(axis=0),
        "hsv_std": hsv.reshape(-1, 3).std(axis=0),
        "lab_mean": lab.reshape(-1, 3).mean(axis=0),
        "lab_std": lab.reshape(-1, 3).std(axis=0),
    }


def patch_similarity(sig1: dict[str, np.ndarray] | None, sig2: dict[str, np.ndarray] | None) -> float:
    if sig1 is None or sig2 is None:
        return 0.5
    mean_dist = float(np.linalg.norm(sig1["lab_mean"] - sig2["lab_mean"]) / 100.0)
    std_dist = float(np.linalg.norm(sig1["lab_std"] - sig2["lab_std"]) / 60.0)
    score = 1.0 - min(1.0, mean_dist * 0.75 + std_dist * 0.25)
    return round(max(0.0, score), 4)


def _item_center(item: BoxEvidence) -> tuple[float, float]:
    x1, y1, x2, y2 = _box_bounds(item.block.points)
    return (x1 + x2) / 2, (y1 + y2) / 2


def _center_distance(a: BoxEvidence, b: BoxEvidence) -> float:
    ax, ay = _item_center(a)
    bx, by = _item_center(b)
    return float(np.hypot(ax - bx, ay - by))


def _is_noise_only_item(item: BoxEvidence) -> bool:
    return bool(item.noise_hits and not (item.medical_hits or item.illegal_hits))


def _contact_like_digits(item: BoxEvidence) -> bool:
    text = item.block.text
    return bool(re.search(r"\d{6,}", text)) or bool(item.contact_hits)


def collect_core_items(items: list[BoxEvidence]) -> list[BoxEvidence]:
    cores = [item for item in items if item.medical_hits or item.illegal_hits]
    if not cores:
        return []
    selected: dict[int, BoxEvidence] = {item.index: item for item in cores}
    for item in items:
        if item.index in selected or _is_noise_only_item(item) or item.phones or item.wechat_ids or item.qqs:
            continue
        if any(_center_distance(item, core) <= 90 for core in cores):
            selected[item.index] = item
    return list(selected.values())


def collect_contact_items(items: list[BoxEvidence]) -> list[BoxEvidence]:
    return [item for item in items if item.phones or item.wechat_ids or item.qqs or _contact_like_digits(item)]


def contact_alignment_score(contact_block: OCRBlock, core_blocks: list[OCRBlock]) -> tuple[float, float]:
    if not core_blocks:
        return 0.2, 90.0
    contact_angle = estimate_block_angle(contact_block)
    core_angles = [estimate_block_angle(block) for block in core_blocks]
    core_angle = float(np.median(core_angles))
    angle_diff = abs(contact_angle - core_angle)
    angle_diff = min(angle_diff, abs(angle_diff - 90.0))
    if angle_diff <= 12:
        return 1.0, angle_diff
    if angle_diff <= 25:
        return 0.65, angle_diff
    return 0.2, angle_diff


def contact_background_similarity(contact_item: BoxEvidence, core_items: list[BoxEvidence], image: np.ndarray | None) -> float:
    if image is None or not core_items:
        return 0.5
    contact_sig = estimate_patch_signature(extract_patch(image, _box_bounds(contact_item.block.points)))
    sims = [
        patch_similarity(contact_sig, estimate_patch_signature(extract_patch(image, _box_bounds(core.block.points))))
        for core in core_items
    ]
    return round(sum(sims) / max(1, len(sims)), 4)


def score_contact_attachment(
    contact_item: BoxEvidence,
    core_items: list[BoxEvidence],
    image: np.ndarray | None,
    candidate_items: list[BoxEvidence],
) -> tuple[float, dict[str, float], str]:
    if not core_items:
        return 0.0, {"angle_diff": 90.0, "patch_similarity": 0.0}, "no_core_items"

    core_blocks = [item.block for item in core_items]
    align_score, angle_diff = contact_alignment_score(contact_item.block, core_blocks)
    patch_score = contact_background_similarity(contact_item, core_items, image)

    region_bbox = (
        min(_box_bounds(item.block.points)[0] for item in core_items),
        min(_box_bounds(item.block.points)[1] for item in core_items),
        max(_box_bounds(item.block.points)[2] for item in core_items),
        max(_box_bounds(item.block.points)[3] for item in core_items),
    )
    diag = max(1.0, float(np.hypot(region_bbox[2] - region_bbox[0], region_bbox[3] - region_bbox[1])))
    min_dist = min(_center_distance(contact_item, core) for core in core_items)
    distance_score = max(0.0, 1.0 - min(1.0, min_dist / diag))

    same_cluster_score = 1.0 if any(_center_distance(contact_item, core) <= 80 for core in core_items) else 0.35
    cx, cy = _item_center(contact_item)
    core_centers = [_item_center(core) for core in core_items]
    core_cx = sum(x for x, _ in core_centers) / len(core_centers)
    core_cy = sum(y for _, y in core_centers) / len(core_centers)
    layout_score = 0.8 if (cy >= core_cy - 20 and abs(cx - core_cx) <= max(60.0, diag * 0.4)) else 0.4

    nearby_noise = any(
        other.noise_hits and _center_distance(contact_item, other) <= 85
        for other in candidate_items
        if other.index != contact_item.index
    ) or bool(contact_item.noise_hits)
    noise_penalty = 0.35 if nearby_noise else 0.0

    score = align_score * 0.24 + patch_score * 0.24 + distance_score * 0.20 + same_cluster_score * 0.16 + layout_score * 0.16 - noise_penalty
    score = round(max(0.0, min(1.0, score)), 4)

    reason = ""
    if nearby_noise and score < 0.6:
        reason = "noise_nearby"
    elif angle_diff > 25:
        reason = "angle_mismatch"
    elif patch_score < 0.45:
        reason = "background_mismatch"
    elif score < 0.55:
        reason = "low_attachment_score"
    return score, {"angle_diff": round(angle_diff, 4), "patch_similarity": patch_score}, reason


def filter_attached_contacts(
    items: list[BoxEvidence],
    all_items: list[BoxEvidence],
    image: np.ndarray | None,
    request_id: str,
    group_index: int,
) -> tuple[list[str], list[str], list[str], list[int]]:
    core_items = collect_core_items(items)
    contact_items = collect_contact_items(items)
    logger.info("evaluate_blocks request_id=%s group=%s core_items=%s contact_items=%s", request_id, group_index, [item.index for item in core_items], [item.index for item in contact_items])
    if not core_items or not contact_items:
        return [], [], [], []

    best_phone: tuple[float, BoxEvidence] | None = None
    best_wechat: tuple[float, BoxEvidence] | None = None
    best_qq: tuple[float, BoxEvidence] | None = None
    attached_indices: list[int] = []

    for contact_item in contact_items:
        score, debug, reject_reason = score_contact_attachment(contact_item, core_items, image, all_items)
        accepted = score >= 0.55
        logger.info(
            "evaluate_blocks request_id=%s group=%s contact_item=%s attachment_score=%.4f angle_diff=%.4f patch_similarity=%.4f accepted=%s reject_reason=%s",
            request_id,
            group_index,
            contact_item.index,
            score,
            debug["angle_diff"],
            debug["patch_similarity"],
            accepted,
            reject_reason or "",
        )
        if not accepted:
            continue
        attached_indices.append(contact_item.index)
        if contact_item.phones and (best_phone is None or score > best_phone[0]):
            best_phone = (score, contact_item)
        if contact_item.wechat_ids and (best_wechat is None or score > best_wechat[0]):
            best_wechat = (score, contact_item)
        if contact_item.qqs and (best_qq is None or score > best_qq[0]):
            best_qq = (score, contact_item)

    phones = best_phone[1].phones if best_phone is not None else []
    wechat_ids = best_wechat[1].wechat_ids if best_wechat is not None else []
    qqs = best_qq[1].qqs if best_qq is not None else []
    return phones, wechat_ids, qqs, sorted(set(attached_indices))


@lru_cache(maxsize=1)
def get_rule_config() -> RuleConfig:
    path = Path(get_settings().rules_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return RuleConfig.model_validate(data)


def _normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return digits if len(digits) == 11 and digits.startswith("1") else ""


def _contains_any(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def _match_aliases(text: str, alias_map: dict[str, list[str]]) -> list[str]:
    matched: list[str] = []
    for canonical, aliases in alias_map.items():
        if canonical in text or any(alias in text for alias in aliases):
            matched.append(canonical)
    return matched


def _box_bounds(points: list[list[int]]) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_contains(bbox: tuple[int, int, int, int], point_x: float, point_y: float) -> bool:
    return bbox[0] <= point_x <= bbox[2] and bbox[1] <= point_y <= bbox[3]


def _expand_bbox(bbox: tuple[int, int, int, int], padding_x: int, padding_y: int) -> tuple[int, int, int, int]:
    return bbox[0] - padding_x, bbox[1] - padding_y, bbox[2] + padding_x, bbox[3] + padding_y


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _risk_level(score: int, config: RuleConfig) -> str:
    if score >= config.high_risk_score:
        return "high"
    if score >= config.suspicious_score:
        return "medium"
    return "low"


def _is_suspicious_score(score: int, config: RuleConfig, medical_hits: list[str], illegal_hits: list[str]) -> bool:
    return score >= config.suspicious_score or bool(medical_hits and illegal_hits)


def _should_keep_region(score: int, medical_hits: list[str], illegal_hits: list[str], phones: list[str], wechat_ids: list[str], qqs: list[str]) -> bool:
    return bool(medical_hits or illegal_hits or phones or wechat_ids or qqs or score > 0)


def _build_evidence(blocks: list[OCRBlock]) -> list[BoxEvidence]:
    config = get_rule_config()
    phone_pattern = re.compile(config.phone_pattern)
    wechat_pattern = re.compile(config.wechat_pattern, re.IGNORECASE)
    qq_pattern = re.compile(config.qq_pattern, re.IGNORECASE)
    extra_noise_keywords = ["刻章", "办证", "贷款", "发票"]
    evidence_list: list[BoxEvidence] = []
    for index, block in enumerate(blocks):
        text = block.text
        evidence_list.append(
            BoxEvidence(
                index=index,
                block=block,
                medical_hits=sorted(set(_contains_any(text, config.medical_keywords) + _match_aliases(text, config.medical_aliases))),
                illegal_hits=sorted(set(_contains_any(text, config.illegal_keywords) + _match_aliases(text, config.illegal_aliases))),
                contact_hits=sorted(set(_contains_any(text, config.contact_keywords))),
                phones=sorted({item for item in (_normalize_phone(raw) for raw in phone_pattern.findall(text)) if item}),
                wechat_ids=sorted(set(wechat_pattern.findall(text))),
                qqs=sorted(set(qq_pattern.findall(text))),
                noise_hits=_contains_any(text, config.noise_keywords + extra_noise_keywords),
            )
        )
    return evidence_list


def _candidate_evidence(candidate: CardCandidate, evidence_list: list[BoxEvidence]) -> list[BoxEvidence]:
    x1, y1, x2, y2 = candidate.bbox
    expanded = _expand_bbox(candidate.bbox, max(10, (x2 - x1) // 8), max(10, (y2 - y1) // 8))
    matched: list[BoxEvidence] = []
    for item in evidence_list:
        bx1, by1, bx2, by2 = _box_bounds(item.block.points)
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2
        if _bbox_contains(expanded, center_x, center_y) or _intersects(expanded, (bx1, by1, bx2, by2)):
            matched.append(item)
    return matched


def _build_text_regions(evidence_list: list[BoxEvidence]) -> list[tuple[list[list[int]], tuple[int, int, int, int], list[BoxEvidence]]]:
    config = get_rule_config()
    anchors = [item for item in evidence_list if item.medical_hits or item.illegal_hits or item.phones or item.wechat_ids or item.qqs]
    used: set[int] = set()
    regions: list[tuple[list[list[int]], tuple[int, int, int, int], list[BoxEvidence]]] = []

    for anchor in anchors:
        if anchor.index in used:
            continue
        anchor_bbox = _box_bounds(anchor.block.points)
        anchor_center_x = (anchor_bbox[0] + anchor_bbox[2]) / 2
        anchor_center_y = (anchor_bbox[1] + anchor_bbox[3]) / 2
        anchor_w = max(20, anchor_bbox[2] - anchor_bbox[0])
        anchor_h = max(20, anchor_bbox[3] - anchor_bbox[1])
        limit_x = max(70, int(anchor_w * config.group_limit_x_ratio))
        limit_y = max(120, int(anchor_h * config.group_limit_y_ratio))
        selected: dict[int, BoxEvidence] = {anchor.index: anchor}

        for item in evidence_list:
            if item.index in selected:
                continue
            if item.noise_hits and not (item.medical_hits or item.illegal_hits or item.phones or item.wechat_ids or item.qqs):
                continue
            item_bbox = _box_bounds(item.block.points)
            item_center_x = (item_bbox[0] + item_bbox[2]) / 2
            item_center_y = (item_bbox[1] + item_bbox[3]) / 2
            if abs(item_center_x - anchor_center_x) <= limit_x and abs(item_center_y - anchor_center_y) <= limit_y:
                selected[item.index] = item

        merged = (
            min(_box_bounds(item.block.points)[0] for item in selected.values()),
            min(_box_bounds(item.block.points)[1] for item in selected.values()),
            max(_box_bounds(item.block.points)[2] for item in selected.values()),
            max(_box_bounds(item.block.points)[3] for item in selected.values()),
        )
        padding_x = max(config.min_region_padding, int((merged[2] - merged[0]) * config.region_padding_x_ratio))
        padding_y = max(config.min_region_padding, int((merged[3] - merged[1]) * config.region_padding_y_ratio))
        bbox = (merged[0] - padding_x, merged[1] - padding_y, merged[2] + padding_x, merged[3] + padding_y)
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        duplicate = False
        for _, existing_bbox, existing_items in regions:
            overlap = _iou(existing_bbox, bbox)
            shared_boxes = len({item.index for item in existing_items} & {item.index for item in selected.values()})
            if overlap >= 0.35 or shared_boxes >= max(1, min(len(existing_items), len(selected)) // 2):
                duplicate = True
                break
        if not duplicate:
            regions.append((points, bbox, list(selected.values())))
        used.update(selected.keys())

    return regions


def _refine_region(candidate_items: list[BoxEvidence], fallback_points: list[list[int]]) -> tuple[list[list[int]], tuple[int, int, int, int]]:
    point_cloud: list[list[float]] = []
    for item in candidate_items:
        point_cloud.extend([[float(point[0]), float(point[1])] for point in item.block.points])
    if len(point_cloud) < 4:
        xs = [point[0] for point in fallback_points]
        ys = [point[1] for point in fallback_points]
        return fallback_points, (min(xs), min(ys), max(xs), max(ys))
    points = np.array(point_cloud, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    (center_x, center_y), (width, height), angle = rect
    refined_rect = ((center_x, center_y), (max(width, 24.0) * 1.55, max(height, 24.0) * 1.8), angle)
    refined_points = [[int(round(x)), int(round(y))] for x, y in cv2.boxPoints(refined_rect)]
    xs = [point[0] for point in refined_points]
    ys = [point[1] for point in refined_points]
    return refined_points, (min(xs), min(ys), max(xs), max(ys))


def _clip_region(points: list[list[int]], image_shape: tuple[int, int, int] | None) -> tuple[list[list[int]], tuple[int, int, int, int]]:
    if image_shape is None:
        clipped = [[max(0, point[0]), max(0, point[1])] for point in points]
    else:
        height, width = image_shape[:2]
        clipped = [[min(max(0, point[0]), width - 1), min(max(0, point[1]), height - 1)] for point in points]
    xs = [point[0] for point in clipped]
    ys = [point[1] for point in clipped]
    return clipped, (min(xs), min(ys), max(xs), max(ys))


def _expand_region_with_nearby_hits(region_items: list[BoxEvidence], all_items: list[BoxEvidence]) -> list[BoxEvidence]:
    if not region_items:
        return region_items
    region_bbox = (
        min(_box_bounds(item.block.points)[0] for item in region_items),
        min(_box_bounds(item.block.points)[1] for item in region_items),
        max(_box_bounds(item.block.points)[2] for item in region_items),
        max(_box_bounds(item.block.points)[3] for item in region_items),
    )
    width = max(40, region_bbox[2] - region_bbox[0])
    height = max(40, region_bbox[3] - region_bbox[1])
    expanded_bbox = _expand_bbox(region_bbox, max(40, int(width * 0.35)), max(40, int(height * 0.35)))
    region_angles = [estimate_block_angle(item.block) for item in region_items if item.medical_hits or item.illegal_hits]
    region_angle = float(np.median(region_angles)) if region_angles else 0.0

    selected: dict[int, BoxEvidence] = {item.index: item for item in region_items}
    for item in all_items:
        if item.index in selected or not (item.medical_hits or item.illegal_hits):
            continue
        if _is_noise_only_item(item):
            continue
        bbox = _box_bounds(item.block.points)
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        if not (_bbox_contains(expanded_bbox, center_x, center_y) or _intersects(expanded_bbox, bbox)):
            continue
        angle_diff = abs(estimate_block_angle(item.block) - region_angle)
        angle_diff = min(angle_diff, abs(angle_diff - 90.0))
        if angle_diff <= 25:
            selected[item.index] = item
    return list(selected.values())


def _score_candidate(
    candidate_items: list[BoxEvidence],
    *,
    all_items: list[BoxEvidence],
    image: np.ndarray | None = None,
    request_id: str = "rule-evaluate",
    group_index: int = 0,
    global_noise_hits: set[str] | None = None,
) -> tuple[int, list[str], list[str], list[str], list[str], list[str], list[str], list[int]]:
    config = get_rule_config()
    scores = config.scores
    ordered_items = sorted(candidate_items, key=lambda item: (((_box_bounds(item.block.points)[1] + _box_bounds(item.block.points)[3]) / 2), _box_bounds(item.block.points)[0]))
    combined_text = "".join(item.block.text.strip() for item in ordered_items if item.block.text.strip())
    single_chars = {item.block.text.strip() for item in ordered_items if len(item.block.text.strip()) == 1}
    inferred_medical_hits: set[str] = set()
    inferred_illegal_hits: set[str] = set()
    if {"医", "保"} <= single_chars:
        inferred_medical_hits.add("医保")
    if {"医", "保", "卡"} <= single_chars:
        inferred_medical_hits.add("医保卡")
    if {"取", "现"} <= single_chars:
        inferred_illegal_hits.add("取现")
    if {"提", "现"} <= single_chars:
        inferred_illegal_hits.add("提现")
    medical_hits = sorted(
        {
            word
            for word in (
                {word for item in candidate_items for word in item.medical_hits}
                | set(_contains_any(combined_text, config.medical_keywords))
                | set(_match_aliases(combined_text, config.medical_aliases))
                | inferred_medical_hits
            )
        }
    )
    illegal_hits = sorted(
        {
            word
            for word in (
                {word for item in candidate_items for word in item.illegal_hits}
                | set(_contains_any(combined_text, config.illegal_keywords))
                | set(_match_aliases(combined_text, config.illegal_aliases))
                | inferred_illegal_hits
            )
        }
    )
    phones, wechat_ids, qqs, attached_contact_indices = filter_attached_contacts(
        candidate_items,
        all_items=all_items,
        image=image,
        request_id=request_id,
        group_index=group_index,
    )
    phone_pattern = re.compile(config.phone_pattern)
    combined_phones = sorted({item for item in (_normalize_phone(raw) for raw in phone_pattern.findall(combined_text)) if item})
    if combined_phones:
        phones = sorted(set(phones) | set(combined_phones))
    noise_hits = sorted({word for item in candidate_items for word in item.noise_hits} | set(global_noise_hits or set()))

    score = 0
    hit_rules: list[str] = []
    if medical_hits:
        score += min(scores.get("medical_cap", 35), scores.get("medical_base", 12) + len(medical_hits) * scores.get("medical_per_hit", 6))
        hit_rules.append("medical.keyword")
    if illegal_hits:
        score += min(scores.get("illegal_cap", 40), scores.get("illegal_base", 15) + len(illegal_hits) * scores.get("illegal_per_hit", 8))
        hit_rules.append("illegal.keyword")
    if phones:
        score += scores.get("phone", 24)
        hit_rules.append("contact.phone")
    if wechat_ids:
        score += scores.get("wechat", 18)
        hit_rules.append("contact.wechat")
    if qqs:
        score += scores.get("qq", 16)
        hit_rules.append("contact.qq")
    if medical_hits and (phones or wechat_ids or qqs):
        score += scores.get("medical_with_contact", 20)
        hit_rules.append("combo.medical_contact")
    if medical_hits and illegal_hits:
        score += scores.get("medical_illegal_combo", 12)
        hit_rules.append("combo.medical_illegal")
    if len(illegal_hits) >= 2:
        score += scores.get("illegal_multi_bonus", 10)
        hit_rules.append("combo.multi_illegal")
    if phones and not medical_hits and not illegal_hits:
        if not noise_hits:
            score = max(score, scores.get("phone_only_floor", 48))
            hit_rules.append("fallback.phone_only")
        else:
            logger.info("score_candidate fallback.phone_only skipped phones=%s reason=noise_hits noise_hits=%s", phones, noise_hits)
    if medical_hits:
        # Business rule: once medical-ad keywords are recovered, treat the region
        # as suspicious at minimum instead of leaving it below the suspicious floor.
        score = max(score, config.suspicious_score)
    return min(score, 100), sorted(set(hit_rules)), medical_hits, illegal_hits, phones, wechat_ids, qqs, attached_contact_indices


def _as_blocks(records: list[OCRRecord] | list[OCRBlockInput]) -> list[OCRBlock]:
    return [OCRBlock(text=item.text, points=item.points, confidence=item.confidence) for item in records]


def evaluate_blocks(blocks: list[OCRBlock], image: np.ndarray | None = None, request_id: str = "rule-evaluate") -> RuleEvaluationResponse:
    config = get_rule_config()
    evidence_list = _build_evidence(blocks)
    global_noise_hits = {word for item in evidence_list for word in item.noise_hits}
    ads: list[AdRegion] = []
    groups: list[tuple[list[list[int]], tuple[int, int, int, int], list[BoxEvidence]]] = []

    if image is not None:
        for candidate in detect_card_candidates(image):
            items = _candidate_evidence(candidate, evidence_list)
            if items:
                groups.append((candidate.box, candidate.bbox, items))

    groups.extend(_build_text_regions(evidence_list))
    logger.info("evaluate_blocks request_id=%s groups_count=%s phones=%s global_noise_hits=%s", request_id, len(groups), sorted({phone for item in evidence_list for phone in item.phones}), sorted(global_noise_hits))

    for group_index, (region_points, region_bbox, items) in enumerate(groups, start=1):
        score, hit_rules, medical_hits, illegal_hits, phones, wechat_ids, qqs, attached_contact_indices = _score_candidate(
            items,
            all_items=evidence_list,
            image=image,
            request_id=request_id,
            group_index=group_index,
            global_noise_hits=global_noise_hits,
        )
        logger.info(
            "evaluate_blocks request_id=%s group=%s score=%s phones=%s medical_hits=%s illegal_hits=%s hit_rules=%s block_indices=%s attached_contact_indices=%s",
            request_id,
            group_index,
            score,
            phones,
            medical_hits,
            illegal_hits,
            hit_rules,
            [item.index for item in items],
            attached_contact_indices,
        )
        if not _should_keep_region(score, medical_hits, illegal_hits, phones, wechat_ids, qqs):
            logger.info(
                "evaluate_blocks request_id=%s group=%s skipped reason=no_effective_signal suspicious_score=%s medical_hits=%s illegal_hits=%s",
                request_id,
                group_index,
                config.suspicious_score,
                medical_hits,
                illegal_hits,
            )
            continue
        attached_contact_set = set(attached_contact_indices)
        region_items = [
            item
            for item in items
            if not collect_contact_items([item]) or item.index in attached_contact_set
        ]
        if not region_items:
            region_items = items
        region_items = _expand_region_with_nearby_hits(region_items, evidence_list)
        refined_points, _ = _refine_region(region_items, region_points)
        refined_points, refined_bbox = _clip_region(refined_points, image.shape if image is not None else None)
        ad = AdRegion(
            ad_index=len(ads) + 1,
            points=refined_points,
            x1=refined_bbox[0],
            y1=refined_bbox[1],
            x2=refined_bbox[2],
            y2=refined_bbox[3],
            block_indices=[item.index for item in region_items],
            source_texts=[item.block.text for item in region_items],
            hit_keywords=sorted(set(medical_hits + illegal_hits)),
            hit_rules=hit_rules,
            phones=phones,
            wechat_ids=wechat_ids,
            qqs=qqs,
            risk_score=score,
            risk_level=_risk_level(score, config),  # type: ignore[arg-type]
            suspicious=_is_suspicious_score(score, config, medical_hits, illegal_hits),
        )

        duplicate_index = -1
        for index, existing in enumerate(ads):
            overlap = _iou((existing.x1, existing.y1, existing.x2, existing.y2), (ad.x1, ad.y1, ad.x2, ad.y2))
            shared = len(set(existing.block_indices) & set(ad.block_indices))
            if overlap >= 0.45 or shared >= max(1, min(len(existing.block_indices), len(ad.block_indices)) // 2):
                duplicate_index = index
                break
        if duplicate_index >= 0:
            if ads[duplicate_index].risk_score >= ad.risk_score:
                continue
            ads.pop(duplicate_index)
        ads.append(ad)

    block_matches: dict[int, dict[str, set[str]]] = {}
    for item in evidence_list:
        keywords = set(item.medical_hits + item.illegal_hits)
        rules: set[str] = set()
        clue_types: set[str] = set()
        if item.medical_hits:
            rules.add("medical.keyword")
            clue_types.add("risk_word")
        if item.illegal_hits:
            rules.add("illegal.keyword")
            clue_types.add("risk_word")
        if item.phones:
            rules.add("contact.phone")
            clue_types.add("phone")
        if item.wechat_ids:
            rules.add("contact.wechat")
            clue_types.add("wechat")
        if item.qqs:
            rules.add("contact.qq")
            clue_types.add("qq")
        if keywords or rules:
            block_matches[item.index] = {"keywords": keywords, "rules": rules, "types": clue_types}

    updated_blocks: list[OCRBlock] = []
    for index, block in enumerate(blocks):
        matches = block_matches.get(index)
        updated_blocks.append(
            OCRBlock(
                text=block.text,
                points=block.points,
                confidence=block.confidence,
                matched=bool(matches),
                hit_keywords=sorted(matches["keywords"]) if matches else [],
                hit_rules=sorted(matches["rules"]) if matches else [],
                clue_types=sorted(matches["types"]) if matches else [],
            )
        )

    phones = sorted({phone for item in evidence_list for phone in item.phones})
    wechat_ids = sorted({wechat for item in evidence_list for wechat in item.wechat_ids})
    qqs = sorted({qq for item in evidence_list for qq in item.qqs})
    hit_keywords = sorted({keyword for block in updated_blocks for keyword in block.hit_keywords} | {keyword for ad in ads for keyword in ad.hit_keywords})
    hit_rules = sorted({rule for block in updated_blocks for rule in block.hit_rules} | {rule for ad in ads for rule in ad.hit_rules})
    risk_score = max((ad.risk_score for ad in ads), default=0)
    ocr_text = "\n".join(block.text for block in updated_blocks)
    if not ads:
        if not groups:
            reason = "no_groups"
        elif any(item.phones for item in evidence_list) and not any(item.medical_hits or item.illegal_hits for item in evidence_list):
            reason = "phone_only_without_business_keywords"
        elif global_noise_hits:
            reason = f"noise_keywords={sorted(global_noise_hits)}"
        else:
            reason = "all_groups_filtered"
        logger.info("evaluate_blocks request_id=%s ads_empty=true reason=%s hit_rules=%s hit_keywords=%s", request_id, reason, hit_rules, hit_keywords)
    else:
        logger.info("evaluate_blocks request_id=%s ads_empty=false ads_count=%s risk_score=%s", request_id, len(ads), risk_score)
    return RuleEvaluationResponse(
        request_id=request_id,
        ocr_text=ocr_text,
        ocr_blocks=updated_blocks,
        phones=phones,
        wechat_ids=wechat_ids,
        qqs=qqs,
        hit_keywords=hit_keywords,
        hit_rules=hit_rules,
        risk_score=risk_score,
        risk_level=_risk_level(risk_score, config),  # type: ignore[arg-type]
        suspicious=any(ad.suspicious for ad in ads),
        ads=ads,
    )


def evaluate_request(payload: RuleEvaluateRequest) -> RuleEvaluationResponse:
    blocks = _as_blocks(payload.ocr_blocks)
    if not blocks and payload.ocr_text:
        blocks = [OCRBlock(text=line, points=[[0, 0], [0, 0], [0, 0], [0, 0]], confidence=0.0) for line in payload.ocr_text.splitlines() if line.strip()]
    result = evaluate_blocks(blocks=blocks, image=None, request_id=payload.request_id or "rule-evaluate")
    if payload.ocr_text and not result.ocr_text:
        result.ocr_text = payload.ocr_text
    return result
