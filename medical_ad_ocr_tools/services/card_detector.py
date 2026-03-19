from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CardCandidate:
    box: list[list[int]]
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    inverse_perspective_matrix: list[list[float]]


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp_card(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rect = _order_points(points.astype(np.float32))
    width = max(int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))), 32)
    height = max(int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))), 32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, inverse_matrix


def detect_card_candidates(image: np.ndarray) -> list[CardCandidate]:
    height, width = image.shape[:2]
    image_area = height * width
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    seen: set[tuple[int, int, int, int]] = set()
    candidates: list[CardCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.0012 or area > image_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        points = cv2.boxPoints(cv2.minAreaRect(contour)) if len(approx) != 4 else approx.reshape(4, 2).astype(np.float32)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        if w < 28 or h < 28 or w > width * 0.55 or h > height * 0.55:
            continue
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            continue
        if max(w / max(h, 1), h / max(w, 1)) > 6.5:
            continue

        crop = image[y : y + h, x : x + w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        if float(np.mean(hsv[:, :, 1])) > 150 and float(np.std(hsv[:, :, 2])) < 18:
            continue

        warped, inverse_matrix = _warp_card(image, points)
        if warped.shape[0] < 28 or warped.shape[1] < 28:
            continue

        key = (x // 12, y // 12, (x + w) // 12, (y + h) // 12)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            CardCandidate(
                box=[[int(round(px)), int(round(py))] for px, py in points],
                bbox=(x, y, x + w, y + h),
                crop=warped,
                inverse_perspective_matrix=inverse_matrix.tolist(),
            )
        )

    candidates.sort(key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]), reverse=True)
    return candidates[:12]
