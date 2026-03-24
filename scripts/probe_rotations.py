from __future__ import annotations

import json
from pathlib import Path

import cv2

from medical_ad_ocr_tools.core.settings import get_settings
from medical_ad_ocr_tools.services.ocr_service import _deduplicate, _predict_candidate, get_ocr_engine
from medical_ad_ocr_tools.services.ocr_service import OCRCandidate


IMAGE_NAMES = [
    "19cd1bf7211X0d.jpg",
    "19cd1c0add3kuh.jpg",
    "19cd1c505b33aG.jpg",
    "19cd1e9e674idX.jpg",
]


def candidate_from_image(image, name: str) -> OCRCandidate:
    return OCRCandidate(
        name=name,
        variant=name,
        image=image,
        crop_width=image.shape[1],
        crop_height=image.shape[0],
        scale=1.0,
    )


def main() -> None:
    settings = get_settings()
    engine = get_ocr_engine()
    base = Path("storage/analysis")
    report: dict[str, object] = {}
    for name in IMAGE_NAMES:
        image = cv2.imread(str(base / name))
        variants = {
            "normal": image,
            "rot90": cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            "rot180": cv2.rotate(image, cv2.ROTATE_180),
            "rot270": cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        rows = []
        for variant_name, variant_image in variants.items():
            records = _predict_candidate(engine, candidate_from_image(variant_image, variant_name), settings.ocr.use_textline_orientation)
            records = _deduplicate(records)
            rows.append(
                {
                    "variant": variant_name,
                    "texts": [record.text for record in records[:12]],
                }
            )
        report[name] = rows
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
