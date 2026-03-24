from __future__ import annotations

import json
from pathlib import Path

from medical_ad_ocr_tools.services.workflow import analyze_image


IMAGE_DIR = Path("storage/analysis")
IMAGE_NAMES = [
    "19cd1bf7211X0d.jpg",
    "19cd1c0add3kuh.jpg",
    "19cd1c505b33aG.jpg",
    "19cd1e9e674idX.jpg",
    "19cd1c8edf66kW.jpg",
]


def main() -> None:
    results: list[dict[str, object]] = []
    for name in IMAGE_NAMES:
        image_path = IMAGE_DIR / name
        result = analyze_image(
            request_id=f"diag-{image_path.stem}",
            image_path=image_path,
            image_source=f"local://{name}",
        )
        payload = result.response.model_dump()
        results.append(
            {
                "image": name,
                "ocr_text": payload["ocr_text"],
                "risk_score": payload["risk_score"],
                "suspicious": payload["suspicious"],
                "hit_keywords": payload["hit_keywords"],
                "hit_rules": payload["hit_rules"],
                "round1_triggered_focus_retry": payload["round1_triggered_focus_retry"],
                "focus_retry_reason": payload["focus_retry_reason"],
                "focus_region": payload["focus_region"],
                "focus_retry_variant": payload["focus_retry_variant"],
                "focus_retry_added_boxes": payload["focus_retry_added_boxes"],
                "ocr_blocks": [
                    {"text": block["text"], "confidence": block["confidence"], "matched": block["matched"]}
                    for block in payload["ocr_blocks"][:20]
                ],
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
