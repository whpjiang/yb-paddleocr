from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import paddle

from medical_ad_ocr_tools.services.ocr_service import get_ocr_engine, run_ocr
from medical_ad_ocr_tools.services.workflow import analyze_image


def main() -> None:
    image_path = Path(r"d:\Workspace\LearnAI\yb-PaddleOCR\storage\tmp\e8e4b0ed04b5a63c23549fc3a01d742d.jpg")
    image_source = "https://hyw-jiankangchuanboyuan.oss-cn-hangzhou.aliyuncs.com/2026-03-19/19d03b40308Oq0.jpg"
    request_id = "bench-local-20260319"

    print("device=", paddle.get_device())
    print("image_exists=", image_path.exists())

    warm0 = perf_counter()
    get_ocr_engine()
    warm1 = perf_counter()
    print(f"warm_engine_s={warm1 - warm0:.3f}")

    img = cv2.imread(str(image_path))
    print("image_shape=", None if img is None else img.shape)

    r0 = perf_counter()
    ocr_result = run_ocr(image_path, request_id=request_id)
    r1 = perf_counter()
    print(f"ocr_s={r1 - r0:.3f}")
    print("ocr_boxes=", len(ocr_result.records))
    print("focus_retry=", ocr_result.round1_triggered_focus_retry)
    print("focus_retry_reason=", ocr_result.focus_retry_reason)
    print("focus_region=", None if ocr_result.focus_region is None else ocr_result.focus_region.model_dump())
    print("focus_retry_added_boxes=", ocr_result.focus_retry_added_boxes)

    w0 = perf_counter()
    workflow_result = analyze_image(request_id=request_id, image_path=image_path, image_source=image_source)
    w1 = perf_counter()
    print(f"analyze_image_total_s={w1 - w0:.3f}")
    print("risk_score=", workflow_result.response.risk_score)
    print("risk_level=", workflow_result.response.risk_level)
    print("suspicious=", workflow_result.response.suspicious)
    print("phones=", workflow_result.response.phones)
    print("wechat_ids=", workflow_result.response.wechat_ids)
    print("qqs=", workflow_result.response.qqs)


if __name__ == "__main__":
    main()
