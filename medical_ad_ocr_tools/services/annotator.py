from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from medical_ad_ocr_tools.core.models import OCRBlock, RuleEvaluationResponse
from medical_ad_ocr_tools.core.settings import get_settings

RED = (220, 38, 38)
YELLOW = (234, 179, 8)
WHITE = (255, 255, 255)


def _get_font(size: int) -> ImageFont.ImageFont:
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _line_color(block: OCRBlock) -> tuple[int, int, int]:
    if "risk_word" in block.clue_types:
        return RED
    return YELLOW


def _draw_block(draw: ImageDraw.ImageDraw, block: OCRBlock, line_width: int) -> None:
    draw.polygon([tuple(point) for point in block.points], outline=_line_color(block), width=line_width)


def draw_annotations(image_path: Path, request_id: str, evaluation: RuleEvaluationResponse) -> Path | None:
    if not evaluation.suspicious:
        return None
    settings = get_settings()
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    base = max(image.width, image.height)
    font_size = max(settings.annotation.font_size_min, min(settings.annotation.font_size_max, base // 28))
    line_width = max(settings.annotation.line_width, min(settings.annotation.line_width + 4, base // 180))
    font = _get_font(font_size)

    highlighted_indices = {index for ad in evaluation.ads for index in ad.block_indices}
    for index, block in enumerate(evaluation.ocr_blocks):
        if not block.matched:
            if index not in highlighted_indices:
                continue
        _draw_block(draw, block, line_width)

    for ad in evaluation.ads:
        draw.polygon([tuple(point) for point in ad.points], outline=RED, width=line_width + 1)
        label = f"医保广告 {ad.risk_level.upper()} {ad.risk_score}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(12, ad.x1)
        label_y = max(12, ad.y1 - text_height - 20)
        draw.rounded_rectangle((label_x, label_y, label_x + text_width + 20, label_y + text_height + 14), radius=8, fill=RED)
        draw.text((label_x + 10, label_y + 7), label, fill=WHITE, font=font)

    max_edge = settings.annotation.max_edge
    if max(image.width, image.height) > max_edge:
        ratio = max_edge / float(max(image.width, image.height))
        image = image.resize((int(image.width * ratio), int(image.height * ratio)))

    output_path = settings.runtime.output_dir / f"{request_id}.png"
    image.save(output_path, format="PNG", optimize=True)
    return output_path
