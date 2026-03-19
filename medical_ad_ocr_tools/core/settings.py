from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from medical_ad_ocr_tools.core.models import FlatConfig


class AppConfig(BaseModel):
    name: str = "medical-ad-ocr-tools"
    version: str = "0.1.0"


class RuntimeConfig(BaseModel):
    temp_dir: Path = Path("storage/tmp")
    output_dir: Path = Path("storage/annotated")
    request_timeout_seconds: int = 15


class OCRConfig(BaseModel):
    lang: str = "ch"
    version: str = "PP-OCRv5"
    use_textline_orientation: bool = True
    text_detection_model_name: str | None = None
    text_recognition_model_name: str | None = None
    text_detection_model_dir: Path | None = None
    text_recognition_model_dir: Path | None = None
    textline_orientation_model_dir: Path | None = None
    text_det_limit_side_len: int = 1536
    text_det_box_thresh: float = 0.45
    text_det_thresh: float = 0.25
    text_det_unclip_ratio: float = 1.6
    text_rec_score_thresh: float = 0.3
    focus_retry_enabled: bool = True
    focus_retry_max_regions: int = 1
    focus_retry_scale: float = 1.8
    focus_retry_min_region_score: float = 0.55
    focus_retry_min_keyword_hits: int = 2
    focus_retry_mid_risk_min: int = 40
    focus_retry_mid_risk_max: int = 69
    focus_retry_enable_sharpen: bool = True
    focus_retry_enable_contrast: bool = True
    focus_retry_enable_rotate: bool = False
    focus_retry_enable_mirror: bool = True
    focus_retry_enable_perspective: bool = True
    focus_retry_deskew_angle_min: float = 8.0
    focus_retry_deskew_angle_max: float = 20.0


class AnnotationConfig(BaseModel):
    quality: int = 82
    max_edge: int = 1800
    line_width: int = 4
    font_size_min: int = 20
    font_size_max: int = 42


class OSSConfig(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    bucket_name: str | None = None
    access_key_id: str | None = None
    access_key_secret: str | None = None
    prefix: str = "medical-ad-ocr-tools"
    public_base_url: str | None = None


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    oss: OSSConfig = Field(default_factory=OSSConfig)
    rules_path: Path = Path("config/rules.yaml")
    config_path: Path = Path("config/config.yaml")

    def to_flat(self) -> FlatConfig:
        return FlatConfig(
            app_name=self.app.name,
            version=self.app.version,
            temp_dir=str(self.runtime.temp_dir),
            output_dir=str(self.runtime.output_dir),
            request_timeout_seconds=self.runtime.request_timeout_seconds,
            ocr_lang=self.ocr.lang,
            ocr_version=self.ocr.version,
            use_textline_orientation=self.ocr.use_textline_orientation,
            text_detection_model_name=self.ocr.text_detection_model_name,
            text_recognition_model_name=self.ocr.text_recognition_model_name,
            text_detection_model_dir=str(self.ocr.text_detection_model_dir) if self.ocr.text_detection_model_dir else None,
            text_recognition_model_dir=str(self.ocr.text_recognition_model_dir) if self.ocr.text_recognition_model_dir else None,
            textline_orientation_model_dir=str(self.ocr.textline_orientation_model_dir) if self.ocr.textline_orientation_model_dir else None,
            text_det_limit_side_len=self.ocr.text_det_limit_side_len,
            text_det_box_thresh=self.ocr.text_det_box_thresh,
            text_det_thresh=self.ocr.text_det_thresh,
            text_det_unclip_ratio=self.ocr.text_det_unclip_ratio,
            text_rec_score_thresh=self.ocr.text_rec_score_thresh,
            focus_retry_enabled=self.ocr.focus_retry_enabled,
            focus_retry_max_regions=self.ocr.focus_retry_max_regions,
            focus_retry_scale=self.ocr.focus_retry_scale,
            focus_retry_min_region_score=self.ocr.focus_retry_min_region_score,
            focus_retry_min_keyword_hits=self.ocr.focus_retry_min_keyword_hits,
            focus_retry_mid_risk_min=self.ocr.focus_retry_mid_risk_min,
            focus_retry_mid_risk_max=self.ocr.focus_retry_mid_risk_max,
            focus_retry_enable_sharpen=self.ocr.focus_retry_enable_sharpen,
            focus_retry_enable_contrast=self.ocr.focus_retry_enable_contrast,
            focus_retry_enable_rotate=self.ocr.focus_retry_enable_rotate,
            focus_retry_enable_mirror=self.ocr.focus_retry_enable_mirror,
            focus_retry_enable_perspective=self.ocr.focus_retry_enable_perspective,
            focus_retry_deskew_angle_min=self.ocr.focus_retry_deskew_angle_min,
            focus_retry_deskew_angle_max=self.ocr.focus_retry_deskew_angle_max,
            annotation_quality=self.annotation.quality,
            annotation_max_edge=self.annotation.max_edge,
            annotation_line_width=self.annotation.line_width,
            annotation_font_size_min=self.annotation.font_size_min,
            annotation_font_size_max=self.annotation.font_size_max,
            oss_enabled=self.oss.enabled,
            oss_endpoint=self.oss.endpoint,
            oss_bucket_name=self.oss.bucket_name,
            oss_access_key_id=self.oss.access_key_id,
            oss_access_key_secret=self.oss.access_key_secret,
            oss_prefix=self.oss.prefix,
            oss_public_base_url=self.oss.public_base_url,
            rules_path=str(self.rules_path),
            config_path=str(self.config_path),
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _set_nested(target: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = target
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, tuple[str, Any]] = {
        "MEDICAL_AD_APP_NAME": ("app.name", str),
        "MEDICAL_AD_APP_VERSION": ("app.version", str),
        "MEDICAL_AD_TEMP_DIR": ("runtime.temp_dir", Path),
        "MEDICAL_AD_OUTPUT_DIR": ("runtime.output_dir", Path),
        "MEDICAL_AD_REQUEST_TIMEOUT_SECONDS": ("runtime.request_timeout_seconds", int),
        "MEDICAL_AD_OCR_LANG": ("ocr.lang", str),
        "MEDICAL_AD_OCR_VERSION": ("ocr.version", str),
        "MEDICAL_AD_USE_TEXTLINE_ORIENTATION": ("ocr.use_textline_orientation", _parse_bool),
        "MEDICAL_AD_TEXT_DETECTION_MODEL_NAME": ("ocr.text_detection_model_name", str),
        "MEDICAL_AD_TEXT_RECOGNITION_MODEL_NAME": ("ocr.text_recognition_model_name", str),
        "MEDICAL_AD_TEXT_DETECTION_MODEL_DIR": ("ocr.text_detection_model_dir", Path),
        "MEDICAL_AD_TEXT_RECOGNITION_MODEL_DIR": ("ocr.text_recognition_model_dir", Path),
        "MEDICAL_AD_TEXTLINE_ORIENTATION_MODEL_DIR": ("ocr.textline_orientation_model_dir", Path),
        "MEDICAL_AD_TEXT_DET_LIMIT_SIDE_LEN": ("ocr.text_det_limit_side_len", int),
        "MEDICAL_AD_TEXT_DET_BOX_THRESH": ("ocr.text_det_box_thresh", float),
        "MEDICAL_AD_TEXT_DET_THRESH": ("ocr.text_det_thresh", float),
        "MEDICAL_AD_TEXT_DET_UNCLIP_RATIO": ("ocr.text_det_unclip_ratio", float),
        "MEDICAL_AD_TEXT_REC_SCORE_THRESH": ("ocr.text_rec_score_thresh", float),
        "MEDICAL_AD_FOCUS_RETRY_ENABLED": ("ocr.focus_retry_enabled", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_MAX_REGIONS": ("ocr.focus_retry_max_regions", int),
        "MEDICAL_AD_FOCUS_RETRY_SCALE": ("ocr.focus_retry_scale", float),
        "MEDICAL_AD_FOCUS_RETRY_MIN_REGION_SCORE": ("ocr.focus_retry_min_region_score", float),
        "MEDICAL_AD_FOCUS_RETRY_MIN_KEYWORD_HITS": ("ocr.focus_retry_min_keyword_hits", int),
        "MEDICAL_AD_FOCUS_RETRY_MID_RISK_MIN": ("ocr.focus_retry_mid_risk_min", int),
        "MEDICAL_AD_FOCUS_RETRY_MID_RISK_MAX": ("ocr.focus_retry_mid_risk_max", int),
        "MEDICAL_AD_FOCUS_RETRY_ENABLE_SHARPEN": ("ocr.focus_retry_enable_sharpen", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_ENABLE_CONTRAST": ("ocr.focus_retry_enable_contrast", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_ENABLE_ROTATE": ("ocr.focus_retry_enable_rotate", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_ENABLE_MIRROR": ("ocr.focus_retry_enable_mirror", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_ENABLE_PERSPECTIVE": ("ocr.focus_retry_enable_perspective", _parse_bool),
        "MEDICAL_AD_FOCUS_RETRY_DESKEW_ANGLE_MIN": ("ocr.focus_retry_deskew_angle_min", float),
        "MEDICAL_AD_FOCUS_RETRY_DESKEW_ANGLE_MAX": ("ocr.focus_retry_deskew_angle_max", float),
        "MEDICAL_AD_ANNOTATION_QUALITY": ("annotation.quality", int),
        "MEDICAL_AD_ANNOTATION_MAX_EDGE": ("annotation.max_edge", int),
        "MEDICAL_AD_ANNOTATION_LINE_WIDTH": ("annotation.line_width", int),
        "MEDICAL_AD_ANNOTATION_FONT_SIZE_MIN": ("annotation.font_size_min", int),
        "MEDICAL_AD_ANNOTATION_FONT_SIZE_MAX": ("annotation.font_size_max", int),
        "MEDICAL_AD_OSS_ENABLED": ("oss.enabled", _parse_bool),
        "MEDICAL_AD_OSS_ENDPOINT": ("oss.endpoint", str),
        "MEDICAL_AD_OSS_BUCKET_NAME": ("oss.bucket_name", str),
        "MEDICAL_AD_OSS_ACCESS_KEY_ID": ("oss.access_key_id", str),
        "MEDICAL_AD_OSS_ACCESS_KEY_SECRET": ("oss.access_key_secret", str),
        "MEDICAL_AD_OSS_PREFIX": ("oss.prefix", str),
        "MEDICAL_AD_OSS_PUBLIC_BASE_URL": ("oss.public_base_url", str),
        "MEDICAL_AD_RULES_PATH": ("rules_path", Path),
    }
    result = dict(data)
    for env_name, (path, parser) in overrides.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        _set_nested(result, path, parser(raw))
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(os.getenv("MEDICAL_AD_CONFIG_PATH", "config/config.yaml"))
    data = _read_yaml(config_path)
    data["config_path"] = config_path
    merged = _apply_env_overrides(data)
    settings = Settings.model_validate(merged)
    settings.runtime.temp_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime.output_dir.mkdir(parents=True, exist_ok=True)
    settings.rules_path = Path(settings.rules_path)
    settings.config_path = Path(settings.config_path)
    return settings
