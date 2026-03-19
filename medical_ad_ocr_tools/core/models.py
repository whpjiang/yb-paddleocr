from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class ImageSourcePayload(BaseModel):
    request_id: str | None = Field(default=None, description="Optional request id from caller.")
    image_url: HttpUrl | None = Field(default=None, description="Remote image URL.")

    @model_validator(mode="after")
    def validate_source(self) -> "ImageSourcePayload":
        if not self.image_url:
            raise ValueError("image_url is required when request body is JSON.")
        return self


class OCRBlockInput(BaseModel):
    text: str
    points: list[list[int]]
    confidence: float = 0.0


class RuleEvaluateRequest(BaseModel):
    request_id: str | None = None
    ocr_text: str | None = None
    ocr_blocks: list[OCRBlockInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_content(self) -> "RuleEvaluateRequest":
        if not self.ocr_text and not self.ocr_blocks:
            raise ValueError("ocr_text or ocr_blocks is required.")
        return self


class OCRBlock(BaseModel):
    text: str
    points: list[list[int]]
    confidence: float
    matched: bool = False
    hit_keywords: list[str] = Field(default_factory=list)
    hit_rules: list[str] = Field(default_factory=list)
    clue_types: list[str] = Field(default_factory=list)


class AdRegion(BaseModel):
    ad_index: int
    points: list[list[int]]
    x1: int
    y1: int
    x2: int
    y2: int
    block_indices: list[int] = Field(default_factory=list)
    source_texts: list[str] = Field(default_factory=list)
    hit_keywords: list[str] = Field(default_factory=list)
    hit_rules: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    wechat_ids: list[str] = Field(default_factory=list)
    qqs: list[str] = Field(default_factory=list)
    risk_score: int = 0
    risk_level: Literal["low", "medium", "high"] = "low"
    suspicious: bool = False


class FocusRegion(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class RuleEvaluationResponse(BaseModel):
    request_id: str
    ocr_text: str
    ocr_blocks: list[OCRBlock]
    phones: list[str]
    wechat_ids: list[str]
    qqs: list[str]
    hit_keywords: list[str]
    hit_rules: list[str]
    risk_score: int
    risk_level: Literal["low", "medium", "high"]
    suspicious: bool
    ads: list[AdRegion]


class AnalyzeResponse(RuleEvaluationResponse):
    image_source: str
    ocr_confidence: float
    round1_triggered_focus_retry: bool = False
    focus_retry_reason: str = ""
    focus_region: FocusRegion | None = None
    focus_retry_added_boxes: int = 0
    annotated_image_url: str | None = None
    annotated_image_oss_key: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class OCRRecord(BaseModel):
    text: str
    points: list[list[int]]
    confidence: float


class OCRRunResult(BaseModel):
    records: list[OCRRecord]
    avg_confidence: float
    ocr_text: str
    round1_triggered_focus_retry: bool = False
    focus_retry_reason: str = ""
    focus_region: FocusRegion | None = None
    focus_retry_added_boxes: int = 0


class AnalyzeArtifacts(BaseModel):
    source_path: str
    source_label: str
    original_image_shape: tuple[int, int, int]
    annotated_image_path: str | None = None
    annotated_image_url: str | None = None
    annotated_image_oss_key: str | None = None


class WorkflowResult(BaseModel):
    response: AnalyzeResponse
    artifacts: AnalyzeArtifacts


class RuleConfig(BaseModel):
    medical_keywords: list[str]
    illegal_keywords: list[str]
    contact_keywords: list[str]
    noise_keywords: list[str]
    medical_aliases: dict[str, list[str]] = Field(default_factory=dict)
    illegal_aliases: dict[str, list[str]] = Field(default_factory=dict)
    phone_pattern: str
    wechat_pattern: str
    qq_pattern: str
    suspicious_score: int = 45
    high_risk_score: int = 80
    group_limit_x_ratio: float = 2.2
    group_limit_y_ratio: float = 5.0
    region_padding_x_ratio: float = 0.18
    region_padding_y_ratio: float = 0.20
    min_region_padding: int = 12
    scores: dict[str, int] = Field(default_factory=dict)


class FlatConfig(BaseModel):
    app_name: str
    version: str
    temp_dir: str
    output_dir: str
    request_timeout_seconds: int
    ocr_lang: str
    ocr_version: str
    use_textline_orientation: bool
    text_detection_model_name: str | None
    text_recognition_model_name: str | None
    text_detection_model_dir: str | None
    text_recognition_model_dir: str | None
    textline_orientation_model_dir: str | None
    text_det_limit_side_len: int
    text_det_box_thresh: float
    text_det_thresh: float
    text_det_unclip_ratio: float
    text_rec_score_thresh: float
    focus_retry_enabled: bool
    focus_retry_max_regions: int
    focus_retry_scale: float
    focus_retry_min_region_score: float
    focus_retry_min_keyword_hits: int
    focus_retry_mid_risk_min: int
    focus_retry_mid_risk_max: int
    focus_retry_enable_sharpen: bool
    focus_retry_enable_contrast: bool
    focus_retry_enable_rotate: bool
    annotation_quality: int
    annotation_max_edge: int
    annotation_line_width: int
    annotation_font_size_min: int
    annotation_font_size_max: int
    oss_enabled: bool
    oss_endpoint: str | None
    oss_bucket_name: str | None
    oss_access_key_id: str | None
    oss_access_key_secret: str | None
    oss_prefix: str
    oss_public_base_url: str | None
    rules_path: str
    config_path: str

    def to_safe_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        for field_name in ("oss_access_key_id", "oss_access_key_secret"):
            if data.get(field_name):
                data[field_name] = "***"
        return data
