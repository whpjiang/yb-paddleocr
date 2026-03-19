# 接口对接文档

本文档用于业务系统对接 `medical-ad-ocr-tools`。

基础信息：

- 服务默认地址：`http://127.0.0.1:8000`
- 文档地址：`/docs`
- 返回格式：`application/json`

## 1. 健康检查

接口：

```http
GET /health
```

响应示例：

```json
{
  "status": "ok",
  "service": "medical-ad-ocr-tools",
  "version": "0.1.0"
}
```

字段说明：

- `status`：服务状态，固定为 `ok`
- `service`：服务名称
- `version`：服务版本

## 2. OCR 分析接口

接口：

```http
POST /tools/medical-ad/ocr/analyze
Content-Type: application/json
```

请求体示例：

```json
{
  "request_id": "task-008",
  "image_url": "https://example.com/demo.jpg"
}
```

请求字段说明：

- `request_id`
  - 类型：`string`
  - 必填：否
  - 含义：业务方请求标识，用于日志追踪。未传时服务会自动生成。
- `image_url`
  - 类型：`string`
  - 必填：是
  - 含义：待识别图片的可访问 URL。

响应字段说明：

- `request_id`
  - 类型：`string`
  - 含义：最终请求标识。
- `image_source`
  - 类型：`string`
  - 含义：图片来源，通常为原始 `image_url`。
- `ocr_text`
  - 类型：`string`
  - 含义：OCR 全文，按识别顺序换行拼接。
- `ocr_confidence`
  - 类型：`number`
  - 含义：OCR 文字块平均置信度。
- `ocr_blocks`
  - 类型：`array`
  - 含义：OCR 明细块列表。
- `phones`
  - 类型：`array[string]`
  - 含义：抽取到的手机号列表。
- `wechat_ids`
  - 类型：`array[string]`
  - 含义：抽取到的微信号列表。
- `qqs`
  - 类型：`array[string]`
  - 含义：抽取到的 QQ 号列表。
- `hit_keywords`
  - 类型：`array[string]`
  - 含义：全图命中的风险关键词汇总。
- `hit_rules`
  - 类型：`array[string]`
  - 含义：全图命中的规则编码汇总。
- `risk_score`
  - 类型：`integer`
  - 含义：风险分，范围 `0~100`。
- `risk_level`
  - 类型：`string`
  - 含义：风险等级，取值为 `low`、`medium`、`high`。
- `suspicious`
  - 类型：`boolean`
  - 含义：是否判定为疑似医保非法小广告。
- `ads`
  - 类型：`array`
  - 含义：识别出的疑似广告区域列表。
- `round1_triggered_focus_retry`
  - 类型：`boolean`
  - 含义：首轮 OCR 后是否触发了“重点区域二次补识别”。
- `focus_retry_reason`
  - 类型：`string`
  - 含义：触发重点补识别的原因。
- `focus_region`
  - 类型：`object|null`
  - 含义：二次补识别选中的重点区域。
- `focus_retry_added_boxes`
  - 类型：`integer`
  - 含义：二次补识别相比首轮新增的 OCR 文字块数量。
- `focus_retry_variant`
  - 类型：`string`
  - 含义：二次补识别最终采用的候选版本，可能为 `normal`、`mirrored`、`rotate_90`、`rotate_180`、`rotate_270`、`deskew`、`perspective`。
- `focus_retry_semantic_score`
  - 类型：`number`
  - 含义：重点补识别最终候选的业务语义得分。
- `round1_low_semantic_confidence`
  - 类型：`boolean`
  - 含义：首轮 OCR 是否存在“识别结果置信度看起来不低，但业务语义明显不可信”的情况。
- `selected_by_semantic_score`
  - 类型：`boolean`
  - 含义：最终 focus retry 候选是否是按语义得分优选出来的。
- `focus_region_angle`
  - 类型：`number`
  - 含义：重点区域估计偏斜角度，单位为度。
- `focus_region_shape`
  - 类型：`string`
  - 含义：重点区域形态，可能为 `rectangle`、`trapezoid`、`irregular`。
- `annotated_image_url`
  - 类型：`string|null`
  - 含义：命中风险且上传 OSS 成功后的标注图 URL。
- `annotated_image_oss_key`
  - 类型：`string|null`
  - 含义：标注图上传到 OSS 后的对象 Key。

### 2.1 `ocr_blocks` 字段结构

每个 `ocr_blocks` 元素结构如下：

```json
{
  "text": "保取现",
  "points": [[254, 575], [330, 547], [340, 575], [264, 603]],
  "confidence": 0.9184,
  "matched": true,
  "hit_keywords": ["取现"],
  "hit_rules": ["illegal.keyword"],
  "clue_types": ["risk_word"]
}
```

字段说明：

- `text`：当前文字块识别结果
- `points`：四点坐标，表示文字框位置，顺序为多边形四个顶点
- `confidence`：当前文字块识别置信度
- `matched`：当前文字块是否命中风险规则
- `hit_keywords`：当前文字块命中的关键词
- `hit_rules`：当前文字块命中的规则编码
- `clue_types`：当前文字块命中的线索类型，常见值如下：
  - `risk_word`：风险词
  - `phone`：手机号
  - `wechat`：微信号
  - `qq`：QQ 号

### 2.2 `ads` 字段结构

每个 `ads` 元素结构如下：

```json
{
  "ad_index": 1,
  "points": [[100, 200], [500, 200], [500, 700], [100, 700]],
  "x1": 100,
  "y1": 200,
  "x2": 500,
  "y2": 700,
  "block_indices": [0, 1, 2],
  "source_texts": ["医保", "取现", "微信 abc123"],
  "hit_keywords": ["医保", "取现"],
  "hit_rules": ["medical.keyword", "illegal.keyword", "contact.wechat"],
  "phones": [],
  "wechat_ids": ["abc123"],
  "qqs": [],
  "risk_score": 78,
  "risk_level": "medium",
  "suspicious": true
}
```

字段说明：

- `ad_index`：广告区域序号
- `points`：广告区域多边形坐标
- `x1/y1/x2/y2`：广告区域外接矩形
- `block_indices`：该广告区域关联到的 `ocr_blocks` 下标
- `source_texts`：该区域关联的原始文本
- `hit_keywords`：该区域命中的关键词
- `hit_rules`：该区域命中的规则
- `phones/wechat_ids/qqs`：该区域抽取出的联系方式
- `risk_score`：该区域风险分
- `risk_level`：该区域风险等级
- `suspicious`：该区域是否可疑

### 2.3 `focus_region` 字段结构

当 `round1_triggered_focus_retry=true` 时，`focus_region` 结构如下：

```json
{
  "x1": 0,
  "y1": 364,
  "x2": 717,
  "y2": 742,
  "score": 0.6967
}
```

字段说明：

- `x1/y1/x2/y2`：重点补识别区域坐标
- `score`：重点区域评分，分值越高代表越可疑

### 2.4 OCR 分析响应示例

```json
{
  "request_id": "task-008",
  "ocr_text": "保取现\n3734436636",
  "ocr_blocks": [
    {
      "text": "保取现",
      "points": [[254, 575], [330, 547], [340, 575], [264, 603]],
      "confidence": 0.9184093475341797,
      "matched": true,
      "hit_keywords": ["取现"],
      "hit_rules": ["illegal.keyword"],
      "clue_types": ["risk_word"]
    },
    {
      "text": "3734436636",
      "points": [[251, 597], [339, 562], [352, 595], [265, 630]],
      "confidence": 0.7499339580535889,
      "matched": false,
      "hit_keywords": [],
      "hit_rules": [],
      "clue_types": []
    }
  ],
  "phones": [],
  "wechat_ids": [],
  "qqs": [],
  "hit_keywords": ["取现"],
  "hit_rules": ["illegal.keyword"],
  "risk_score": 0,
  "risk_level": "low",
  "suspicious": false,
  "ads": [],
  "image_source": "https://example.com/demo.jpg",
  "ocr_confidence": 0.8342,
  "round1_triggered_focus_retry": false,
  "focus_retry_reason": "",
  "focus_region": null,
  "focus_retry_added_boxes": 0,
  "focus_retry_variant": "",
  "focus_retry_semantic_score": 0.0,
  "round1_low_semantic_confidence": false,
  "selected_by_semantic_score": false,
  "focus_region_angle": 0.0,
  "focus_region_shape": "",
  "annotated_image_url": null,
  "annotated_image_oss_key": null
}
```

说明：

- 该结果表示 OCR 已识别到风险词 `取现`
- 但未识别到完整联系方式
- 也未形成有效广告区域
- 因此最终风险分为 `0`，风险等级为 `low`

## 3. 规则评估接口

接口：

```http
POST /tools/medical-ad/rule/evaluate
Content-Type: application/json
```

请求体示例：

```json
{
  "request_id": "rule-001",
  "ocr_text": "医保卡高价回收\n微信 abc123",
  "ocr_blocks": [
    {
      "text": "医保卡高价回收",
      "points": [[10, 10], [240, 10], [240, 50], [10, 50]],
      "confidence": 0.99
    },
    {
      "text": "微信 abc123",
      "points": [[10, 60], [220, 60], [220, 100], [10, 100]],
      "confidence": 0.98
    }
  ]
}
```

请求字段说明：

- `request_id`
  - 类型：`string`
  - 必填：否
  - 含义：业务请求标识。
- `ocr_text`
  - 类型：`string`
  - 必填：否
  - 含义：OCR 全文。若 `ocr_blocks` 为空，可只传全文。
- `ocr_blocks`
  - 类型：`array`
  - 必填：否
  - 含义：OCR 明细块。若传入，将优先基于文字块做规则识别。

约束：

- `ocr_text` 和 `ocr_blocks` 至少要传一个。

响应结构：

- 与 `POST /tools/medical-ad/ocr/analyze` 基本一致
- 不包含 `image_source`、`ocr_confidence`、`round1_triggered_focus_retry`、`focus_retry_reason`、`focus_region`、`focus_retry_added_boxes`、`annotated_image_url`、`annotated_image_oss_key`

## 4. 规则编码说明

当前常见规则编码：

- `medical.keyword`：命中医保相关关键词
- `illegal.keyword`：命中套现、回收、取现等违规关键词
- `combo.medical_illegal`：医保相关关键词和违规关键词同时出现
- `contact.phone`：命中手机号
- `contact.wechat`：命中微信号
- `contact.qq`：命中 QQ 号
- `combo.medical_contact`：医保关键词和联系方式同时出现
- `combo.multi_illegal`：命中多个违规关键词
- `fallback.phone_only`：仅手机号但达到兜底判定条件

## 5. 对接建议

- 业务系统至少保存：`request_id`、`risk_score`、`risk_level`、`suspicious`、`hit_keywords`、`hit_rules`
- 若需要前端回显，建议同时保存：`ocr_blocks`、`ads`、`annotated_image_url`
- 若你们自己已有 OCR，也可以只对接 `POST /tools/medical-ad/rule/evaluate`
