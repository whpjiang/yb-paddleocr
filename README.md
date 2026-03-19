# medical-ad-ocr-tools

一个无状态的单体工具服务，面向业务系统提供医保非法小广告 OCR 识别、规则判定、标注压缩和 OSS 上传能力。

## 特性

- 支持图片上传和图片 URL 两种输入方式
- 基于 `PaddleOCR 3.4.0` 提取全文、文字框和置信度
- 保留多候选裁剪、多区域聚合、多小广告识别能力
- 提取手机号、微信号、QQ 号等线索
- 输出命中关键词、命中规则、风险分、风险等级和疑似标记
- 对命中风险区域进行标注并压缩后上传到阿里云 OSS
- 无数据库、无 Redis、无 Celery、无审核流转依赖

## 技术基线

- Python 3.10
- paddlepaddle==3.1.1
- paddleocr==3.4.0
- fastapi==0.115.12
- uvicorn[standard]==0.34.0
- pydantic==2.10.6
- opencv-python-headless==4.10.0.84
- Pillow==11.1.0
- numpy==1.26.4
- shapely==2.0.7
- python-multipart==0.0.20
- oss2==2.19.1

## 项目结构

```text
medical_ad_ocr_tools/
  core/
  services/
  main.py
config/
  config.yaml
  rules.yaml
storage/
  annotated/
  tmp/
.env.example
Dockerfile
requirements.txt
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动

```bash
uvicorn medical_ad_ocr_tools.main:app --host 0.0.0.0 --port 8000
```

## 配置

- 默认配置文件是 [config/config.yaml](/d:/Workspace/LearnAI/yb-PaddleOCR/config/config.yaml)
- 规则配置文件是 [config/rules.yaml](/d:/Workspace/LearnAI/yb-PaddleOCR/config/rules.yaml)
- OSS 配置通过环境变量覆盖
- 参考 [.env.example](/d:/Workspace/LearnAI/yb-PaddleOCR/.env.example)
- OCR 支持优先使用项目内本地模型目录，避免部署时在线下载

核心环境变量：

- `MEDICAL_AD_CONFIG_PATH`
- `MEDICAL_AD_OSS_ENABLED`
- `MEDICAL_AD_OSS_ENDPOINT`
- `MEDICAL_AD_OSS_BUCKET_NAME`
- `MEDICAL_AD_OSS_ACCESS_KEY_ID`
- `MEDICAL_AD_OSS_ACCESS_KEY_SECRET`
- `MEDICAL_AD_OSS_PREFIX`
- `MEDICAL_AD_OSS_PUBLIC_BASE_URL`
- `MEDICAL_AD_TEXT_DETECTION_MODEL_DIR`
- `MEDICAL_AD_TEXT_RECOGNITION_MODEL_DIR`
- `MEDICAL_AD_TEXTLINE_ORIENTATION_MODEL_DIR`

## 本地模型

推荐把模型随部署包一起带上，目录示例：

```text
models/
  ocr/
    PP-OCRv5_mobile_det/
    PP-OCRv5_mobile_rec/
```

当 [config/config.yaml](/d:/Workspace/LearnAI/yb-PaddleOCR/config/config.yaml) 或环境变量中配置了这些目录后，服务会优先读取本地模型，不再依赖运行时下载。

如果你已经在本机下载过模型，也可以直接把缓存目录拷贝到项目内：

```text
C:\Users\<user>\.paddlex\official_models\PP-OCRv5_mobile_det
C:\Users\<user>\.paddlex\official_models\PP-OCRv5_mobile_rec
```

## 接口

### `GET /health`

返回服务存活状态。

### `POST /tools/medical-ad/ocr/analyze`

当前只支持 `application/json`。

```json
{
  "request_id": "req-002",
  "image_url": "https://example.com/demo.jpg"
}
```

返回字段包含：

- `ocr_text`
- `ocr_blocks`
- `phones`
- `wechat_ids`
- `qqs`
- `hit_keywords`
- `hit_rules`
- `risk_score`
- `risk_level`
- `suspicious`
- `annotated_image_url`
- `annotated_image_oss_key`
- `ads`

### `POST /tools/medical-ad/rule/evaluate`

用于只做规则识别，不跑 OCR。

```json
{
  "request_id": "rule-001",
  "ocr_blocks": [
    {
      "text": "医保卡高价回收 微信 abc123",
      "points": [[10, 10], [240, 10], [240, 50], [10, 50]],
      "confidence": 0.99
    }
  ]
}
```

## Docker

```bash
docker build -t medical-ad-ocr-tools .
docker run --rm -p 8000:8000 --env-file .env medical-ad-ocr-tools
```

## 说明

- 当前实现不接入 MySQL、Redis、Celery，也不集成千问视觉模型。
- 服务保留多个小广告区域识别输出，整体结果和每个广告区域结果都会返回。
- `PaddleOCR 3.4.0` 在服务启动时会预热并下载缺失模型，首次启动会更慢，但可以避免首个请求承担模型初始化成本。
