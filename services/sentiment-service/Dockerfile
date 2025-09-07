# InvestIQ情感分析服务 Dockerfile
# ONNX Runtime + TensorRT优化，支持GPU加速
# 镜像大小：~2GB（包含优化模型）

FROM nvcr.io/nvidia/pytorch:24.08-py3 AS base

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/transformers \
    ONNX_CACHE=/cache/onnx

# GPU优化环境变量
ENV NVIDIA_VISIBLE_DEVICES=all \
    CUDA_VISIBLE_DEVICES=0 \
    ORT_TENSORRT_FP16_ENABLE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p /cache/huggingface /cache/transformers /cache/onnx /models

# 安装Python依赖
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    onnxruntime-gpu \
    transformers \
    torch \
    numpy \
    structlog \
    investiq-common

# 复制应用代码
COPY app/ ./app/

# 复制模型转换脚本
COPY scripts/ ./scripts/
RUN chmod +x scripts/*.sh

# 预下载和转换模型
RUN python3 -c "\
import os; \
os.environ['TRANSFORMERS_OFFLINE'] = '0'; \
try: \
    from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    model_name = 'IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment'; \
    print(f'Downloading model: {model_name}'); \
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/cache/transformers'); \
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir='/cache/transformers'); \
    print('Model downloaded successfully') \
except Exception as e: \
    print(f'Model download failed: {e}'); \
    print('Will download at runtime') \
"

# 暴露端口
EXPOSE 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# 开发阶段
FROM base AS development

ENV LOG_LEVEL=DEBUG \
    BATCH_SIZE=16

CMD ["python3", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

# 生产阶段
FROM base AS production

# 创建非root用户
RUN groupadd -r sentiment && useradd -r -g sentiment sentiment
RUN chown -R sentiment:sentiment /app /cache /models

USER sentiment

ENV LOG_LEVEL=INFO \
    BATCH_SIZE=32

CMD ["python3", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]