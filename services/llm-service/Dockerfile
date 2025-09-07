# InvestIQ LLM推理服务 Dockerfile
# 基于llama.cpp，支持Jetson AGX Orin GPU加速
# 镜像大小：~4GB（包含模型文件）

FROM ghcr.io/ggerganov/llama.cpp:server-cuda AS base

# 设置工作目录
WORKDIR /app

# 安装Python和依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Python包
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    structlog

# 创建模型目录
RUN mkdir -p /models

# 复制Python服务代码
COPY app/ ./app/

# 复制启动脚本
COPY start.sh ./
RUN chmod +x start.sh

# 暴露端口
EXPOSE 8001 8080

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 环境变量
ENV MODEL_PATH="/models/qwen3-8b-instruct.gguf" \
    LLAMA_CPP_PORT="8080" \
    PYTHON_SERVICE_PORT="8001"

# 开发阶段
FROM base AS development
ENV LOG_LEVEL="DEBUG"
CMD ["./start.sh", "dev"]

# 生产阶段
FROM base AS production

# 创建非root用户
RUN groupadd -r llm && useradd -r -g llm llm
RUN chown -R llm:llm /app /models

USER llm

ENV LOG_LEVEL="INFO"
CMD ["./start.sh", "prod"]