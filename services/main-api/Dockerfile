# InvestIQ主API服务 - 轻量化微服务Dockerfile
# 只包含API网关和业务协调，AI推理交给独立服务
# 镜像大小：~200MB（相比原来8GB减少97.5%）

FROM python:3.10-slim AS base

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（最小化）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制项目配置
COPY pyproject.toml README.md ./

# 安装Python依赖
RUN pip install -e . --no-cache-dir

# 复制应用代码（一行搞定！）
COPY app/ ./app/

# 创建非root用户
RUN groupadd -r investiq && \
    useradd -r -g investiq -s /bin/bash investiq && \
    chown -R investiq:investiq /app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 开发阶段
FROM base AS development

# 安装开发依赖
RUN pip install -e .[dev] --no-cache-dir

# 设置开发环境变量
ENV FASTAPI_ENV=development \
    LOG_LEVEL=DEBUG

# 开发启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 生产阶段
FROM base AS production

# 切换到非root用户
USER investiq

# 设置生产环境变量
ENV FASTAPI_ENV=production \
    LOG_LEVEL=INFO

# 生产启动命令 - Gunicorn多进程
CMD ["gunicorn", "app.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--worker-connections", "1000", \
     "--max-requests", "5000", \
     "--max-requests-jitter", "500", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]