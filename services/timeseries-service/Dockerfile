# InvestIQ时序预测服务 Dockerfile
# 基于python:3.10-slim，CPU专用传统算法
# 镜像大小：~300MB（轻量化，无GPU依赖）

FROM python:3.10-slim AS base

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

# CPU优化环境变量
ENV CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=12 \
    MKL_NUM_THREADS=12 \
    NUMBA_NUM_THREADS=12 \
    NUMBA_DISABLE_CUDA=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 安装Python依赖
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    numpy==1.24.3 \
    pandas==2.1.1 \
    scipy==1.11.3 \
    scikit-learn==1.3.0 \
    statsmodels==0.14.0 \
    arch==6.2.0 \
    structlog==23.2.0 \
    investiq-common

# 复制应用代码
COPY app/ ./app/

# 创建非root用户
RUN groupadd -r timeseries && \
    useradd -r -g timeseries timeseries && \
    chown -R timeseries:timeseries /app

# 暴露端口
EXPOSE 8003

# 健康检查
HEALTHCHECK --interval=30s --timeout=15s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# 开发阶段
FROM base AS development

ENV LOG_LEVEL=DEBUG

CMD ["python3", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]

# 生产阶段
FROM base AS production

USER timeseries

ENV LOG_LEVEL=INFO

CMD ["python3", "-m", "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"]