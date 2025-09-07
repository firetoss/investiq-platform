"""
InvestIQ Platform - Scoring Service
行业与个股评分服务主入口
"""

import sys
import os
import logging
from contextlib import asynccontextmanager

# 添加项目根路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from shared.utils import create_app, setup_health_check, init_database, init_redis
from shared.config import settings, validate_settings

from .routers import industry, equity


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting Scoring Service...")
    
    # 验证配置
    if not validate_settings():
        logger.error("Configuration validation failed")
        raise RuntimeError("Invalid configuration")
    
    # 初始化数据库
    init_database(echo=settings.app.debug)
    logger.info("Database initialized")
    
    # 初始化Redis
    await init_redis()
    logger.info("Redis initialized")
    
    yield
    
    # 关闭时清理资源
    logger.info("Shutting down Scoring Service...")


# 创建FastAPI应用
app = create_app(
    title="InvestIQ Scoring Service",
    version="1.0.0",
    description="Industry and Equity Scoring Service for InvestIQ Platform",
    lifespan=lifespan
)

# 注册路由
app.include_router(
    industry.router,
    prefix=f"{settings.app.api_prefix}/industry",
    tags=["Industry Scoring"]
)

app.include_router(
    equity.router,
    prefix=f"{settings.app.api_prefix}/equity", 
    tags=["Equity Scoring"]
)

# 设置健康检查
health_checker = setup_health_check(app, "scoring-service")

# 根路径
@app.get("/")
async def root():
    """服务根路径"""
    return {
        "service": "InvestIQ Scoring Service",
        "version": "1.0.0",
        "status": "running",
        "description": "Industry and Equity Scoring Service"
    }


if __name__ == "__main__":
    import uvicorn
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO if not settings.app.debug else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 启动服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=settings.app.debug,
        log_level="info" if not settings.app.debug else "debug"
    )