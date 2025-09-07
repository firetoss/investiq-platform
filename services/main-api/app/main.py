"""
InvestIQ Main API Service - 微服务架构的API网关
精简版本，只包含业务协调和API路由，AI推理交给独立服务
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# 临时使用本地实现，等共享库配置完成后再切换
import logging
from typing import Any, Dict
import structlog

# 临时日志设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 临时配置管理器
class ConfigManager:
    def get(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        import os
        env_key = f"INVESTIQ_{key.upper()}"
        value = os.getenv(env_key, default)
        if value is not None and cast_type != str:
            try:
                if cast_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                else:
                    value = cast_type(value)
            except (ValueError, TypeError):
                value = default
        return value
    
    def get_service_url(self, service_name: str) -> str:
        return self.get(f"{service_name.upper()}_SERVICE_URL", f"http://{service_name}-service:8000")

config = ConfigManager()

# 临时健康检查模型
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck(BaseModel):
    status: ServiceStatus
    version: str
    timestamp: datetime = None
    checks: Dict[str, bool] = {}
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        super().__init__(**data)

from .core.config import settings
from .api.v1.router import api_router
from .core.database import engine, init_db
from .core.redis import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting InvestIQ Main API Service")
    
    try:
        # 初始化数据库
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    try:
        # 初始化Redis
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    # 临时跳过微服务客户端注册，稍后实现
    logger.info("Main API Service started (microservice clients will be added later)")
    
    yield
    
    # 清理资源
    logger.info("Shutting down InvestIQ Main API Service")
    try:
        await redis_client.close()
    except Exception as e:
        logger.error(f"Redis cleanup failed: {e}")


# 创建FastAPI应用
app = FastAPI(
    title="InvestIQ Main API",
    description="智能投资决策平台 - 主API服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# 配置日志
setup_logging()

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 包含API路由
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查数据库
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        db_healthy = True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
    
    # 检查Redis
    try:
        await redis_client.ping()
        redis_healthy = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_healthy = False
    
    # 检查微服务
    service_health = await registry.health_check_all()
    
    # 确定整体状态
    all_checks = {
        "database": db_healthy,
        "redis": redis_healthy,
        **{f"service_{name}": health.status == "healthy" 
           for name, health in service_health.items()}
    }
    
    overall_healthy = all(all_checks.values())
    status = ServiceStatus.HEALTHY if overall_healthy else ServiceStatus.DEGRADED
    
    return HealthCheck(
        status=status,
        version="1.0.0",
        checks=all_checks,
        metadata={
            "service_count": len(service_health),
            "healthy_services": sum(1 for h in service_health.values() if h.status == "healthy")
        }
    )


@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development"
    )