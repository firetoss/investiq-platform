"""
InvestIQ Platform - 主应用入口
智能投资决策平台的FastAPI应用
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.api.api_v1.api import api_router
from backend.app.core.config import settings
from backend.app.core.database import engine, create_tables
from backend.app.core.logging import setup_logging
from backend.app.core.redis import redis_client
from backend.app.core.exceptions import (
    InvestIQException,
    ValidationException,
    DatabaseException,
    GPUException,
)
from backend.app.utils.gpu import check_gpu_availability
from backend.app.utils.health import HealthChecker


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # 生成请求ID
        request_id = request.headers.get("X-Request-ID", f"req_{int(start_time * 1000)}")
        
        # 记录请求开始
        logging.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None,
            }
        )
        
        try:
            response = await call_next(request)
            
            # 计算处理时间
            process_time = asyncio.get_event_loop().time() - start_time
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # 记录请求完成
            logging.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            return response
            
        except Exception as e:
            # 记录请求错误
            process_time = asyncio.get_event_loop().time() - start_time
            logging.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": process_time,
                },
                exc_info=True
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logging.info("Starting InvestIQ Platform...")
    
    # 设置日志
    setup_logging()
    
    # 检查GPU可用性
    gpu_available = check_gpu_availability()
    logging.info(f"GPU availability: {gpu_available}")
    
    # 创建数据库表
    await create_tables()
    logging.info("Database tables created/verified")
    
    # 初始化Redis连接
    await redis_client.ping()
    logging.info("Redis connection established")
    
    # 初始化健康检查器
    health_checker = HealthChecker()
    app.state.health_checker = health_checker
    
    logging.info("InvestIQ Platform started successfully")
    
    yield
    
    # 关闭时执行
    logging.info("Shutting down InvestIQ Platform...")
    
    # 关闭数据库连接
    await engine.dispose()
    
    # 关闭Redis连接
    await redis_client.close()
    
    logging.info("InvestIQ Platform shutdown complete")


# 创建FastAPI应用实例
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="智能投资决策平台 - 基于AI增强的行业选择与组合支持系统",
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.ENABLE_OPENAPI else None,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_REDOC else None,
    lifespan=lifespan,
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RequestLoggingMiddleware)


# 异常处理器
@app.exception_handler(InvestIQException)
async def investiq_exception_handler(request: Request, exc: InvestIQException):
    """InvestIQ自定义异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request.headers.get("X-Request-ID"),
        }
    )


@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """验证异常处理器"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "输入数据验证失败",
            "details": exc.details,
            "request_id": request.headers.get("X-Request-ID"),
        }
    )


@app.exception_handler(DatabaseException)
async def database_exception_handler(request: Request, exc: DatabaseException):
    """数据库异常处理器"""
    logging.error(f"Database error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "DATABASE_ERROR",
            "message": "数据库服务暂时不可用",
            "request_id": request.headers.get("X-Request-ID"),
        }
    )


@app.exception_handler(GPUException)
async def gpu_exception_handler(request: Request, exc: GPUException):
    """GPU异常处理器"""
    logging.error(f"GPU error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "error": "GPU_ERROR",
            "message": "GPU计算服务暂时不可用",
            "details": str(exc),
            "request_id": request.headers.get("X-Request-ID"),
        }
    )


# 基础路由
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Welcome to InvestIQ Platform",
        "version": settings.APP_VERSION,
        "docs": "/docs" if settings.ENABLE_DOCS else None,
    }


@app.get("/health")
async def health_check(request: Request):
    """健康检查端点"""
    health_checker = request.app.state.health_checker
    health_status = await health_checker.check_all()
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


@app.get("/metrics")
async def metrics():
    """Prometheus指标端点"""
    if not settings.PROMETHEUS_ENABLED:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics endpoint disabled"}
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/info")
async def app_info():
    """应用信息端点"""
    gpu_info = None
    try:
        import cupy as cp
        gpu_info = {
            "available": True,
            "device_count": cp.cuda.runtime.getDeviceCount(),
            "current_device": cp.cuda.Device().id,
            "memory_info": cp.cuda.Device().mem_info,
        }
    except Exception:
        gpu_info = {"available": False}
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "gpu_info": gpu_info,
        "features": {
            "gpu_acceleration": settings.ENABLE_GPU_ACCELERATION,
            "local_llm": settings.ENABLE_LOCAL_LLM,
            "prometheus": settings.PROMETHEUS_ENABLED,
        }
    }


# 包含API路由
app.include_router(api_router, prefix=settings.API_V1_STR)


# CLI入口点
def cli():
    """命令行入口点"""
    import typer
    
    cli_app = typer.Typer()
    
    @cli_app.command()
    def serve(
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
    ):
        """启动API服务器"""
        if workers > 1:
            # 生产模式使用gunicorn
            import subprocess
            cmd = [
                "gunicorn",
                "backend.app.main:app",
                f"--bind={host}:{port}",
                f"--workers={workers}",
                "--worker-class=uvicorn.workers.UvicornWorker",
                "--access-logfile=-",
                "--error-logfile=-",
            ]
            subprocess.run(cmd)
        else:
            # 开发模式使用uvicorn
            uvicorn.run(
                "backend.app.main:app",
                host=host,
                port=port,
                reload=reload,
                log_level=settings.LOG_LEVEL.lower(),
            )
    
    @cli_app.command()
    def init_db():
        """初始化数据库"""
        asyncio.run(create_tables())
        typer.echo("Database initialized successfully")
    
    @cli_app.command()
    def check_gpu():
        """检查GPU状态"""
        gpu_available = check_gpu_availability()
        if gpu_available:
            try:
                import cupy as cp
                device = cp.cuda.Device()
                mem_info = device.mem_info
                typer.echo(f"GPU Available: Device {device.id}")
                typer.echo(f"Memory: {mem_info[1] // 1024**3}GB total, {mem_info[0] // 1024**3}GB free")
            except Exception as e:
                typer.echo(f"GPU check failed: {e}")
        else:
            typer.echo("GPU not available")
    
    cli_app()


if __name__ == "__main__":
    # 直接运行时使用uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
