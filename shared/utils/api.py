"""
InvestIQ Platform - API工具模块
FastAPI应用创建和通用中间件
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import time
import logging
from typing import Optional, List, Dict, Any
import traceback

from ..config import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    """请求ID中间件，为每个请求分配唯一ID"""
    
    async def dispatch(self, request: Request, call_next):
        # 生成或获取请求ID
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 添加响应头
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""
    
    def __init__(self, app, logger: Optional[logging.Logger] = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # 记录请求信息
        self.logger.info(
            f"Request started - {request.method} {request.url.path} "
            f"[{request_id}] from {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # 记录响应信息
            self.logger.info(
                f"Request completed - {request.method} {request.url.path} "
                f"[{request_id}] {response.status_code} in {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # 记录错误信息
            self.logger.error(
                f"Request failed - {request.method} {request.url.path} "
                f"[{request_id}] {str(e)} in {process_time:.3f}s",
                exc_info=True
            )
            
            raise


def create_app(
    title: str,
    version: str = "1.0.0",
    description: Optional[str] = None,
    enable_cors: bool = True,
    enable_logging: bool = True,
    enable_request_id: bool = True,
    trusted_hosts: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title=title,
        version=version,
        description=description or f"{title} - InvestIQ Platform Microservice",
        docs_url=settings.app.docs_url if settings.app.debug else None,
        openapi_url=settings.app.openapi_url if settings.app.debug else None,
        debug=settings.app.debug
    )
    
    # 添加请求ID中间件
    if enable_request_id:
        app.add_middleware(RequestIDMiddleware)
    
    # 添加日志中间件
    if enable_logging:
        app.add_middleware(LoggingMiddleware, logger=logger)
    
    # 添加CORS中间件
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.app.cors_origins,
            allow_credentials=settings.app.cors_allow_credentials,
            allow_methods=settings.app.cors_allow_methods,
            allow_headers=settings.app.cors_allow_headers,
        )
    
    # 添加可信主机中间件
    if trusted_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
    
    # 添加异常处理器
    setup_exception_handlers(app)
    
    return app


def setup_exception_handlers(app: FastAPI):
    """设置异常处理器"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            },
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": 422,
                    "message": "Validation Error",
                    "details": exc.errors(),
                    "request_id": getattr(request.state, 'request_id', None)
                }
            },
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "request_id": getattr(request.state, 'request_id', None)
                }
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal Server Error",
                    "request_id": getattr(request.state, 'request_id', None),
                    "details": str(exc) if settings.app.debug else None
                }
            },
        )


def get_request_id(request: Request) -> str:
    """获取当前请求的ID"""
    return getattr(request.state, 'request_id', 'unknown')


def create_response(
    data: Any = None,
    message: str = "Success",
    code: int = 200,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建标准化响应格式"""
    response = {
        "code": code,
        "message": message,
        "timestamp": int(time.time() * 1000),
        "data": data
    }
    
    if request_id:
        response["request_id"] = request_id
    
    return response


def create_error_response(
    message: str,
    code: int = 400,
    details: Any = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """创建错误响应格式"""
    response = {
        "error": {
            "code": code,
            "message": message,
            "timestamp": int(time.time() * 1000)
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    if request_id:
        response["error"]["request_id"] = request_id
    
    return response


class HealthCheck:
    """健康检查工具类"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.checks = []
    
    def add_check(self, name: str, check_func):
        """添加健康检查项"""
        self.checks.append({
            "name": name,
            "check": check_func
        })
    
    async def run_checks(self) -> Dict[str, Any]:
        """运行所有健康检查"""
        results = {
            "service": self.service_name,
            "status": "healthy",
            "timestamp": int(time.time() * 1000),
            "checks": []
        }
        
        overall_healthy = True
        
        for check_item in self.checks:
            try:
                result = await check_item["check"]() if callable(check_item["check"]) else check_item["check"]()
                check_result = {
                    "name": check_item["name"],
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else None
                }
            except Exception as e:
                check_result = {
                    "name": check_item["name"],
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
            
            results["checks"].append(check_result)
            
            if check_result["status"] != "healthy":
                overall_healthy = False
        
        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results


def setup_health_check(app: FastAPI, service_name: str, additional_checks: Optional[List] = None):
    """设置健康检查端点"""
    health_checker = HealthCheck(service_name)
    
    # 添加基础检查
    health_checker.add_check("basic", lambda: True)
    
    # 添加额外检查
    if additional_checks:
        for check in additional_checks:
            health_checker.add_check(check["name"], check["check"])
    
    @app.get("/health")
    async def health_check(request: Request):
        """健康检查端点"""
        results = await health_checker.run_checks()
        status_code = 200 if results["status"] == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content=results
        )
    
    return health_checker