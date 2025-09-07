"""
API v1路由入口 - 临时简化版本
"""

from fastapi import APIRouter


# 创建v1路由器
api_router = APIRouter()


@api_router.get("/")
async def api_info():
    """API信息"""
    return {
        "name": "InvestIQ Main API",
        "version": "1.0.0",
        "status": "operational",
        "message": "Microservices architecture - basic endpoints active",
        "endpoints": [
            "/ - API信息",
            "/health - 健康检查"
        ]
    }


@api_router.get("/test")
async def test_endpoint():
    """测试端点"""
    return {
        "message": "Main API is working!",
        "service": "main-api",
        "status": "ok"
    }