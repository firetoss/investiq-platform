"""
InvestIQ Platform - API端点包
"""

# 导入所有端点模块以确保路由注册
from . import scoring, gatekeeper, liquidity

# 创建占位符模块 (后续开发)
class PlaceholderRouter:
    """占位符路由器"""
    def __init__(self, name: str):
        from fastapi import APIRouter
        self.router = APIRouter()
        self.name = name
        
        @self.router.get("/")
        async def placeholder_endpoint():
            return {
                "message": f"{name} API endpoints coming soon",
                "status": "under_development"
            }

# 创建占位符端点
portfolio = PlaceholderRouter("Portfolio")
alerts = PlaceholderRouter("Alerts") 
evidence = PlaceholderRouter("Evidence")
system = PlaceholderRouter("System")

__all__ = [
    "scoring",
    "gatekeeper", 
    "liquidity",
    "portfolio",
    "alerts",
    "evidence", 
    "system"
]
