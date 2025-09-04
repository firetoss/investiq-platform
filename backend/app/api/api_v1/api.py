"""
InvestIQ Platform - API v1 主路由
汇总所有API端点
"""

from fastapi import APIRouter

from backend.app.api.api_v1.endpoints import (
    scoring,
    gatekeeper,
    liquidity,
    portfolio,
    alerts,
    evidence,
    system,
    ai,
    analysis,
    data,
    performance
)
from backend.app.api import API_TAGS

# 创建API路由器
api_router = APIRouter()

# 包含各个模块的路由
api_router.include_router(
    scoring.router,
    prefix="/scoring",
    tags=[API_TAGS["scoring"]]
)

api_router.include_router(
    gatekeeper.router,
    prefix="/gatekeeper", 
    tags=[API_TAGS["gatekeeper"]]
)

api_router.include_router(
    liquidity.router,
    prefix="/liquidity",
    tags=[API_TAGS["liquidity"]]
)

api_router.include_router(
    portfolio.router,
    prefix="/portfolio",
    tags=[API_TAGS["portfolio"]]
)

api_router.include_router(
    alerts.router,
    prefix="/alerts",
    tags=[API_TAGS["alerts"]]
)

api_router.include_router(
    evidence.router,
    prefix="/evidence",
    tags=[API_TAGS["evidence"]]
)

api_router.include_router(
    system.router,
    prefix="/system",
    tags=[API_TAGS["system"]]
)

api_router.include_router(
    ai.router,
    prefix="/ai",
    tags=[API_TAGS["ai"]]
)

api_router.include_router(
    analysis.router,
    prefix="/analysis",
    tags=[API_TAGS["analysis"]]
)

api_router.include_router(
    data.router,
    prefix="/data",
    tags=[API_TAGS["data"]]
)

api_router.include_router(
    performance.router,
    prefix="/performance",
    tags=[API_TAGS["performance"]]
)
