"""
InvestIQ Platform - 投资组合API端点
投资组合构建、管理和风险控制服务的REST API
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.core.config import settings
from backend.app.models.portfolio import (
    Portfolio, Position, PositionTier, CircuitBreakerLevel,
    PortfolioSnapshot, RebalanceRecord, LiquidityCheck, RiskMetrics
)
from backend.app.services.portfolio_service import portfolio_service
from backend.app.api import API_DESCRIPTIONS

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class EquityCandidate(BaseModel):
    """候选个股模型"""
    ticker: str = Field(..., description="股票代码")
    equity_id: str = Field(..., description="个股ID")
    tier: str = Field(..., description="分层 (A/B/C)")
    score: float = Field(..., ge=0, le=100, description="个股评分")
    target_weight: Optional[float] = Field(None, ge=0, le=1, description="目标权重")
    current_price: Optional[float] = Field(None, description="当前价格")
    market_cap: Optional[float] = Field(None, description="市值")
    industry: Optional[str] = Field(None, description="所属行业")

    @validator("tier")
    def validate_tier(cls, v):
        if v not in ["A", "B", "C"]:
            raise ValueError("分层必须为 A、B 或 C")
        return v


class PortfolioConstructRequest(BaseModel):
    """投资组合构建请求模型"""
    name: str = Field(..., description="组合名称")
    description: Optional[str] = Field(None, description="组合描述")
    total_capital: float = Field(..., gt=0, description="总资金")
    leverage_max: float = Field(1.1, ge=1.0, le=2.0, description="最大杠杆比例")
    
    # 分层配置
    tier_config: Optional[Dict[str, List[float]]] = Field(None, description="分层配置")
    
    # 候选股票
    candidates: List[EquityCandidate] = Field(..., description="候选股票列表")
    
    # 构建参数
    cash_buffer: float = Field(0.05, ge=0, le=0.3, description="现金缓冲比例")
    max_single_position: float = Field(0.15, ge=0, le=0.5, description="单票最大权重")
    max_sector_concentration: float = Field(0.4, ge=0, le=1.0, description="单行业最大权重")
    
    # 建仓方式
    entry_mode: str = Field("three_stage", description="建仓方式 (three_stage/immediate)")
    
    # 时间参数
    as_of: Optional[date] = Field(None, description="业务日期")

    @validator("tier_config")
    def validate_tier_config(cls, v):
        if v is not None:
            required_tiers = {"A", "B", "C"}
            if not all(tier in v for tier in required_tiers):
                raise ValueError("分层配置必须包含 A、B、C 三层")
            for tier, limits in v.items():
                if len(limits) != 2 or limits[0] >= limits[1]:
                    raise ValueError(f"分层 {tier} 的配置格式错误，应为 [min, max]")
        return v


class RebalanceRequest(BaseModel):
    """组合再平衡请求模型"""
    portfolio_id: str = Field(..., description="组合ID")
    rebalance_type: str = Field("manual", description="再平衡类型")
    trigger_reason: Optional[str] = Field(None, description="触发原因")
    target_allocation: Optional[Dict[str, float]] = Field(None, description="目标配置")
    max_turnover: float = Field(0.2, ge=0, le=1.0, description="最大换手率")
    as_of: Optional[date] = Field(None, description="业务日期")


class CircuitBreakerRequest(BaseModel):
    """断路器请求模型"""
    portfolio_id: str = Field(..., description="组合ID")
    current_nav: float = Field(..., description="当前净值")
    trigger_reason: str = Field(..., description="触发原因")
    force_trigger: bool = Field(False, description="强制触发")


class PortfolioResponse(BaseModel):
    """投资组合响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.post(
    "/construct",
    response_model=PortfolioResponse,
    summary="构建投资组合",
    description="基于四闸门校验结果和风险约束构建投资组合"
)
async def construct_portfolio(
    request: PortfolioConstructRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """构建投资组合"""
    try:
        # 调用组合构建服务
        result = await portfolio_service.construct_portfolio(
            name=request.name,
            description=request.description,
            total_capital=request.total_capital,
            leverage_max=request.leverage_max,
            tier_config=request.tier_config,
            candidates=request.candidates,
            cash_buffer=request.cash_buffer,
            max_single_position=request.max_single_position,
            max_sector_concentration=request.max_sector_concentration,
            entry_mode=request.entry_mode,
            as_of=request.as_of,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Portfolio construction failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/rebalance",
    response_model=PortfolioResponse,
    summary="投资组合再平衡",
    description="根据目标配置和市场变化进行组合再平衡"
)
async def rebalance_portfolio(
    request: RebalanceRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """投资组合再平衡"""
    try:
        result = await portfolio_service.rebalance_portfolio(
            portfolio_id=request.portfolio_id,
            rebalance_type=request.rebalance_type,
            trigger_reason=request.trigger_reason,
            target_allocation=request.target_allocation,
            max_turnover=request.max_turnover,
            as_of=request.as_of,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Portfolio rebalance failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/circuit-breaker",
    response_model=PortfolioResponse,
    summary="触发回撤断路器",
    description="根据回撤情况触发三级断路器机制"
)
async def trigger_circuit_breaker(
    request: CircuitBreakerRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """触发回撤断路器"""
    try:
        result = await portfolio_service.trigger_circuit_breaker(
            portfolio_id=request.portfolio_id,
            current_nav=request.current_nav,
            trigger_reason=request.trigger_reason,
            force_trigger=request.force_trigger,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Circuit breaker trigger failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/{portfolio_id}/status",
    response_model=PortfolioResponse,
    summary="获取组合状态",
    description="获取投资组合的详细状态信息"
)
async def get_portfolio_status(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    as_of: Optional[date] = Query(None, description="查询日期"),
    include_positions: bool = Query(True, description="是否包含持仓明细"),
    include_metrics: bool = Query(True, description="是否包含风险指标")
):
    """获取组合状态"""
    try:
        result = await portfolio_service.get_portfolio_status(
            portfolio_id=portfolio_id,
            as_of=as_of,
            include_positions=include_positions,
            include_metrics=include_metrics,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get portfolio status failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/{portfolio_id}/performance",
    response_model=PortfolioResponse,
    summary="获取组合绩效",
    description="获取投资组合的绩效指标和历史表现"
)
async def get_portfolio_performance(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    benchmark: Optional[str] = Query(None, description="基准代码")
):
    """获取组合绩效"""
    try:
        result = await portfolio_service.get_portfolio_performance(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            benchmark=benchmark,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get portfolio performance failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/{portfolio_id}/risk-analysis",
    response_model=PortfolioResponse,
    summary="风险分析",
    description="获取投资组合的详细风险分析报告"
)
async def get_risk_analysis(
    portfolio_id: str,
    db: AsyncSession = Depends(get_db),
    as_of: Optional[date] = Query(None, description="分析日期"),
    lookback_days: int = Query(252, description="回望天数"),
    confidence_level: float = Query(0.95, description="置信水平")
):
    """风险分析"""
    try:
        result = await portfolio_service.get_risk_analysis(
            portfolio_id=portfolio_id,
            as_of=as_of,
            lookback_days=lookback_days,
            confidence_level=confidence_level,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/list",
    response_model=PortfolioResponse,
    summary="获取组合列表",
    description="获取所有投资组合的列表"
)
async def list_portfolios(
    db: AsyncSession = Depends(get_db),
    active_only: bool = Query(True, description="仅显示活跃组合"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小")
):
    """获取组合列表"""
    try:
        result = await portfolio_service.list_portfolios(
            active_only=active_only,
            page=page,
            page_size=page_size,
            db=db
        )
        
        return PortfolioResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"List portfolios failed: {e}", exc_info=True)
        return PortfolioResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/config/default",
    summary="获取默认配置",
    description="获取投资组合的默认配置参数"
)
async def get_default_config():
    """获取默认配置"""
    return {
        "tier_config": {
            "A": {"min": 0.12, "max": 0.15, "description": "核心持仓"},
            "B": {"min": 0.08, "max": 0.10, "description": "重要持仓"},
            "C": {"min": 0.03, "max": 0.05, "description": "战术持仓"}
        },
        "risk_limits": {
            "max_leverage": 1.10,
            "max_single_position": 0.15,
            "max_sector_concentration": 0.40,
            "cash_buffer_min": 0.05,
            "cash_buffer_max": 0.30
        },
        "circuit_breaker": {
            "levels": [
                {
                    "level": "L1",
                    "drawdown": -0.10,
                    "actions": ["remove_leverage", "pause_tactical"],
                    "description": "去融资、暂停新开战术仓"
                },
                {
                    "level": "L2", 
                    "drawdown": -0.20,
                    "actions": ["halve_tactical", "clear_watch", "tighten_entry"],
                    "description": "战术仓减半、观察清零、入场门槛上调"
                },
                {
                    "level": "L3",
                    "drawdown": -0.30,
                    "actions": ["keep_top2_3", "cash_rest"],
                    "description": "仅保留2-3只确定性最高核心票，其余现金"
                }
            ]
        },
        "entry_modes": {
            "three_stage": {
                "name": "三段式建仓",
                "stages": [
                    {"stage": 1, "weight": 0.40, "condition": "immediate"},
                    {"stage": 2, "weight": 0.30, "condition": ">=200DMA"},
                    {"stage": 3, "weight": 0.30, "condition": "+/-1*ATR"}
                ]
            },
            "immediate": {
                "name": "立即建仓",
                "stages": [
                    {"stage": 1, "weight": 1.00, "condition": "immediate"}
                ]
            }
        },
        "rebalance_triggers": [
            "deviation_threshold_5pct",
            "monthly_schedule",
            "circuit_breaker_trigger",
            "manual_trigger"
        ]
    }


@router.get(
    "/health",
    summary="投资组合服务健康检查",
    description="检查投资组合构建服务状态"
)
async def portfolio_health_check(
    db: AsyncSession = Depends(get_db)
):
    """投资组合服务健康检查"""
    try:
        # 检查数据库连接
        await db.execute("SELECT 1")
        
        # 检查组合服务状态
        health_status = await portfolio_service.health_check(db)
        
        return {
            "status": "healthy",
            "service_status": health_status,
            "features": {
                "portfolio_construction": True,
                "rebalancing": True,
                "circuit_breaker": True,
                "risk_analysis": True,
                "performance_tracking": True
            },
            "config": {
                "default_leverage_max": settings.portfolio_leverage_max,
                "circuit_breaker_levels": 3,
                "supported_tiers": ["A", "B", "C"]
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Portfolio health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"投资组合服务健康检查失败: {e}"
        )