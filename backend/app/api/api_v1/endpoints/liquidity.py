"""
InvestIQ Platform - 流动性校验API端点
流动性和容量检查服务的REST API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.services.liquidity import (
    liquidity_checker,
    capacity_calculator,
    board_lot_validator,
    liquidity_optimizer
)

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class LiquidityCheckRequest(BaseModel):
    """流动性检查请求模型"""
    ticker: str = Field(..., description="股票代码")
    target_position: float = Field(..., gt=0, description="目标仓位金额")
    currency: str = Field(..., description="货币类型 (CNY/HKD)")
    adv_20: float = Field(..., gt=0, description="20日平均成交额")
    turnover: float = Field(..., ge=0, description="换手率")
    free_float_market_cap: float = Field(..., gt=0, description="自由流通市值")
    participation_rate: Optional[float] = Field(None, ge=0, le=1, description="参与率")
    exit_days: Optional[int] = Field(None, ge=1, description="退出天数")
    market_type: str = Field("A", description="市场类型 (A/H)")
    
    @validator("currency")
    def validate_currency(cls, v):
        if v not in ["CNY", "HKD", "USD"]:
            raise ValueError("货币类型必须是 CNY, HKD 或 USD")
        return v
    
    @validator("market_type")
    def validate_market_type(cls, v):
        if v not in ["A", "H", "US"]:
            raise ValueError("市场类型必须是 A, H 或 US")
        return v


class CapacityCalculationRequest(BaseModel):
    """容量计算请求模型"""
    ticker: str = Field(..., description="股票代码")
    adv_20: float = Field(..., gt=0, description="20日平均成交额")
    free_float_market_cap: float = Field(..., gt=0, description="自由流通市值")
    market_type: str = Field("A", description="市场类型 (A/H)")
    position_type: str = Field("core", description="持仓类型 (core/tactical)")


class BoardLotCheckRequest(BaseModel):
    """整手检查请求模型"""
    ticker: str = Field(..., description="股票代码")
    shares: int = Field(..., gt=0, description="股数")
    board_lot: int = Field(100, gt=0, description="每手股数")
    market_type: str = Field("A", description="市场类型 (A/H)")


class PositionOptimizationRequest(BaseModel):
    """仓位优化请求模型"""
    ticker: str = Field(..., description="股票代码")
    desired_position: float = Field(..., gt=0, description="期望仓位")
    market_data: Dict[str, Any] = Field(..., description="市场数据")
    constraints: Optional[Dict[str, Any]] = Field(None, description="约束条件")


class LiquidityResponse(BaseModel):
    """流动性响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="检查结果")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.post(
    "/check",
    response_model=LiquidityResponse,
    summary="流动性检查",
    description="执行流动性和容量检查，包含ADV要求、绝对底线和自由流通占用检查"
)
async def check_liquidity(
    request: LiquidityCheckRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """流动性检查"""
    try:
        # 调用流动性检查器
        result = await liquidity_checker.check_liquidity(
            ticker=request.ticker,
            target_position=request.target_position,
            currency=request.currency,
            adv_20=request.adv_20,
            turnover=request.turnover,
            free_float_market_cap=request.free_float_market_cap,
            participation_rate=request.participation_rate,
            exit_days=request.exit_days,
            market_type=request.market_type
        )
        
        return LiquidityResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Liquidity check API error: {e}")
        return LiquidityResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/capacity",
    response_model=LiquidityResponse,
    summary="容量计算",
    description="计算最大可建仓位"
)
async def calculate_capacity(
    request: CapacityCalculationRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """容量计算"""
    try:
        # 调用容量计算器
        result = await capacity_calculator.calculate_max_position(
            ticker=request.ticker,
            adv_20=request.adv_20,
            free_float_market_cap=request.free_float_market_cap,
            market_type=request.market_type,
            position_type=request.position_type
        )
        
        return LiquidityResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Capacity calculation API error: {e}")
        return LiquidityResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/board-lot-check",
    response_model=LiquidityResponse,
    summary="整手检查",
    description="检查股数是否符合整手交易要求，特别针对H股"
)
async def check_board_lot(
    request: BoardLotCheckRequest,
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """整手检查"""
    try:
        # 调用整手校验器
        result = board_lot_validator.validate_board_lot(
            ticker=request.ticker,
            shares=request.shares,
            board_lot=request.board_lot,
            market_type=request.market_type
        )
        
        return LiquidityResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Board lot check API error: {e}")
        return LiquidityResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/optimize",
    response_model=LiquidityResponse,
    summary="仓位优化",
    description="基于流动性约束优化仓位大小"
)
async def optimize_position(
    request: PositionOptimizationRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """仓位优化"""
    try:
        # 调用流动性优化器
        result = await liquidity_optimizer.optimize_position_size(
            ticker=request.ticker,
            desired_position=request.desired_position,
            market_data=request.market_data,
            constraints=request.constraints
        )
        
        return LiquidityResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Position optimization API error: {e}")
        return LiquidityResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/requirements/{market_type}",
    summary="获取流动性要求",
    description="获取指定市场的流动性要求说明"
)
async def get_liquidity_requirements(market_type: str = "A"):
    """获取流动性要求"""
    try:
        if market_type not in ["A", "H", "US"]:
            raise HTTPException(
                status_code=400,
                detail="市场类型必须是 A, H 或 US"
            )
        
        requirements = await liquidity_checker.get_liquidity_requirements(market_type)
        
        return {
            "requirements": requirements,
            "examples": {
                "A_stock": "A股: ADV≥3000万，换手≥0.5%，自由流通占用≤2%",
                "H_stock": "H股: ADV≥2000万港币，换手≥0.3%，自由流通占用≤2%"
            },
            "methodology": "基于参与率和退出天数的流动性约束模型"
        }
        
    except Exception as e:
        logger.error(f"Get liquidity requirements API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取流动性要求失败: {e}"
        )


@router.get(
    "/health",
    summary="流动性服务健康检查",
    description="检查流动性校验服务状态"
)
async def liquidity_health_check():
    """流动性服务健康检查"""
    try:
        # 执行简单的流动性检查测试
        test_result = await liquidity_checker.check_liquidity(
            ticker="TEST001",
            target_position=100000,  # 10万
            currency="CNY",
            adv_20=50000000,  # 5000万ADV
            turnover=0.01,    # 1%换手
            free_float_market_cap=10000000000,  # 100亿自由流通市值
            market_type="A"
        )
        
        return {
            "status": "healthy",
            "test_calculation": {
                "input": {
                    "target_position": 100000,
                    "adv_20": 50000000,
                    "turnover": 0.01,
                    "market_type": "A"
                },
                "output": {
                    "overall_pass": test_result["overall_pass"],
                    "adv_min_required": test_result["adv_min_required"],
                    "used_participation_rate": test_result["used_participation_rate"]
                },
                "test_passed": test_result["overall_pass"]
            },
            "configuration": liquidity_checker.liquidity_config,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Liquidity health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"流动性服务健康检查失败: {e}"
        )
