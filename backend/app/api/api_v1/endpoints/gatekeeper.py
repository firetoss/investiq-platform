"""
InvestIQ Platform - 四闸门校验API端点
投资决策四闸门校验服务的REST API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.services.gatekeeper import gatekeeper, entry_plan_generator

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class GateCheckRequest(BaseModel):
    """四闸门校验请求模型"""
    industry_score: float = Field(..., ge=0, le=100, description="行业评分")
    equity_score: float = Field(..., ge=0, le=100, description="个股评分")
    valuation_percentile: Optional[float] = Field(None, ge=0, le=1, description="估值分位 (0-1)")
    above_200dma: bool = Field(..., description="是否在200日均线上方")
    peg: Optional[float] = Field(None, ge=0, description="PEG比率")
    is_growth_stock: bool = Field(False, description="是否为成长股")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="额外上下文")


class EntryPlanRequest(BaseModel):
    """建仓计划请求模型"""
    ticker: str = Field(..., description="股票代码")
    target_position: float = Field(..., gt=0, description="目标仓位金额")
    current_price: float = Field(..., gt=0, description="当前价格")
    technical_data: Optional[Dict[str, Any]] = Field(None, description="技术分析数据")
    risk_tolerance: str = Field("medium", description="风险承受度 (low/medium/high)")


class BatchGateCheckRequest(BaseModel):
    """批量四闸门校验请求模型"""
    requests: List[GateCheckRequest] = Field(..., description="校验请求列表")


class GateCheckResponse(BaseModel):
    """四闸门校验响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="校验结果")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.post(
    "/check",
    response_model=GateCheckResponse,
    summary="四闸门校验",
    description="执行完整的四闸门投资决策校验"
)
async def check_four_gates(
    request: GateCheckRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """四闸门校验"""
    try:
        # 调用四闸门校验器
        result = await gatekeeper.check_all_gates(
            industry_score=request.industry_score,
            equity_score=request.equity_score,
            valuation_percentile=request.valuation_percentile,
            above_200dma=request.above_200dma,
            peg=request.peg,
            is_growth_stock=request.is_growth_stock,
            additional_context=request.additional_context
        )
        
        return GateCheckResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Gate check API error: {e}")
        return GateCheckResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/batch-check",
    response_model=GateCheckResponse,
    summary="批量四闸门校验",
    description="批量执行四闸门校验"
)
async def batch_check_gates(
    request: BatchGateCheckRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """批量四闸门校验"""
    try:
        # 转换请求格式
        check_requests = []
        for i, req in enumerate(request.requests):
            check_req = req.dict()
            check_req["request_id"] = f"{request_id}_{i}" if request_id else f"batch_{i}"
            check_requests.append(check_req)
        
        # 调用批量校验
        results = await gatekeeper.batch_check_gates(check_requests)
        
        return GateCheckResponse(
            success=True,
            data={
                "total_count": len(results),
                "passed_count": len([r for r in results if r.get("overall_pass", False)]),
                "results": results
            },
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch gate check API error: {e}")
        return GateCheckResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/entry-plan",
    response_model=GateCheckResponse,
    summary="生成建仓计划",
    description="基于四闸门校验结果生成三段式建仓计划"
)
async def generate_entry_plan(
    request: EntryPlanRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """生成建仓计划"""
    try:
        # 调用建仓计划生成器
        result = await entry_plan_generator.generate_entry_plan(
            ticker=request.ticker,
            target_position=request.target_position,
            current_price=request.current_price,
            technical_data=request.technical_data,
            risk_tolerance=request.risk_tolerance
        )
        
        return GateCheckResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Entry plan API error: {e}")
        return GateCheckResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/requirements",
    summary="获取四闸门要求",
    description="获取四闸门的详细要求和阈值"
)
async def get_gate_requirements():
    """获取四闸门要求"""
    try:
        requirements = gatekeeper.get_gate_requirements()
        
        return {
            "requirements": requirements,
            "current_thresholds": gatekeeper.gate_thresholds,
            "description": "四闸门投资决策校验要求",
            "methodology": "政策→行业→个股的四重验证机制"
        }
        
    except Exception as e:
        logger.error(f"Get requirements API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取闸门要求失败: {e}"
        )


@router.get(
    "/single/{gate_name}",
    summary="单个闸门校验",
    description="校验单个闸门"
)
async def check_single_gate(
    gate_name: str,
    industry_score: Optional[float] = None,
    equity_score: Optional[float] = None,
    valuation_percentile: Optional[float] = None,
    above_200dma: Optional[bool] = None,
    peg: Optional[float] = None,
    is_growth_stock: bool = False
):
    """单个闸门校验"""
    try:
        # 准备参数
        kwargs = {}
        if gate_name == "industry" and industry_score is not None:
            kwargs["industry_score"] = industry_score
        elif gate_name == "company" and equity_score is not None:
            kwargs["equity_score"] = equity_score
        elif gate_name == "valuation":
            kwargs["valuation_percentile"] = valuation_percentile
            kwargs["peg"] = peg
            kwargs["is_growth_stock"] = is_growth_stock
        elif gate_name == "execution" and above_200dma is not None:
            kwargs["above_200dma"] = above_200dma
        else:
            raise HTTPException(
                status_code=400,
                detail=f"闸门 {gate_name} 缺少必要参数"
            )
        
        # 调用单个闸门校验
        result = await gatekeeper.check_single_gate(gate_name, **kwargs)
        
        return {
            "gate_name": gate_name,
            "result": result,
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Single gate check API error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"单个闸门校验失败: {e}"
        )
