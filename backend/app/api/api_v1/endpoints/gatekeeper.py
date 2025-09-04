"""
InvestIQ Platform - 四闸门校验API端点
投资决策四闸门校验服务的REST API
符合OpenAPI v1.1.1-final规范
"""

from datetime import datetime, date
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query, Response
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.core.exceptions import ValidationException, IdempotencyException
from backend.app.services.gatekeeper import gatekeeper, entry_plan_generator
from backend.app.services.idempotency_service import IdempotencyService

logger = get_logger(__name__)

router = APIRouter()

# 创建幂等性服务实例
idempotency_service = IdempotencyService()


# 错误响应模型
class ErrorResponse(BaseModel):
    """标准错误响应"""
    error: Dict[str, Any] = Field(..., description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误时间")


class ValidationErrorDetail(BaseModel):
    """验证错误详情"""
    field: str = Field(..., description="字段名")
    message: str = Field(..., description="错误信息")


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

    @validator('peg')
    def validate_peg_for_growth_stock(cls, v, values):
        """成长股必须提供PEG"""
        if values.get('is_growth_stock', False) and v is None:
            raise ValueError('成长股必须提供PEG比率')
        return v


class EntryPlanRequest(BaseModel):
    """建仓计划请求模型"""
    ticker: str = Field(..., description="股票代码")
    target_position: float = Field(..., gt=0, description="目标仓位金额")
    current_price: float = Field(..., gt=0, description="当前价格")
    technical_data: Optional[Dict[str, Any]] = Field(None, description="技术分析数据")
    risk_tolerance: str = Field("medium", description="风险承受度 (low/medium/high)")

    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        """验证风险承受度"""
        if v not in ['low', 'medium', 'high']:
            raise ValueError('风险承受度必须为 low, medium 或 high')
        return v


class BatchGateCheckRequest(BaseModel):
    """批量四闸门校验请求模型"""
    requests: List[GateCheckRequest] = Field(..., min_items=1, max_items=100, description="校验请求列表")


class SingleGateResult(BaseModel):
    """单个闸门结果"""
    gate_name: str = Field(..., description="闸门名称")
    pass_: bool = Field(..., alias="pass", description="是否通过")
    score: Optional[float] = Field(None, description="相关评分")
    threshold: Optional[float] = Field(None, description="门槛值")
    margin: Optional[float] = Field(None, description="超出门槛的幅度")
    message: str = Field(..., description="结果说明")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")


class GateCheckResult(BaseModel):
    """四闸门校验结果"""
    overall_pass: bool = Field(..., description="总体是否通过")
    failed_gates: List[str] = Field(..., description="未通过的闸门")
    gates: Dict[str, SingleGateResult] = Field(..., description="各闸门详细结果")
    summary: Dict[str, Any] = Field(..., description="校验摘要")
    checked_at: datetime = Field(..., description="校验时间")
    thresholds_used: Dict[str, float] = Field(..., description="使用的阈值")


# API响应包装器
async def create_success_response(
    data: Any,
    request_id: Optional[str] = None,
    snapshot_ts: Optional[datetime] = None
) -> Dict[str, Any]:
    """创建成功响应"""
    return {
        "data": data,
        "request_id": request_id,
        "snapshot_ts": (snapshot_ts or datetime.utcnow()).isoformat()
    }


async def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    status_code: int = 500
) -> HTTPException:
    """创建错误响应"""
    error_detail = {
        "error": {
            "code": error_code,
            "message": message,
            "details": details or {},
            "request_id": request_id
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    return HTTPException(status_code=status_code, detail=error_detail)


# 辅助函数
async def process_with_idempotency(
    idempotency_key: Optional[str],
    operation_func,
    request_data: Dict[str, Any],
    request_id: Optional[str] = None
) -> Any:
    """处理带幂等性的操作"""
    if not idempotency_key:
        # 没有幂等键，直接执行
        return await operation_func()
    
    try:
        # 检查幂等键
        existing_result = await idempotency_service.get_result(idempotency_key)
        if existing_result:
            logger.info(f"Returning cached result for idempotency key: {idempotency_key}")
            return existing_result
        
        # 执行操作
        result = await operation_func()
        
        # 缓存结果
        await idempotency_service.store_result(idempotency_key, result, request_data)
        
        return result
        
    except IdempotencyException as e:
        raise await create_error_response(
            error_code="IDEMPOTENCY_CONFLICT",
            message="幂等键冲突，请检查请求内容",
            details={"idempotency_key": idempotency_key},
            request_id=request_id,
            status_code=409
        )


def add_response_headers(response: Response, request_id: Optional[str], snapshot_ts: datetime):
    """添加响应头"""
    if request_id:
        response.headers["X-Request-ID"] = request_id
    response.headers["X-Snapshot-Ts"] = snapshot_ts.isoformat()
    response.headers["Content-Type"] = "application/json; charset=utf-8"


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
