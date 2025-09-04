"""
InvestIQ Platform - 评分API端点
行业和个股评分服务的REST API
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.services.scoring_engine import scoring_engine
from backend.app.api import API_DESCRIPTIONS

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class IndustryScoringRequest(BaseModel):
    """行业评分请求模型"""
    industry_id: str = Field(..., description="行业ID")
    score_p: float = Field(..., ge=0, le=100, description="政策强度与确定性 (P)")
    score_e: float = Field(..., ge=0, le=100, description="落地证据 (E)")
    score_m: float = Field(..., ge=0, le=100, description="市场确认 (M)")
    score_r_neg: float = Field(..., ge=0, le=100, description="风险因子 (R-)")
    
    weights: Optional[Dict[str, float]] = Field(None, description="自定义权重")
    evidence_data: Optional[List[Dict[str, Any]]] = Field(None, description="证据数据")
    as_of: Optional[date] = Field(None, description="业务日期")
    
    @validator("weights")
    def validate_weights(cls, v):
        if v is not None:
            required_keys = {"P", "E", "M", "R"}
            if not all(key in v for key in required_keys):
                raise ValueError("权重必须包含 P, E, M, R 四个维度")
            if abs(sum(v.values()) - 1.0) > 0.01:
                raise ValueError("权重总和必须等于1.0")
        return v


class EquityScoringRequest(BaseModel):
    """个股评分请求模型"""
    equity_id: str = Field(..., description="个股ID")
    ticker: str = Field(..., description="股票代码")
    score_q: float = Field(..., ge=0, le=100, description="质量 (Q)")
    score_v: float = Field(..., ge=0, le=100, description="估值 (V)")
    score_m: float = Field(..., ge=0, le=100, description="动量 (M)")
    score_c: float = Field(..., ge=0, le=100, description="政策契合 (C)")
    score_s: float = Field(..., ge=0, le=100, description="份额/护城河 (S)")
    score_r_neg: float = Field(0, ge=0, le=100, description="红旗 (R-)")
    
    weights: Optional[Dict[str, float]] = Field(None, description="自定义权重")
    evidence_data: Optional[List[Dict[str, Any]]] = Field(None, description="证据数据")
    as_of: Optional[date] = Field(None, description="业务日期")
    
    @validator("weights")
    def validate_weights(cls, v):
        if v is not None:
            required_keys = {"Q", "V", "M", "C", "S"}
            if not all(key in v for key in required_keys):
                raise ValueError("权重必须包含 Q, V, M, C, S 五个维度")
            if abs(sum(v.values()) - 1.0) > 0.01:
                raise ValueError("权重总和必须等于1.0")
        return v


class BatchScoringRequest(BaseModel):
    """批量评分请求模型"""
    scoring_type: str = Field(..., description="评分类型 (industry/equity)")
    requests: List[Dict[str, Any]] = Field(..., description="评分请求列表")


class ScoringResponse(BaseModel):
    """评分响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="评分结果")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.post(
    "/industry",
    response_model=ScoringResponse,
    summary="计算行业评分",
    description="基于四维度评分模型计算行业总评分"
)
async def calculate_industry_score(
    request: IndustryScoringRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """计算行业评分"""
    try:
        # 调用评分引擎
        result = await scoring_engine.calculate_industry_score(
            industry_id=request.industry_id,
            score_p=request.score_p,
            score_e=request.score_e,
            score_m=request.score_m,
            score_r_neg=request.score_r_neg,
            weights=request.weights,
            evidence_data=request.evidence_data,
            as_of=request.as_of
        )
        
        return ScoringResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Industry scoring API error: {e}")
        return ScoringResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/equity",
    response_model=ScoringResponse,
    summary="计算个股评分",
    description="基于五维度评分模型计算个股总评分"
)
async def calculate_equity_score(
    request: EquityScoringRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """计算个股评分"""
    try:
        # 调用评分引擎
        result = await scoring_engine.calculate_equity_score(
            equity_id=request.equity_id,
            ticker=request.ticker,
            score_q=request.score_q,
            score_v=request.score_v,
            score_m=request.score_m,
            score_c=request.score_c,
            score_s=request.score_s,
            score_r_neg=request.score_r_neg,
            weights=request.weights,
            evidence_data=request.evidence_data,
            as_of=request.as_of
        )
        
        return ScoringResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Equity scoring API error: {e}")
        return ScoringResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/batch",
    response_model=ScoringResponse,
    summary="批量评分计算",
    description="批量计算行业或个股评分，支持GPU加速"
)
async def batch_calculate_scores(
    request: BatchScoringRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """批量评分计算"""
    try:
        if request.scoring_type == "industry":
            # 批量行业评分
            results = await scoring_engine.batch_calculate_industry_scores(request.requests)
        elif request.scoring_type == "equity":
            # 批量个股评分
            results = await scoring_engine.batch_calculate_equity_scores(request.requests)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的评分类型: {request.scoring_type}"
            )
        
        return ScoringResponse(
            success=True,
            data={
                "scoring_type": request.scoring_type,
                "total_count": len(results),
                "results": results
            },
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch scoring API error: {e}")
        return ScoringResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/weights",
    summary="获取评分权重配置",
    description="获取当前的行业和个股评分权重配置"
)
async def get_scoring_weights():
    """获取评分权重配置"""
    return {
        "industry_weights": settings.industry_score_weights,
        "equity_weights": settings.equity_score_weights,
        "thresholds": {
            "industry_threshold": settings.GATE_INDUSTRY_THRESHOLD,
            "equity_threshold": settings.GATE_EQUITY_THRESHOLD,
            "core_candidate_threshold": 75.0
        },
        "description": {
            "industry": "行业评分权重: P(政策) + E(证据) + M(市场) + R(风险)",
            "equity": "个股评分权重: Q(质量) + V(估值) + M(动量) + C(契合) + S(份额)"
        }
    }


@router.get(
    "/formulas",
    summary="获取评分公式说明",
    description="获取行业和个股评分的计算公式"
)
async def get_scoring_formulas():
    """获取评分公式说明"""
    return {
        "industry_formula": {
            "formula": "Score^{Ind} = 0.35P + 0.25E + 0.25M + 0.15×(100-R^{-})",
            "components": {
                "P": "政策强度与确定性 (主体级别、预算/税惠/标准化)",
                "E": "落地证据 (订单/招标/设备进场/验收/良率/导入节奏)",
                "M": "市场确认 (中期动量、>200DMA占比、景气扩散)",
                "R-": "风险 (估值拥挤、外部限制、补贴依赖、产能过剩)"
            },
            "thresholds": {
                "in_pool": "≥70分",
                "core_candidate": "≥75分"
            }
        },
        "equity_formula": {
            "formula": "Score^{Eq} = 0.30Q + 0.20V + 0.25M + 0.15C + 0.10S - R^{-}",
            "components": {
                "Q": "质量 (ROE中位、现金流质量、毛利趋势、杠杆)",
                "V": "估值 (历史分位与同业对比、PEG、EV/Sales)",
                "M": "动量/预期 (中期趋势、EPS一致预期修正)",
                "C": "政策契合 (资质/订单/产能与政策条款映射)",
                "S": "份额/护城河 (市占与定价权、客户锁定)",
                "R-": "红旗 (应收+存货共振、非标审计、质押高、重大诉讼)"
            },
            "thresholds": {
                "observe": "≥65分且无红旗",
                "build": "≥70分且无红旗"
            }
        }
    }


@router.get(
    "/health",
    summary="评分引擎健康检查",
    description="检查评分引擎和GPU加速状态"
)
async def scoring_health_check():
    """评分引擎健康检查"""
    try:
        # 检查GPU状态
        gpu_status = {
            "available": scoring_engine.gpu_available,
            "enabled": settings.ENABLE_GPU_ACCELERATION
        }
        
        if scoring_engine.gpu_available:
            from backend.app.utils.gpu import get_gpu_info
            gpu_status.update(get_gpu_info())
        
        # 执行简单的评分测试
        test_result = await scoring_engine.calculate_industry_score(
            industry_id="test",
            score_p=80.0,
            score_e=75.0,
            score_m=70.0,
            score_r_neg=20.0
        )
        
        return {
            "status": "healthy",
            "gpu_status": gpu_status,
            "test_calculation": {
                "input": {"P": 80, "E": 75, "M": 70, "R-": 20},
                "output": test_result["total_score"],
                "expected": 74.0,  # 0.35*80 + 0.25*75 + 0.25*70 + 0.15*80
                "test_passed": abs(test_result["total_score"] - 74.0) < 0.1
            },
            "weights": {
                "industry": settings.industry_score_weights,
                "equity": settings.equity_score_weights
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scoring health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"评分引擎健康检查失败: {e}"
        )


@router.get(
    "/benchmark",
    summary="评分引擎性能基准测试",
    description="执行评分引擎性能基准测试"
)
async def scoring_benchmark(
    batch_size: int = Query(100, ge=1, le=1000, description="批量大小")
):
    """评分引擎性能基准测试"""
    try:
        import time
        import random
        
        # 生成测试数据
        test_industry_data = []
        test_equity_data = []
        
        for i in range(batch_size):
            # 行业测试数据
            test_industry_data.append({
                "industry_id": f"test_industry_{i}",
                "score_p": random.uniform(60, 90),
                "score_e": random.uniform(60, 90),
                "score_m": random.uniform(60, 90),
                "score_r_neg": random.uniform(10, 30)
            })
            
            # 个股测试数据
            test_equity_data.append({
                "equity_id": f"test_equity_{i}",
                "ticker": f"TEST{i:04d}",
                "score_q": random.uniform(60, 90),
                "score_v": random.uniform(60, 90),
                "score_m": random.uniform(60, 90),
                "score_c": random.uniform(60, 90),
                "score_s": random.uniform(60, 90),
                "score_r_neg": random.uniform(0, 20)
            })
        
        # 执行基准测试
        start_time = time.time()
        industry_results = await scoring_engine.batch_calculate_industry_scores(test_industry_data)
        industry_time = time.time() - start_time
        
        start_time = time.time()
        equity_results = await scoring_engine.batch_calculate_equity_scores(test_equity_data)
        equity_time = time.time() - start_time
        
        return {
            "benchmark_results": {
                "batch_size": batch_size,
                "industry_scoring": {
                    "duration_ms": round(industry_time * 1000, 2),
                    "throughput_per_second": round(batch_size / industry_time, 2),
                    "avg_time_per_item_ms": round(industry_time * 1000 / batch_size, 2)
                },
                "equity_scoring": {
                    "duration_ms": round(equity_time * 1000, 2),
                    "throughput_per_second": round(batch_size / equity_time, 2),
                    "avg_time_per_item_ms": round(equity_time * 1000 / batch_size, 2)
                },
                "gpu_acceleration": {
                    "available": scoring_engine.gpu_available,
                    "enabled": settings.ENABLE_GPU_ACCELERATION
                }
            },
            "tested_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scoring benchmark failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"基准测试失败: {e}"
        )
