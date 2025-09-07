"""
InvestIQ Platform - Scoring Service Schemas
API请求和响应的数据模型定义
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal


class EvidenceItem(BaseModel):
    """证据项"""
    type: str = Field(..., description="证据类型")
    url: Optional[str] = Field(None, description="证据链接")
    title: Optional[str] = Field(None, description="证据标题")
    source: Optional[str] = Field(None, description="数据源")
    note: Optional[str] = Field(None, description="备注")


class IndustryScoreRequest(BaseModel):
    """行业评分请求"""
    industry_id: str = Field(..., description="行业ID")
    industry_name: Optional[str] = Field(None, description="行业名称")
    P: float = Field(..., ge=0, le=100, description="政策强度评分 (0-100)")
    E: float = Field(..., ge=0, le=100, description="落地证据评分 (0-100)")
    M: float = Field(..., ge=0, le=100, description="市场确认评分 (0-100)")
    R_neg: float = Field(..., ge=0, le=100, description="风险扣分 (0-100)")
    evidences: Optional[List[EvidenceItem]] = Field(default=[], description="证据清单")
    created_by: Optional[str] = Field(None, description="创建者")
    
    @validator('P', 'E', 'M', 'R_neg')
    def validate_score_range(cls, v):
        if not (0 <= v <= 100):
            raise ValueError('Score must be between 0 and 100')
        return v


class IndustryScoreResponse(BaseModel):
    """行业评分响应"""
    industry_id: str
    industry_name: Optional[str]
    score: float = Field(..., description="计算得出的总分")
    P: float = Field(..., description="政策强度评分")
    E: float = Field(..., description="落地证据评分")
    M: float = Field(..., description="市场确认评分")
    R_neg: float = Field(..., description="风险扣分")
    is_qualified_for_pool: bool = Field(..., description="是否符合入池条件")
    is_core_candidate: bool = Field(..., description="是否为核心候选")
    score_breakdown: Dict[str, float] = Field(..., description="评分拆解")
    snapshot_id: str = Field(..., description="快照ID")
    as_of: date = Field(..., description="评分日期")
    confidence: Optional[float] = Field(None, description="置信度")
    
    @classmethod
    def from_snapshot(cls, snapshot):
        """从数据库快照创建响应"""
        return cls(
            industry_id=snapshot.industry_id,
            industry_name=snapshot.industry_name,
            score=float(snapshot.score),
            P=float(snapshot.P),
            E=float(snapshot.E),
            M=float(snapshot.M),
            R_neg=float(snapshot.R_neg),
            is_qualified_for_pool=snapshot.is_qualified_for_pool(),
            is_core_candidate=snapshot.is_core_candidate(),
            score_breakdown=snapshot.get_score_breakdown(),
            snapshot_id=str(snapshot.id),
            as_of=snapshot.as_of,
            confidence=float(snapshot.confidence) if snapshot.confidence else None
        )


class IndustryScoreSnapshotResponse(BaseModel):
    """行业评分快照响应（包含更多元数据）"""
    id: str
    industry_id: str
    industry_name: Optional[str]
    score: float
    P: float
    E: float
    M: float
    R_neg: float
    is_qualified_for_pool: bool
    is_core_candidate: bool
    score_breakdown: Dict[str, float]
    as_of: date
    confidence: Optional[float]
    is_partial: bool
    is_stale: bool
    created_at: datetime
    created_by: Optional[str]
    
    @classmethod
    def from_snapshot(cls, snapshot):
        """从数据库快照创建响应"""
        return cls(
            id=str(snapshot.id),
            industry_id=snapshot.industry_id,
            industry_name=snapshot.industry_name,
            score=float(snapshot.score),
            P=float(snapshot.P),
            E=float(snapshot.E),
            M=float(snapshot.M),
            R_neg=float(snapshot.R_neg),
            is_qualified_for_pool=snapshot.is_qualified_for_pool(),
            is_core_candidate=snapshot.is_core_candidate(),
            score_breakdown=snapshot.get_score_breakdown(),
            as_of=snapshot.as_of,
            confidence=float(snapshot.confidence) if snapshot.confidence else None,
            is_partial=snapshot.is_partial,
            is_stale=snapshot.is_stale,
            created_at=snapshot.created_at,
            created_by=snapshot.created_by
        )


class IndustryScoreListResponse(BaseModel):
    """行业评分列表响应"""
    items: List[IndustryScoreSnapshotResponse]
    total: int
    skip: int = 0
    limit: int = 100
    qualified_count: Optional[int] = None
    core_count: Optional[int] = None