"""
InvestIQ Platform - 行业数据模型
行业信息和评分快照模型
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from backend.app.core.database import Base


class Industry(Base):
    """行业基础信息表"""
    
    __tablename__ = "industries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(50), unique=True, nullable=False, comment="行业代码")
    name = Column(String(200), nullable=False, comment="行业名称")
    name_en = Column(String(200), comment="英文名称")
    level = Column(Integer, default=1, comment="行业层级")
    parent_id = Column(UUID(as_uuid=True), comment="父行业ID")
    
    # 分类信息
    classification_system = Column(String(50), comment="分类体系 (GICS/SHENWAN/CITIC等)")
    sector = Column(String(100), comment="所属板块")
    
    # 基础属性
    description = Column(Text, comment="行业描述")
    keywords = Column(JSON, comment="关键词列表")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否活跃")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_industry_code', 'code'),
        Index('idx_industry_parent', 'parent_id'),
        Index('idx_industry_classification', 'classification_system'),
    )
    
    def __repr__(self):
        return f"<Industry(code='{self.code}', name='{self.name}')>"


class IndustryScoreSnapshot(Base):
    """行业评分快照表 - 时态表"""
    
    __tablename__ = "industry_score_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    industry_id = Column(UUID(as_uuid=True), nullable=False, comment="行业ID")
    
    # 评分维度 (基于PRD公式)
    score_p = Column(Float, comment="政策强度与确定性 (P)")
    score_e = Column(Float, comment="落地证据 (E)")
    score_m = Column(Float, comment="市场确认 (M)")
    score_r_neg = Column(Float, comment="风险因子 (R-)")
    
    # 计算结果
    total_score = Column(Float, comment="总评分")
    
    # 权重配置 (可调整)
    weights = Column(JSON, comment="权重配置 {P: 0.35, E: 0.25, M: 0.25, R: 0.15}")
    
    # 评分详情
    score_details = Column(JSON, comment="评分详细信息")
    
    # 证据和来源
    evidence_count = Column(Integer, default=0, comment="证据数量")
    evidence_hash = Column(String(64), comment="证据哈希")
    
    # 状态信息
    confidence = Column(Float, comment="置信度 (0-100)")
    is_partial = Column(Boolean, default=False, comment="是否部分数据")
    is_stale = Column(Boolean, default=False, comment="是否过期数据")
    
    # 时间信息
    as_of = Column(Date, nullable=False, comment="业务日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 审计信息
    editor = Column(String(100), comment="编辑者")
    source = Column(String(100), comment="数据来源")
    method = Column(String(50), comment="计算方法")
    
    # 索引
    __table_args__ = (
        Index('idx_industry_score_industry_date', 'industry_id', 'as_of'),
        Index('idx_industry_score_date', 'as_of'),
        Index('idx_industry_score_total', 'total_score'),
        Index('idx_industry_score_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<IndustryScoreSnapshot(industry_id='{self.industry_id}', score={self.total_score}, as_of='{self.as_of}')>"
    
    @property
    def score_breakdown(self) -> Dict[str, float]:
        """评分分解"""
        if not all([self.score_p, self.score_e, self.score_m, self.score_r_neg]):
            return {}
        
        weights = self.weights or {"P": 0.35, "E": 0.25, "M": 0.25, "R": 0.15}
        
        return {
            "P_weighted": self.score_p * weights.get("P", 0.35),
            "E_weighted": self.score_e * weights.get("E", 0.25),
            "M_weighted": self.score_m * weights.get("M", 0.25),
            "R_weighted": (100 - self.score_r_neg) * weights.get("R", 0.15),
            "total": self.total_score
        }
    
    def calculate_score(self) -> float:
        """计算总评分"""
        if not all([self.score_p is not None, self.score_e is not None, 
                   self.score_m is not None, self.score_r_neg is not None]):
            return 0.0
        
        weights = self.weights or {"P": 0.35, "E": 0.25, "M": 0.25, "R": 0.15}
        
        # 基于PRD公式: Score^{Ind} = 0.35P + 0.25E + 0.25M + 0.15×(100-R^{-})
        score = (
            self.score_p * weights.get("P", 0.35) +
            self.score_e * weights.get("E", 0.25) +
            self.score_m * weights.get("M", 0.25) +
            (100 - self.score_r_neg) * weights.get("R", 0.15)
        )
        
        return round(score, 2)
    
    def meets_threshold(self, threshold: float = 70.0) -> bool:
        """检查是否达到入池阈值"""
        return self.total_score >= threshold if self.total_score else False
    
    def is_core_candidate(self, threshold: float = 75.0) -> bool:
        """检查是否为核心候选"""
        return self.total_score >= threshold if self.total_score else False


class IndustryMetrics(Base):
    """行业指标数据表"""
    
    __tablename__ = "industry_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    industry_id = Column(UUID(as_uuid=True), nullable=False, comment="行业ID")
    
    # 市场指标
    market_cap = Column(Float, comment="总市值")
    avg_pe = Column(Float, comment="平均PE")
    avg_pb = Column(Float, comment="平均PB")
    avg_roe = Column(Float, comment="平均ROE")
    
    # 动量指标
    momentum_1m = Column(Float, comment="1个月动量")
    momentum_3m = Column(Float, comment="3个月动量")
    momentum_6m = Column(Float, comment="6个月动量")
    above_200dma_pct = Column(Float, comment="200日均线上方占比")
    
    # 估值分位
    pe_percentile = Column(Float, comment="PE历史分位")
    pb_percentile = Column(Float, comment="PB历史分位")
    
    # 拥挤度指标
    crowding_score = Column(Float, comment="拥挤度评分")
    position_concentration = Column(Float, comment="持仓集中度")
    
    # 时间信息
    as_of = Column(Date, nullable=False, comment="业务日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_industry_metrics_industry_date', 'industry_id', 'as_of'),
        Index('idx_industry_metrics_date', 'as_of'),
    )
    
    def __repr__(self):
        return f"<IndustryMetrics(industry_id='{self.industry_id}', as_of='{self.as_of}')>"
