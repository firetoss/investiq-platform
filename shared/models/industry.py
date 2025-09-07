"""
InvestIQ Platform - 行业评分数据模型
"""

from sqlalchemy import Column, String, DECIMAL, Date, Boolean, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from .base import BaseModel, validate_score, get_current_date, ScoreMethod
from typing import Optional, Dict, List
from datetime import date


class IndustryScoreSnapshot(BaseModel):
    """行业评分快照表"""
    __tablename__ = 'industry_score_snapshots'
    
    # 基本信息
    industry_id = Column(String(100), nullable=False, comment="行业ID")
    industry_name = Column(String(200), comment="行业名称")
    
    # 评分维度 (0-100分)
    P = Column(DECIMAL(5, 2), comment="政策强度评分")
    E = Column(DECIMAL(5, 2), comment="落地证据评分")
    M = Column(DECIMAL(5, 2), comment="市场确认评分") 
    R_neg = Column(DECIMAL(5, 2), comment="风险扣分")
    
    # 总分
    score = Column(DECIMAL(5, 2), nullable=False, comment="行业总分")
    
    # 元数据
    method = Column(String(50), default=ScoreMethod.STANDARD, comment="评分方法")
    as_of = Column(Date, nullable=False, default=get_current_date, comment="评分日期")
    source = Column(String(200), comment="数据源")
    
    # 数据质量标记
    is_partial = Column(Boolean, default=False, comment="是否部分数据")
    is_stale = Column(Boolean, default=False, comment="是否过期数据")
    confidence = Column(DECIMAL(5, 2), comment="置信度 (0-100)")
    
    # 扩展字段
    metadata = Column(JSONB, comment="扩展元数据")
    
    def calculate_score(self) -> float:
        """计算行业评分
        Score = 0.35*P + 0.25*E + 0.25*M + 0.15*(100-R_neg)
        """
        if None in [self.P, self.E, self.M, self.R_neg]:
            return 0.0
            
        score = (0.35 * float(self.P) + 
                0.25 * float(self.E) + 
                0.25 * float(self.M) + 
                0.15 * (100 - float(self.R_neg)))
        
        return round(score, 2)
    
    def update_score(self):
        """更新总分"""
        self.score = self.calculate_score()
    
    def is_qualified_for_pool(self) -> bool:
        """是否符合入池条件 (≥70分)"""
        return float(self.score) >= 70.0
    
    def is_core_candidate(self) -> bool:
        """是否为核心候选 (≥75分)"""
        return float(self.score) >= 75.0
    
    def get_score_breakdown(self) -> Dict[str, float]:
        """获取评分拆解"""
        return {
            "政策强度 (35%)": float(self.P) * 0.35 if self.P else 0,
            "落地证据 (25%)": float(self.E) * 0.25 if self.E else 0,
            "市场确认 (25%)": float(self.M) * 0.25 if self.M else 0,
            "风险扣分 (15%)": (100 - float(self.R_neg)) * 0.15 if self.R_neg else 0,
            "总分": float(self.score)
        }
    
    def validate_scores(self) -> List[str]:
        """验证评分有效性"""
        errors = []
        
        if self.P is not None and not validate_score(float(self.P)):
            errors.append("政策强度评分应在0-100之间")
        
        if self.E is not None and not validate_score(float(self.E)):
            errors.append("落地证据评分应在0-100之间")
            
        if self.M is not None and not validate_score(float(self.M)):
            errors.append("市场确认评分应在0-100之间")
            
        if self.R_neg is not None and not validate_score(float(self.R_neg)):
            errors.append("风险扣分应在0-100之间")
        
        if self.confidence is not None and not validate_score(float(self.confidence)):
            errors.append("置信度应在0-100之间")
            
        return errors
    
    def to_dict(self) -> Dict:
        """转换为字典，包含计算字段"""
        result = super().to_dict()
        result.update({
            "is_qualified_for_pool": self.is_qualified_for_pool(),
            "is_core_candidate": self.is_core_candidate(),
            "score_breakdown": self.get_score_breakdown()
        })
        return result


# 创建索引
Index('idx_industry_snapshots_industry_id', IndustryScoreSnapshot.industry_id)
Index('idx_industry_snapshots_as_of', IndustryScoreSnapshot.as_of.desc())
Index('idx_industry_snapshots_score', IndustryScoreSnapshot.score.desc())
Index('idx_industry_snapshots_compound', 
      IndustryScoreSnapshot.industry_id, 
      IndustryScoreSnapshot.as_of.desc(),
      IndustryScoreSnapshot.score.desc())


class IndustryMaster(BaseModel):
    """行业主数据表"""
    __tablename__ = 'industry_master'
    
    # 基本信息
    industry_id = Column(String(100), nullable=False, unique=True, comment="行业ID")
    industry_name = Column(String(200), nullable=False, comment="行业名称")
    industry_name_en = Column(String(200), comment="英文名称")
    
    # 层级信息
    parent_id = Column(String(100), comment="父行业ID")
    level = Column(String(10), comment="行业层级")
    path = Column(String(500), comment="行业路径")
    
    # 分类信息
    category = Column(String(100), comment="行业分类")
    sector = Column(String(100), comment="所属板块")
    theme = Column(String(100), comment="主题概念")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    # 扩展信息
    description = Column(Text, comment="行业描述")
    keywords = Column(JSONB, comment="关键词")
    metadata = Column(JSONB, comment="扩展元数据")
    
    def get_full_path(self) -> str:
        """获取完整路径名称"""
        return self.path if self.path else self.industry_name


# 创建索引
Index('idx_industry_master_id', IndustryMaster.industry_id)
Index('idx_industry_master_name', IndustryMaster.industry_name)
Index('idx_industry_master_parent', IndustryMaster.parent_id)
Index('idx_industry_master_active', IndustryMaster.is_active)