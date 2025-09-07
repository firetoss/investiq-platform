"""
InvestIQ Platform - 个股评分数据模型
"""

from sqlalchemy import Column, String, DECIMAL, Date, Boolean, Integer, BigInteger, Index, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from .base import BaseModel, validate_score, validate_ticker, validate_currency, validate_exchange, get_current_date
from typing import Optional, Dict, List
from datetime import date


class Equity(BaseModel):
    """个股基础信息表"""
    __tablename__ = 'equities'
    
    # 基本信息
    ticker = Column(String(20), nullable=False, unique=True, comment="股票代码")
    exchange = Column(String(10), nullable=False, comment="交易所")
    name = Column(String(200), nullable=False, comment="股票名称")
    name_en = Column(String(200), comment="英文名称")
    
    # 市值信息 (单位: 分/港仙)
    market_cap = Column(BigInteger, comment="总市值")
    free_float_cap = Column(BigInteger, comment="自由流通市值")
    
    # 流动性信息
    adv20 = Column(BigInteger, comment="20日平均成交额")
    board_lot = Column(Integer, default=100, comment="每手股数")
    
    # 基础属性
    currency = Column(String(3), default='CNY', comment="交易货币")
    industry_id = Column(String(100), comment="所属行业ID")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_suspended = Column(Boolean, default=False, comment="是否停牌")
    
    # 扩展信息
    metadata = Column(JSONB, comment="扩展元数据")
    
    def validate_basic_info(self) -> List[str]:
        """验证基础信息"""
        errors = []
        
        if not validate_ticker(self.ticker):
            errors.append(f"无效的股票代码格式: {self.ticker}")
            
        if not validate_exchange(self.exchange):
            errors.append(f"无效的交易所代码: {self.exchange}")
            
        if not validate_currency(self.currency):
            errors.append(f"无效的货币代码: {self.currency}")
            
        return errors
    
    def get_market_type(self) -> str:
        """获取市场类型"""
        if self.ticker.endswith('.HK'):
            return 'H'
        elif self.ticker.endswith('.SH') or self.ticker.endswith('.SZ'):
            return 'A'
        return 'Unknown'
    
    def to_dict(self) -> Dict:
        """转换为字典，包含计算字段"""
        result = super().to_dict()
        result.update({
            "market_type": self.get_market_type()
        })
        return result


class EquityScoreSnapshot(BaseModel):
    """个股评分快照表"""
    __tablename__ = 'equity_score_snapshots'
    
    # 关联股票
    ticker = Column(String(20), ForeignKey('equities.ticker'), nullable=False, comment="股票代码")
    
    # 评分维度 (0-100分)
    Q = Column(DECIMAL(5, 2), comment="质量评分 (30%)")
    V = Column(DECIMAL(5, 2), comment="估值评分 (20%)")
    M = Column(DECIMAL(5, 2), comment="动量评分 (25%)")
    C = Column(DECIMAL(5, 2), comment="政策契合评分 (15%)")
    S = Column(DECIMAL(5, 2), comment="份额护城河评分 (10%)")
    R_neg = Column(DECIMAL(5, 2), comment="红旗扣分")
    
    # 总分
    score = Column(DECIMAL(5, 2), nullable=False, comment="个股总分")
    
    # 元数据
    as_of = Column(Date, nullable=False, default=get_current_date, comment="评分日期")
    
    # 数据质量标记
    is_partial = Column(Boolean, default=False, comment="是否部分数据")
    is_stale = Column(Boolean, default=False, comment="是否过期数据")
    confidence = Column(DECIMAL(5, 2), comment="置信度 (0-100)")
    
    # 扩展字段
    metadata = Column(JSONB, comment="扩展元数据")
    
    # 关系
    equity = relationship("Equity", back_populates="score_snapshots")
    
    def calculate_score(self) -> float:
        """计算个股评分
        Score = Q*0.3 + V*0.2 + M*0.25 + C*0.15 + S*0.1 - R_neg
        """
        if None in [self.Q, self.V, self.M, self.C, self.S]:
            return 0.0
            
        base_score = (float(self.Q) * 0.30 +
                     float(self.V) * 0.20 +
                     float(self.M) * 0.25 +
                     float(self.C) * 0.15 +
                     float(self.S) * 0.10)
        
        # 扣除红旗分数
        red_flag_deduction = float(self.R_neg) if self.R_neg else 0
        
        final_score = max(0, base_score - red_flag_deduction)
        return round(final_score, 2)
    
    def update_score(self):
        """更新总分"""
        self.score = self.calculate_score()
    
    def is_qualified_for_observation(self) -> bool:
        """是否符合观察条件 (≥65分且无红旗)"""
        return float(self.score) >= 65.0 and (not self.R_neg or float(self.R_neg) == 0)
    
    def is_qualified_for_building(self) -> bool:
        """是否符合建仓条件 (≥70分)"""
        return float(self.score) >= 70.0
    
    def has_red_flags(self) -> bool:
        """是否存在红旗"""
        return self.R_neg is not None and float(self.R_neg) > 0
    
    def get_score_breakdown(self) -> Dict[str, float]:
        """获取评分拆解"""
        return {
            "质量 (30%)": float(self.Q) * 0.30 if self.Q else 0,
            "估值 (20%)": float(self.V) * 0.20 if self.V else 0,
            "动量 (25%)": float(self.M) * 0.25 if self.M else 0,
            "政策契合 (15%)": float(self.C) * 0.15 if self.C else 0,
            "份额护城河 (10%)": float(self.S) * 0.10 if self.S else 0,
            "红旗扣分": -(float(self.R_neg)) if self.R_neg else 0,
            "总分": float(self.score)
        }
    
    def validate_scores(self) -> List[str]:
        """验证评分有效性"""
        errors = []
        
        score_fields = [
            ("质量评分", self.Q),
            ("估值评分", self.V),
            ("动量评分", self.M),
            ("政策契合评分", self.C),
            ("份额护城河评分", self.S),
            ("红旗扣分", self.R_neg),
            ("置信度", self.confidence)
        ]
        
        for field_name, value in score_fields:
            if value is not None and not validate_score(float(value)):
                errors.append(f"{field_name}应在0-100之间")
                
        return errors
    
    def to_dict(self) -> Dict:
        """转换为字典，包含计算字段"""
        result = super().to_dict()
        result.update({
            "is_qualified_for_observation": self.is_qualified_for_observation(),
            "is_qualified_for_building": self.is_qualified_for_building(),
            "has_red_flags": self.has_red_flags(),
            "score_breakdown": self.get_score_breakdown()
        })
        return result


class ValuationPercentileSnapshot(BaseModel):
    """估值分位数快照表"""
    __tablename__ = 'valuation_percentile_snapshots'
    
    # 关联股票
    ticker = Column(String(20), ForeignKey('equities.ticker'), nullable=False, comment="股票代码")
    
    # 估值信息
    window_years = Column(Integer, default=5, comment="历史窗口年数")
    percentile = Column(DECIMAL(5, 2), nullable=False, comment="历史分位数")
    method = Column(String(50), default='PE', comment="估值方法")
    
    # 时间信息
    as_of = Column(Date, nullable=False, default=get_current_date, comment="计算日期")
    
    # 数据质量
    is_partial = Column(Boolean, default=False, comment="是否部分数据")
    
    # 扩展字段
    metadata = Column(JSONB, comment="扩展元数据")
    
    # 关系
    equity = relationship("Equity")
    
    def is_attractive_valuation(self, growth_stock: bool = False) -> bool:
        """是否具有吸引力的估值"""
        threshold = 0.80 if growth_stock else 0.70  # 成长股80%，价值股70%
        return float(self.percentile) <= threshold
    
    def get_valuation_level(self) -> str:
        """获取估值水平描述"""
        pct = float(self.percentile)
        if pct <= 0.2:
            return "极低估值"
        elif pct <= 0.4:
            return "低估值"
        elif pct <= 0.6:
            return "合理估值"
        elif pct <= 0.8:
            return "偏高估值"
        else:
            return "高估值"


# 为Equity模型添加关系
Equity.score_snapshots = relationship("EquityScoreSnapshot", back_populates="equity", cascade="all, delete-orphan")
Equity.valuation_snapshots = relationship("ValuationPercentileSnapshot", cascade="all, delete-orphan")


# 创建索引
Index('idx_equities_ticker', Equity.ticker)
Index('idx_equities_exchange', Equity.exchange)
Index('idx_equities_industry', Equity.industry_id)
Index('idx_equities_active', Equity.is_active)

Index('idx_equity_snapshots_ticker', EquityScoreSnapshot.ticker)
Index('idx_equity_snapshots_as_of', EquityScoreSnapshot.as_of.desc())
Index('idx_equity_snapshots_score', EquityScoreSnapshot.score.desc())
Index('idx_equity_snapshots_compound', 
      EquityScoreSnapshot.ticker, 
      EquityScoreSnapshot.as_of.desc(),
      EquityScoreSnapshot.score.desc())

Index('idx_valuation_snapshots_ticker', ValuationPercentileSnapshot.ticker)
Index('idx_valuation_snapshots_as_of', ValuationPercentileSnapshot.as_of.desc())
Index('idx_valuation_snapshots_method', ValuationPercentileSnapshot.method)