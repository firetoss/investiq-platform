"""
InvestIQ Platform - 个股数据模型
个股信息和评分快照模型
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, JSON, Boolean, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from backend.app.core.database import Base


class Equity(Base):
    """个股基础信息表"""
    
    __tablename__ = "equities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), unique=True, nullable=False, comment="股票代码")
    exchange = Column(String(10), nullable=False, comment="交易所 (SH/SZ/HK)")
    name = Column(String(200), nullable=False, comment="股票名称")
    name_en = Column(String(200), comment="英文名称")
    
    # 基础信息
    industry_id = Column(UUID(as_uuid=True), comment="所属行业ID")
    sector = Column(String(100), comment="所属板块")
    market_type = Column(String(20), comment="市场类型 (A/H/US)")
    
    # 市场数据
    currency = Column(String(3), default="CNY", comment="计价货币")
    board_lot = Column(Integer, default=100, comment="每手股数")
    listing_date = Column(Date, comment="上市日期")
    
    # 公司信息
    company_name = Column(String(200), comment="公司全称")
    business_scope = Column(Text, comment="经营范围")
    main_business = Column(Text, comment="主营业务")
    
    # 财务基础数据
    market_cap = Column(Float, comment="总市值")
    free_float_market_cap = Column(Float, comment="自由流通市值")
    shares_outstanding = Column(Float, comment="总股本")
    free_float_shares = Column(Float, comment="自由流通股本")
    
    # 流动性数据
    adv_20 = Column(Float, comment="20日平均成交额")
    avg_turnover = Column(Float, comment="平均换手率")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否活跃")
    is_suspended = Column(Boolean, default=False, comment="是否停牌")
    delisting_date = Column(Date, comment="退市日期")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_equity_ticker', 'ticker'),
        Index('idx_equity_exchange', 'exchange'),
        Index('idx_equity_industry', 'industry_id'),
        Index('idx_equity_market_type', 'market_type'),
    )
    
    def __repr__(self):
        return f"<Equity(ticker='{self.ticker}', name='{self.name}')>"


class EquityScoreSnapshot(Base):
    """个股评分快照表 - 时态表"""
    
    __tablename__ = "equity_score_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equity_id = Column(UUID(as_uuid=True), nullable=False, comment="个股ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 评分维度 (基于PRD权重)
    score_q = Column(Float, comment="质量 (Q) - 30%")
    score_v = Column(Float, comment="估值 (V) - 20%")
    score_m = Column(Float, comment="动量 (M) - 25%")
    score_c = Column(Float, comment="政策契合 (C) - 15%")
    score_s = Column(Float, comment="份额/护城河 (S) - 10%")
    score_r_neg = Column(Float, comment="红旗 (R-) - 扣分项")
    
    # 计算结果
    total_score = Column(Float, comment="总评分")
    
    # 权重配置
    weights = Column(JSON, comment="权重配置 {Q: 0.30, V: 0.20, M: 0.25, C: 0.15, S: 0.10}")
    
    # 评分详情
    score_details = Column(JSON, comment="评分详细信息")
    
    # 质量指标详情
    quality_metrics = Column(JSON, comment="质量指标 {roe, ocf_ratio, margin_trend, leverage}")
    
    # 估值指标详情
    valuation_metrics = Column(JSON, comment="估值指标 {pe, pb, ev_ebitda, peg}")
    valuation_percentile = Column(Float, comment="估值历史分位")
    
    # 动量指标详情
    momentum_metrics = Column(JSON, comment="动量指标 {price_trend, eps_revision}")
    above_200dma = Column(Boolean, comment="是否在200日均线上方")
    
    # 红旗指标
    red_flags = Column(JSON, comment="红旗列表")
    has_red_flags = Column(Boolean, default=False, comment="是否有红旗")
    
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
        Index('idx_equity_score_equity_date', 'equity_id', 'as_of'),
        Index('idx_equity_score_ticker_date', 'ticker', 'as_of'),
        Index('idx_equity_score_date', 'as_of'),
        Index('idx_equity_score_total', 'total_score'),
        Index('idx_equity_score_created', 'created_at'),
    )
    
    def __repr__(self):
        return f"<EquityScoreSnapshot(ticker='{self.ticker}', score={self.total_score}, as_of='{self.as_of}')>"
    
    @property
    def score_breakdown(self) -> Dict[str, float]:
        """评分分解"""
        if not all([self.score_q is not None, self.score_v is not None, 
                   self.score_m is not None, self.score_c is not None, self.score_s is not None]):
            return {}
        
        weights = self.weights or {"Q": 0.30, "V": 0.20, "M": 0.25, "C": 0.15, "S": 0.10}
        
        breakdown = {
            "Q_weighted": self.score_q * weights.get("Q", 0.30),
            "V_weighted": self.score_v * weights.get("V", 0.20),
            "M_weighted": self.score_m * weights.get("M", 0.25),
            "C_weighted": self.score_c * weights.get("C", 0.15),
            "S_weighted": self.score_s * weights.get("S", 0.10),
            "R_neg": self.score_r_neg or 0,
            "total": self.total_score
        }
        
        return breakdown
    
    def calculate_score(self) -> float:
        """计算总评分"""
        if not all([self.score_q is not None, self.score_v is not None, 
                   self.score_m is not None, self.score_c is not None, self.score_s is not None]):
            return 0.0
        
        weights = self.weights or {"Q": 0.30, "V": 0.20, "M": 0.25, "C": 0.15, "S": 0.10}
        
        # 基础评分
        base_score = (
            self.score_q * weights.get("Q", 0.30) +
            self.score_v * weights.get("V", 0.20) +
            self.score_m * weights.get("M", 0.25) +
            self.score_c * weights.get("C", 0.15) +
            self.score_s * weights.get("S", 0.10)
        )
        
        # 扣除红旗分数
        red_flag_penalty = self.score_r_neg or 0
        final_score = base_score - red_flag_penalty
        
        return round(max(0, final_score), 2)
    
    def meets_observe_threshold(self, threshold: float = 65.0) -> bool:
        """检查是否达到观察阈值"""
        return (self.total_score >= threshold and not self.has_red_flags) if self.total_score else False
    
    def meets_build_threshold(self, threshold: float = 70.0) -> bool:
        """检查是否达到建仓阈值"""
        return (self.total_score >= threshold and not self.has_red_flags) if self.total_score else False
    
    def check_valuation_gate(self, max_percentile: float = 0.7, growth_max_percentile: float = 0.8, max_peg: float = 1.5) -> bool:
        """检查估值闸门"""
        if self.valuation_percentile is None:
            return False
        
        # 获取PEG值
        valuation_data = self.valuation_metrics or {}
        peg = valuation_data.get("peg")
        
        # 判断是否为成长股
        is_growth = peg is not None and peg <= max_peg
        
        if is_growth:
            return self.valuation_percentile <= growth_max_percentile and peg <= max_peg
        else:
            return self.valuation_percentile <= max_percentile
    
    def check_execution_gate(self) -> bool:
        """检查执行闸门"""
        return self.above_200dma if self.above_200dma is not None else False


class ValuationPercentileSnapshot(Base):
    """估值分位快照表"""
    
    __tablename__ = "valuation_percentile_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equity_id = Column(UUID(as_uuid=True), nullable=False, comment="个股ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 估值分位数据
    pe_percentile = Column(Float, comment="PE历史分位")
    pb_percentile = Column(Float, comment="PB历史分位")
    ev_ebitda_percentile = Column(Float, comment="EV/EBITDA历史分位")
    ev_sales_percentile = Column(Float, comment="EV/Sales历史分位")
    
    # 计算参数
    window_years = Column(Integer, default=5, comment="计算窗口年数")
    method = Column(String(50), comment="计算方法")
    
    # 回退链信息
    primary_metric = Column(String(20), comment="主要指标")
    fallback_used = Column(String(20), comment="使用的回退指标")
    is_partial = Column(Boolean, default=False, comment="是否使用回退")
    
    # 时间信息
    as_of = Column(Date, nullable=False, comment="业务日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_valuation_percentile_equity_date', 'equity_id', 'as_of'),
        Index('idx_valuation_percentile_ticker_date', 'ticker', 'as_of'),
        Index('idx_valuation_percentile_date', 'as_of'),
    )
    
    def __repr__(self):
        return f"<ValuationPercentileSnapshot(ticker='{self.ticker}', pe_pct={self.pe_percentile}, as_of='{self.as_of}')>"
    
    def get_primary_percentile(self) -> Optional[float]:
        """获取主要估值分位"""
        # 按优先级返回可用的估值分位
        for metric in ['pe_percentile', 'ev_ebitda_percentile', 'ev_sales_percentile', 'pb_percentile']:
            value = getattr(self, metric)
            if value is not None:
                return value
        return None


class TechnicalIndicator(Base):
    """技术指标数据表"""
    
    __tablename__ = "technical_indicators"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equity_id = Column(UUID(as_uuid=True), nullable=False, comment="个股ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 价格数据
    close_price = Column(Float, comment="收盘价")
    
    # 移动平均线
    ma_5 = Column(Float, comment="5日均线")
    ma_10 = Column(Float, comment="10日均线")
    ma_20 = Column(Float, comment="20日均线")
    ma_50 = Column(Float, comment="50日均线")
    ma_200 = Column(Float, comment="200日均线")
    
    # 技术指标
    rsi_14 = Column(Float, comment="14日RSI")
    macd = Column(Float, comment="MACD")
    macd_signal = Column(Float, comment="MACD信号线")
    bollinger_upper = Column(Float, comment="布林带上轨")
    bollinger_lower = Column(Float, comment="布林带下轨")
    
    # 成交量指标
    volume = Column(Float, comment="成交量")
    volume_ma_20 = Column(Float, comment="20日平均成交量")
    volume_ratio = Column(Float, comment="量比")
    
    # 动量指标
    momentum_1d = Column(Float, comment="1日动量")
    momentum_5d = Column(Float, comment="5日动量")
    momentum_20d = Column(Float, comment="20日动量")
    
    # 状态标识
    above_ma_200 = Column(Boolean, comment="是否在200日均线上方")
    is_uptrend = Column(Boolean, comment="是否上升趋势")
    
    # 时间信息
    trade_date = Column(Date, nullable=False, comment="交易日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_technical_equity_date', 'equity_id', 'trade_date'),
        Index('idx_technical_ticker_date', 'ticker', 'trade_date'),
        Index('idx_technical_date', 'trade_date'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicator(ticker='{self.ticker}', date='{self.trade_date}', price={self.close_price})>"
