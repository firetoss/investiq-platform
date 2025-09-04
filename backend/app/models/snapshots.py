"""
InvestIQ Platform - Point-in-Time 快照模型
实现时间旅行查询和数据版本控制
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Boolean, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.declarative import declared_attr
import uuid

from backend.app.models.base import Base, TimestampMixin


class PITSnapshotMixin:
    """Point-in-Time 快照基础混入类"""
    
    @declared_attr
    def as_of(cls):
        return Column(Date, nullable=False, comment="业务日期")
    
    @declared_attr
    def snapshot_ts(cls):
        return Column(DateTime, nullable=False, default=datetime.utcnow, comment="快照生成时间戳")
    
    @declared_attr
    def is_partial(cls):
        return Column(Boolean, nullable=False, default=False, comment="是否为部分数据")
    
    @declared_attr
    def is_stale(cls):
        return Column(Boolean, nullable=False, default=False, comment="是否为过期数据")
    
    @declared_attr
    def confidence(cls):
        return Column(Float, nullable=False, default=100.0, comment="置信度 (0-100)")
    
    @declared_attr
    def data_source(cls):
        return Column(String(100), nullable=False, comment="数据来源")
    
    @declared_attr
    def method(cls):
        return Column(String(50), nullable=False, comment="计算方法")


class IndustryScoreSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """行业评分时点快照"""
    
    __tablename__ = "industry_score_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    industry_id = Column(String(50), nullable=False, comment="行业ID")
    industry_name = Column(String(200), nullable=False, comment="行业名称")
    
    # 评分数据
    overall_score = Column(Float, nullable=False, comment="总评分")
    policy_score = Column(Float, nullable=False, comment="政策评分 (P)")
    execution_score = Column(Float, nullable=False, comment="落地评分 (E)")
    market_score = Column(Float, nullable=False, comment="市场评分 (M)")
    risk_score = Column(Float, nullable=False, comment="风险评分 (R-)")
    
    # 权重和计算详情
    weights_used = Column(JSONB, nullable=False, comment="使用的权重")
    breakdown = Column(JSONB, nullable=False, comment="评分分解")
    calculation_details = Column(JSONB, nullable=True, comment="计算详情")
    
    # 阈值检查结果
    meets_threshold = Column(Boolean, nullable=False, comment="是否达到门槛")
    is_core_candidate = Column(Boolean, nullable=False, comment="是否为核心候选")
    
    # 证据相关
    evidence_count = Column(Integer, nullable=False, default=0, comment="证据数量")
    evidence_quality_score = Column(Float, nullable=True, comment="证据质量评分")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_industry_score_snapshots_industry_asof", "industry_id", "as_of"),
        Index("ix_industry_score_snapshots_ts", "snapshot_ts"),
        Index("ix_industry_score_snapshots_score", "overall_score"),
        {"comment": "行业评分时点快照表"}
    )


class EquityScoreSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """个股评分时点快照"""
    
    __tablename__ = "equity_score_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equity_id = Column(String(50), nullable=False, comment="股票ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    company_name = Column(String(200), nullable=False, comment="公司名称")
    
    # 评分数据
    overall_score = Column(Float, nullable=False, comment="总评分")
    quality_score = Column(Float, nullable=False, comment="质量评分 (Q)")
    valuation_score = Column(Float, nullable=False, comment="估值评分 (V)")
    momentum_score = Column(Float, nullable=False, comment="动量评分 (M)")
    catalyst_score = Column(Float, nullable=False, comment="催化评分 (C)")
    share_score = Column(Float, nullable=False, comment="份额评分 (S)")
    red_flag_penalty = Column(Float, nullable=False, default=0, comment="红旗扣分")
    
    # 权重和计算详情
    weights_used = Column(JSONB, nullable=False, comment="使用的权重")
    breakdown = Column(JSONB, nullable=False, comment="评分分解")
    red_flags = Column(JSONB, nullable=True, comment="红旗详情")
    
    # 阈值检查结果
    meets_observe_threshold = Column(Boolean, nullable=False, comment="达到观察门槛")
    meets_build_threshold = Column(Boolean, nullable=False, comment="达到建仓门槛")
    has_red_flags = Column(Boolean, nullable=False, default=False, comment="是否有红旗")
    
    # 证据相关
    evidence_count = Column(Integer, nullable=False, default=0, comment="证据数量")
    evidence_quality_score = Column(Float, nullable=True, comment="证据质量评分")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_equity_score_snapshots_ticker_asof", "ticker", "as_of"),
        Index("ix_equity_score_snapshots_ts", "snapshot_ts"),
        Index("ix_equity_score_snapshots_score", "overall_score"),
        {"comment": "个股评分时点快照表"}
    )


class ValuationPercentileSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """估值分位时点快照"""
    
    __tablename__ = "valuation_percentile_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 估值分位数据
    window_years = Column(Integer, nullable=False, comment="窗口年数")
    pe_percentile = Column(Float, nullable=True, comment="PE分位")
    ev_ebitda_percentile = Column(Float, nullable=True, comment="EV/EBITDA分位")
    ev_sales_percentile = Column(Float, nullable=True, comment="EV/Sales分位")
    pb_percentile = Column(Float, nullable=True, comment="PB分位")
    
    # 主要使用的分位
    primary_percentile = Column(Float, nullable=False, comment="主要分位")
    primary_method = Column(String(50), nullable=False, comment="主要方法")
    
    # 回退链信息
    fallback_used = Column(Boolean, nullable=False, default=False, comment="是否使用了回退")
    fallback_chain = Column(JSONB, nullable=True, comment="回退链详情")
    
    # 计算原始数据
    current_pe = Column(Float, nullable=True, comment="当前PE")
    current_ev_ebitda = Column(Float, nullable=True, comment="当前EV/EBITDA") 
    current_ev_sales = Column(Float, nullable=True, comment="当前EV/Sales")
    current_pb = Column(Float, nullable=True, comment="当前PB")
    
    # 行业对比
    sector_z_score = Column(Float, nullable=True, comment="行业Z-Score")
    sector_percentile = Column(Float, nullable=True, comment="行业分位")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_valuation_percentile_snapshots_ticker_asof", "ticker", "as_of"),
        Index("ix_valuation_percentile_snapshots_percentile", "primary_percentile"),
        {"comment": "估值分位时点快照表"}
    )


class TechnicalIndicatorSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """技术指标时点快照"""
    
    __tablename__ = "technical_indicator_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 价格数据
    current_price = Column(Float, nullable=False, comment="当前价格")
    previous_close = Column(Float, nullable=False, comment="前收盘价")
    
    # 均线数据
    ma_20 = Column(Float, nullable=True, comment="20日均线")
    ma_50 = Column(Float, nullable=True, comment="50日均线")
    ma_200 = Column(Float, nullable=True, comment="200日均线")
    
    # 200DMA状态 (关键指标)
    above_200dma = Column(Boolean, nullable=False, comment="是否在200日均线上方")
    days_above_200dma = Column(Integer, nullable=True, comment="连续多少天在200DMA上方")
    dma_trend_confirmed = Column(Boolean, nullable=False, default=False, comment="趋势是否确认")
    
    # 停牌处理
    is_suspended = Column(Boolean, nullable=False, default=False, comment="是否停牌")
    suspension_days = Column(Integer, nullable=False, default=0, comment="停牌天数")
    last_trading_day = Column(Date, nullable=True, comment="最后交易日")
    
    # 成交量数据
    volume = Column(Float, nullable=True, comment="成交量")
    adv_20 = Column(Float, nullable=True, comment="20日平均成交额")
    turnover_rate = Column(Float, nullable=True, comment="换手率")
    
    # 技术指标
    rsi_14 = Column(Float, nullable=True, comment="14日RSI")
    atr_14 = Column(Float, nullable=True, comment="14日ATR")
    volatility_20 = Column(Float, nullable=True, comment="20日波动率")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_technical_indicator_snapshots_ticker_asof", "ticker", "as_of"),
        Index("ix_technical_indicator_snapshots_200dma", "above_200dma"),
        Index("ix_technical_indicator_snapshots_suspended", "is_suspended"),
        {"comment": "技术指标时点快照表"}
    )


class PortfolioSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """投资组合时点快照"""
    
    __tablename__ = "portfolio_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, comment="组合ID")
    portfolio_name = Column(String(200), nullable=False, comment="组合名称")
    
    # 组合价值
    total_value = Column(Float, nullable=False, comment="总价值")
    cash_amount = Column(Float, nullable=False, comment="现金金额")
    invested_amount = Column(Float, nullable=False, comment="投资金额")
    leverage_ratio = Column(Float, nullable=False, default=1.0, comment="杠杆比率")
    
    # 收益指标
    total_pnl = Column(Float, nullable=False, comment="总盈亏")
    total_pnl_pct = Column(Float, nullable=False, comment="总收益率")
    daily_pnl = Column(Float, nullable=False, comment="日盈亏")
    max_drawdown = Column(Float, nullable=False, comment="最大回撤")
    
    # 风险指标
    sharpe_ratio = Column(Float, nullable=True, comment="夏普比率")
    volatility = Column(Float, nullable=True, comment="波动率")
    beta = Column(Float, nullable=True, comment="Beta系数")
    var_95 = Column(Float, nullable=True, comment="VaR(95%)")
    
    # A/B/C分层配置
    tier_allocation = Column(JSONB, nullable=False, comment="分层配置")
    tier_performance = Column(JSONB, nullable=False, comment="分层表现")
    
    # 断路器状态
    circuit_breaker_level = Column(Integer, nullable=False, default=0, comment="断路器级别")
    circuit_breaker_triggered = Column(Boolean, nullable=False, default=False, comment="断路器是否触发")
    circuit_breaker_actions = Column(JSONB, nullable=True, comment="断路器执行动作")
    
    # 持仓快照
    positions_snapshot = Column(JSONB, nullable=False, comment="持仓快照")
    position_count = Column(Integer, nullable=False, comment="持仓数量")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_portfolio_snapshots_portfolio_asof", "portfolio_id", "as_of"),
        Index("ix_portfolio_snapshots_pnl", "total_pnl_pct"),
        Index("ix_portfolio_snapshots_drawdown", "max_drawdown"),
        {"comment": "投资组合时点快照表"}
    )


class EvidenceSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """证据链时点快照"""
    
    __tablename__ = "evidence_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evidence_id = Column(UUID(as_uuid=True), nullable=False, comment="证据ID")
    
    # 关联实体
    entity_type = Column(String(50), nullable=False, comment="实体类型")
    entity_id = Column(String(100), nullable=False, comment="实体ID")
    
    # 证据内容
    evidence_type = Column(String(50), nullable=False, comment="证据类型")
    title = Column(String(500), nullable=False, comment="证据标题")
    content_summary = Column(Text, nullable=True, comment="内容摘要")
    source_url = Column(String(1000), nullable=True, comment="源URL")
    
    # 文件信息
    file_hash = Column(String(64), nullable=True, comment="文件SHA-256哈希")
    file_size = Column(Integer, nullable=True, comment="文件大小")
    file_type = Column(String(50), nullable=True, comment="文件类型")
    storage_path = Column(String(500), nullable=True, comment="存储路径")
    
    # 质量评分
    quality_score = Column(Float, nullable=False, default=80.0, comment="质量评分")
    reliability_score = Column(Float, nullable=False, default=80.0, comment="可靠性评分")
    relevance_score = Column(Float, nullable=False, default=80.0, comment="相关性评分")
    
    # 生命周期
    lifecycle_stage = Column(String(20), nullable=False, default="active", comment="生命周期阶段")
    retention_until = Column(Date, nullable=True, comment="保留到期日")
    archive_status = Column(String(20), nullable=False, default="active", comment="归档状态")
    
    # 修正链
    superseded_by = Column(UUID(as_uuid=True), nullable=True, comment="被哪个版本替代")
    correction_reason = Column(String(500), nullable=True, comment="修正原因")
    
    __table_args__ = (
        Index("ix_evidence_snapshots_entity", "entity_type", "entity_id", "as_of"),
        Index("ix_evidence_snapshots_type", "evidence_type"),
        Index("ix_evidence_snapshots_hash", "file_hash"),
        {"comment": "证据链时点快照表"}
    )


class ConfigurationSnapshot(Base, TimestampMixin, PITSnapshotMixin):
    """配置参数时点快照"""
    
    __tablename__ = "configuration_snapshots"
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_version = Column(String(50), nullable=False, comment="配置版本")
    config_hash = Column(String(64), nullable=False, comment="配置哈希")
    
    # 配置内容
    scoring_config = Column(JSONB, nullable=False, comment="评分配置")
    gate_config = Column(JSONB, nullable=False, comment="闸门配置")
    liquidity_config = Column(JSONB, nullable=False, comment="流动性配置")
    portfolio_config = Column(JSONB, nullable=False, comment="组合配置")
    alert_config = Column(JSONB, nullable=False, comment="告警配置")
    
    # 红线参数 (不可随意更改)
    red_line_params = Column(JSONB, nullable=False, comment="红线参数")
    red_line_hash = Column(String(64), nullable=False, comment="红线参数哈希")
    
    # 变更信息
    changed_by = Column(String(100), nullable=False, comment="变更人")
    change_reason = Column(String(500), nullable=False, comment="变更原因")
    approval_status = Column(String(20), nullable=False, default="pending", comment="审批状态")
    approved_by = Column(String(100), nullable=True, comment="审批人")
    
    # CI对比器检查结果
    ci_check_passed = Column(Boolean, nullable=False, default=False, comment="CI检查是否通过")
    ci_check_details = Column(JSONB, nullable=True, comment="CI检查详情")
    
    __table_args__ = (
        Index("ix_configuration_snapshots_version", "config_version"),
        Index("ix_configuration_snapshots_hash", "config_hash"),
        Index("ix_configuration_snapshots_red_line", "red_line_hash"),
        {"comment": "配置参数时点快照表"}
    )