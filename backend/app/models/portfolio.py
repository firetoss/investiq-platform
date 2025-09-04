"""
InvestIQ Platform - 投资组合数据模型
投资组合、持仓和风险管理模型
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, JSON, Boolean, Index, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
import enum

from backend.app.core.database import Base


class PositionTier(enum.Enum):
    """持仓分层"""
    A = "A"  # 核心持仓 12-15%
    B = "B"  # 重要持仓 8-10%
    C = "C"  # 战术持仓 3-5%


class CircuitBreakerLevel(enum.Enum):
    """断路器级别"""
    L0 = "L0"  # 正常状态
    L1 = "L1"  # -10% 去融资、暂停新开战术仓
    L2 = "L2"  # -20% 战术仓减半、观察清零
    L3 = "L3"  # -30% 仅保留2-3只核心票


class Portfolio(Base):
    """投资组合表"""
    
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, comment="组合名称")
    description = Column(Text, comment="组合描述")
    
    # 资金配置
    total_capital = Column(Float, nullable=False, comment="总资金")
    cash_position = Column(Float, default=0, comment="现金头寸")
    leverage_ratio = Column(Float, default=1.0, comment="杠杆比例")
    max_leverage = Column(Float, default=1.1, comment="最大杠杆")
    
    # 组合配置
    tier_config = Column(JSON, comment="分层配置 {A: [0.12, 0.15], B: [0.08, 0.10], C: [0.03, 0.05]}")
    risk_config = Column(JSON, comment="风险配置")
    
    # 断路器状态
    circuit_breaker_level = Column(Enum(CircuitBreakerLevel), default=CircuitBreakerLevel.L0, comment="断路器级别")
    circuit_breaker_armed = Column(Boolean, default=True, comment="断路器是否布防")
    max_drawdown = Column(Float, comment="最大回撤")
    current_drawdown = Column(Float, comment="当前回撤")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否活跃")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_portfolio_name', 'name'),
        Index('idx_portfolio_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Portfolio(name='{self.name}', capital={self.total_capital})>"
    
    @property
    def available_capital(self) -> float:
        """可用资金"""
        return self.total_capital * self.leverage_ratio
    
    @property
    def tier_limits(self) -> Dict[str, Dict[str, float]]:
        """分层限制"""
        default_config = {
            "A": {"min": 0.12, "max": 0.15},
            "B": {"min": 0.08, "max": 0.10},
            "C": {"min": 0.03, "max": 0.05}
        }
        return self.tier_config or default_config


class Position(Base):
    """持仓表"""
    
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, comment="组合ID")
    equity_id = Column(UUID(as_uuid=True), nullable=False, comment="个股ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 持仓信息
    tier = Column(Enum(PositionTier), nullable=False, comment="持仓分层")
    shares = Column(Float, default=0, comment="持股数量")
    avg_cost = Column(Float, comment="平均成本")
    current_price = Column(Float, comment="当前价格")
    market_value = Column(Float, comment="市值")
    
    # 目标配置
    target_weight = Column(Float, comment="目标权重")
    current_weight = Column(Float, comment="当前权重")
    
    # 建仓计划
    entry_plan = Column(JSON, comment="建仓计划 {leg1: 40%, leg2: 30%, leg3: 30%}")
    entry_progress = Column(Float, default=0, comment="建仓进度")
    
    # 风险管理
    stop_loss = Column(Float, comment="止损价格")
    take_profit = Column(Float, comment="止盈价格")
    max_position_size = Column(Float, comment="最大仓位")
    
    # 融资信息
    margin_used = Column(Float, default=0, comment="使用的融资")
    margin_ratio = Column(Float, default=0, comment="融资比例")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否活跃")
    entry_date = Column(Date, comment="建仓日期")
    exit_date = Column(Date, comment="清仓日期")
    
    # 时间戳
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_position_portfolio', 'portfolio_id'),
        Index('idx_position_equity', 'equity_id'),
        Index('idx_position_ticker', 'ticker'),
        Index('idx_position_tier', 'tier'),
        Index('idx_position_active', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Position(ticker='{self.ticker}', tier='{self.tier.value}', weight={self.current_weight})>"
    
    @property
    def unrealized_pnl(self) -> Optional[float]:
        """未实现盈亏"""
        if self.avg_cost and self.current_price and self.shares:
            return (self.current_price - self.avg_cost) * self.shares
        return None
    
    @property
    def unrealized_pnl_pct(self) -> Optional[float]:
        """未实现盈亏百分比"""
        if self.avg_cost and self.current_price:
            return (self.current_price - self.avg_cost) / self.avg_cost
        return None
    
    def check_stop_loss(self) -> bool:
        """检查是否触发止损"""
        if self.stop_loss and self.current_price:
            return self.current_price <= self.stop_loss
        return False
    
    def check_take_profit(self) -> bool:
        """检查是否触发止盈"""
        if self.take_profit and self.current_price:
            return self.current_price >= self.take_profit
        return False


class PortfolioSnapshot(Base):
    """投资组合快照表"""
    
    __tablename__ = "portfolio_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, comment="组合ID")
    
    # 组合指标
    total_value = Column(Float, comment="总价值")
    cash_value = Column(Float, comment="现金价值")
    position_value = Column(Float, comment="持仓价值")
    leverage_ratio = Column(Float, comment="杠杆比例")
    
    # 收益指标
    daily_return = Column(Float, comment="日收益率")
    cumulative_return = Column(Float, comment="累计收益率")
    annualized_return = Column(Float, comment="年化收益率")
    
    # 风险指标
    volatility = Column(Float, comment="波动率")
    sharpe_ratio = Column(Float, comment="夏普比率")
    max_drawdown = Column(Float, comment="最大回撤")
    current_drawdown = Column(Float, comment="当前回撤")
    
    # 分层统计
    tier_allocation = Column(JSON, comment="分层配置 {A: 0.45, B: 0.30, C: 0.15, Cash: 0.10}")
    sector_allocation = Column(JSON, comment="行业配置")
    
    # 断路器状态
    circuit_breaker_level = Column(Enum(CircuitBreakerLevel), comment="断路器级别")
    circuit_breaker_triggered = Column(Boolean, default=False, comment="是否触发断路器")
    
    # 时间信息
    snapshot_date = Column(Date, nullable=False, comment="快照日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_portfolio_snapshot_portfolio_date', 'portfolio_id', 'snapshot_date'),
        Index('idx_portfolio_snapshot_date', 'snapshot_date'),
    )
    
    def __repr__(self):
        return f"<PortfolioSnapshot(portfolio_id='{self.portfolio_id}', date='{self.snapshot_date}', return={self.daily_return})>"


class LiquidityCheck(Base):
    """流动性检查记录表"""
    
    __tablename__ = "liquidity_checks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    equity_id = Column(UUID(as_uuid=True), nullable=False, comment="个股ID")
    ticker = Column(String(20), nullable=False, comment="股票代码")
    
    # 检查参数
    target_position = Column(Float, nullable=False, comment="目标仓位金额")
    participation_rate = Column(Float, comment="参与率")
    exit_days = Column(Integer, comment="退出天数")
    
    # 市场数据
    adv_20 = Column(Float, comment="20日平均成交额")
    current_turnover = Column(Float, comment="当前换手率")
    free_float_market_cap = Column(Float, comment="自由流通市值")
    
    # 检查结果
    adv_min_required = Column(Float, comment="最小ADV要求")
    absolute_floor_pass = Column(Boolean, comment="绝对底线是否通过")
    free_float_cap_pass = Column(Boolean, comment="自由流通占用是否通过")
    overall_pass = Column(Boolean, comment="整体是否通过")
    
    # 回显信息
    used_participation_rate = Column(Float, comment="使用的参与率")
    used_exit_days = Column(Integer, comment="使用的退出天数")
    free_float_utilization_pct = Column(Float, comment="自由流通占用百分比")
    
    # 备注和建议
    notes = Column(JSON, comment="备注列表")
    recommendations = Column(JSON, comment="建议列表")
    
    # 时间信息
    check_date = Column(Date, nullable=False, comment="检查日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_liquidity_check_equity_date', 'equity_id', 'check_date'),
        Index('idx_liquidity_check_ticker_date', 'ticker', 'check_date'),
        Index('idx_liquidity_check_date', 'check_date'),
        Index('idx_liquidity_check_pass', 'overall_pass'),
    )
    
    def __repr__(self):
        return f"<LiquidityCheck(ticker='{self.ticker}', pass={self.overall_pass}, date='{self.check_date}')>"


class RebalanceRecord(Base):
    """再平衡记录表"""
    
    __tablename__ = "rebalance_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, comment="组合ID")
    
    # 再平衡信息
    rebalance_type = Column(String(50), comment="再平衡类型 (scheduled/triggered/manual)")
    trigger_reason = Column(String(200), comment="触发原因")
    
    # 再平衡前后状态
    before_allocation = Column(JSON, comment="再平衡前配置")
    after_allocation = Column(JSON, comment="再平衡后配置")
    target_allocation = Column(JSON, comment="目标配置")
    
    # 交易计划
    trades_planned = Column(JSON, comment="计划交易列表")
    trades_executed = Column(JSON, comment="已执行交易列表")
    
    # 执行状态
    status = Column(String(20), default="planned", comment="状态 (planned/executing/completed/failed)")
    execution_progress = Column(Float, default=0, comment="执行进度")
    
    # 成本和影响
    estimated_cost = Column(Float, comment="预估成本")
    actual_cost = Column(Float, comment="实际成本")
    market_impact = Column(Float, comment="市场冲击")
    
    # 时间信息
    planned_date = Column(Date, comment="计划日期")
    execution_date = Column(Date, comment="执行日期")
    completed_date = Column(Date, comment="完成日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_rebalance_portfolio_date', 'portfolio_id', 'planned_date'),
        Index('idx_rebalance_status', 'status'),
        Index('idx_rebalance_type', 'rebalance_type'),
    )
    
    def __repr__(self):
        return f"<RebalanceRecord(portfolio_id='{self.portfolio_id}', type='{self.rebalance_type}', status='{self.status}')>"


class RiskMetrics(Base):
    """风险指标表"""
    
    __tablename__ = "risk_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), nullable=False, comment="组合ID")
    
    # VaR指标
    var_1d_95 = Column(Float, comment="1日95% VaR")
    var_1d_99 = Column(Float, comment="1日99% VaR")
    cvar_1d_95 = Column(Float, comment="1日95% CVaR")
    
    # 风险分解
    systematic_risk = Column(Float, comment="系统性风险")
    idiosyncratic_risk = Column(Float, comment="特异性风险")
    concentration_risk = Column(Float, comment="集中度风险")
    
    # 行业暴露
    sector_exposure = Column(JSON, comment="行业暴露")
    style_exposure = Column(JSON, comment="风格暴露")
    
    # 相关性指标
    portfolio_beta = Column(Float, comment="组合Beta")
    correlation_with_market = Column(Float, comment="与市场相关性")
    
    # 流动性风险
    liquidity_score = Column(Float, comment="流动性评分")
    days_to_liquidate = Column(Float, comment="清仓所需天数")
    
    # 时间信息
    calculation_date = Column(Date, nullable=False, comment="计算日期")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_risk_metrics_portfolio_date', 'portfolio_id', 'calculation_date'),
        Index('idx_risk_metrics_date', 'calculation_date'),
    )
    
    def __repr__(self):
        return f"<RiskMetrics(portfolio_id='{self.portfolio_id}', date='{self.calculation_date}', var={self.var_1d_95})>"
