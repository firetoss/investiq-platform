"""
InvestIQ Platform - 数据模型包
导入所有数据模型，确保SQLAlchemy能够发现和创建表
"""

# 导入基础类
from backend.app.core.database import Base

# 导入行业相关模型
from backend.app.models.industry import (
    Industry,
    IndustryScoreSnapshot,
    IndustryMetrics,
)

# 导入个股相关模型
from backend.app.models.equity import (
    Equity,
    EquityScoreSnapshot,
    ValuationPercentileSnapshot,
    TechnicalIndicator,
)

# 导入投资组合相关模型
from backend.app.models.portfolio import (
    Portfolio,
    Position,
    PortfolioSnapshot,
    LiquidityCheck,
    RebalanceRecord,
    RiskMetrics,
    PositionTier,
    CircuitBreakerLevel,
)

# 导入告警相关模型
from backend.app.models.alert import (
    Alert,
    AlertRule,
    EventCalendar,
    NotificationLog,
    AlertMetrics,
    AlertType,
    AlertSeverity,
    AlertStatus,
)

# 导入证据和审计相关模型
from backend.app.models.evidence import (
    EvidenceItem,
    DecisionLog,
    User,
    AuditSummary,
    DataLineage,
    EvidenceType,
    EvidenceSource,
)

# 导出所有模型类
__all__ = [
    # 基础
    "Base",
    
    # 行业模型
    "Industry",
    "IndustryScoreSnapshot", 
    "IndustryMetrics",
    
    # 个股模型
    "Equity",
    "EquityScoreSnapshot",
    "ValuationPercentileSnapshot",
    "TechnicalIndicator",
    
    # 投资组合模型
    "Portfolio",
    "Position",
    "PortfolioSnapshot",
    "LiquidityCheck",
    "RebalanceRecord",
    "RiskMetrics",
    "PositionTier",
    "CircuitBreakerLevel",
    
    # 告警模型
    "Alert",
    "AlertRule",
    "EventCalendar",
    "NotificationLog",
    "AlertMetrics",
    "AlertType",
    "AlertSeverity",
    "AlertStatus",
    
    # 证据和审计模型
    "EvidenceItem",
    "DecisionLog",
    "User",
    "AuditSummary",
    "DataLineage",
    "EvidenceType",
    "EvidenceSource",
]

# 模型注册表 - 用于动态访问
MODEL_REGISTRY = {
    # 行业
    "industry": Industry,
    "industry_score_snapshot": IndustryScoreSnapshot,
    "industry_metrics": IndustryMetrics,
    
    # 个股
    "equity": Equity,
    "equity_score_snapshot": EquityScoreSnapshot,
    "valuation_percentile_snapshot": ValuationPercentileSnapshot,
    "technical_indicator": TechnicalIndicator,
    
    # 投资组合
    "portfolio": Portfolio,
    "position": Position,
    "portfolio_snapshot": PortfolioSnapshot,
    "liquidity_check": LiquidityCheck,
    "rebalance_record": RebalanceRecord,
    "risk_metrics": RiskMetrics,
    
    # 告警
    "alert": Alert,
    "alert_rule": AlertRule,
    "event_calendar": EventCalendar,
    "notification_log": NotificationLog,
    "alert_metrics": AlertMetrics,
    
    # 证据和审计
    "evidence_item": EvidenceItem,
    "decision_log": DecisionLog,
    "user": User,
    "audit_summary": AuditSummary,
    "data_lineage": DataLineage,
}

def get_model_by_name(model_name: str):
    """根据名称获取模型类"""
    return MODEL_REGISTRY.get(model_name.lower())

def get_all_models():
    """获取所有模型类"""
    return list(MODEL_REGISTRY.values())

def get_model_names():
    """获取所有模型名称"""
    return list(MODEL_REGISTRY.keys())
