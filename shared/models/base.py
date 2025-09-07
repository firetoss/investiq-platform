"""
InvestIQ Platform - 基础数据模型
SQLAlchemy 基础类和通用字段定义
"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer, BigInteger, DECIMAL, Date, UUID
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional
import uuid


# SQLAlchemy基础类
Base = declarative_base()


class BaseModel(Base):
    """所有数据模型的基础类"""
    __abstract__ = True
    
    # 主键
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 时间戳字段
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # 创建者
    created_by = Column(String(100))
    
    def to_dict(self) -> dict:
        """转换为字典"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            elif isinstance(value, uuid.UUID):
                result[column.name] = str(value)
            else:
                result[column.name] = value
        return result
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UserMixin:
    """用户信息混入类"""
    created_by = Column(String(100))
    updated_by = Column(String(100))


class SoftDeleteMixin:
    """软删除混入类"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True))
    deleted_by = Column(String(100))


class VersionMixin:
    """版本控制混入类"""
    version = Column(Integer, default=1, nullable=False)
    is_current = Column(Boolean, default=True, nullable=False)
    superseded_by = Column(PG_UUID(as_uuid=True))


class AuditMixin:
    """审计信息混入类"""
    request_id = Column(PG_UUID(as_uuid=True))
    user_agent = Column(String(500))
    ip_address = Column(String(45))  # IPv6 compatible


# 通用枚举类型
class EntityType:
    """实体类型枚举"""
    INDUSTRY = "industry"
    EQUITY = "equity"
    PORTFOLIO = "portfolio"
    EVIDENCE = "evidence"


class AlertType:
    """告警类型枚举"""
    EVENT = "event"
    KPI = "kpi"
    TREND = "trend"
    DRAWDOWN = "drawdown"


class AlertSeverity:
    """告警严重度枚举"""
    P1 = "P1"  # 最高优先级
    P2 = "P2"  # 中等优先级
    P3 = "P3"  # 低优先级


class AlertStatus:
    """告警状态枚举"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class PositionTier:
    """持仓层级枚举"""
    A = "A"  # 核心持仓
    B = "B"  # 重要持仓
    C = "C"  # 战术持仓


class PositionStatus:
    """持仓状态枚举"""
    DRAFT = "draft"
    REVIEW = "review"
    IN_POOL = "in_pool"
    BUILD_PLAN = "build_plan"
    ACTIVE = "active"
    WATCH = "watch"
    EXIT = "exit"


class EvidenceType:
    """证据类型枚举"""
    POLICY = "policy"
    ORDER = "order"
    TENDER = "tender"
    ACCEPTANCE = "acceptance"
    FINANCIAL = "financial"
    MARKET = "market"
    NEWS = "news"
    RESEARCH = "research"


class Currency:
    """货币枚举"""
    CNY = "CNY"  # 人民币
    HKD = "HKD"  # 港币
    USD = "USD"  # 美元


class Exchange:
    """交易所枚举"""
    SSE = "SSE"   # 上海证券交易所
    SZSE = "SZSE" # 深圳证券交易所
    HKEX = "HKEX" # 香港交易所
    

class ScoreMethod:
    """评分方法枚举"""
    STANDARD = "standard"
    WEIGHTED = "weighted"
    ADJUSTED = "adjusted"


class ValuationMethod:
    """估值方法枚举"""
    PE = "PE"
    EV_EBITDA = "EV_EBITDA"
    EV_SALES = "EV_SALES"
    PB = "PB"
    PEG = "PEG"
    Z_SCORE = "Z_SCORE"


# 数据验证函数
def validate_score(score: Optional[float]) -> bool:
    """验证评分范围 (0-100)"""
    if score is None:
        return True
    return 0 <= score <= 100


def validate_percentage(value: Optional[float]) -> bool:
    """验证百分比范围 (0-1)"""
    if value is None:
        return True
    return 0 <= value <= 1


def validate_ticker(ticker: str) -> bool:
    """验证股票代码格式"""
    if not ticker:
        return False
    
    # A股格式：6位数字.SH 或 6位数字.SZ
    if ticker.endswith('.SH') or ticker.endswith('.SZ'):
        code = ticker[:-3]
        return len(code) == 6 and code.isdigit()
    
    # H股格式：4位数字.HK
    if ticker.endswith('.HK'):
        code = ticker[:-3]
        return len(code) == 4 and code.isdigit()
    
    return False


def validate_currency(currency: str) -> bool:
    """验证货币代码"""
    return currency in [Currency.CNY, Currency.HKD, Currency.USD]


def validate_exchange(exchange: str) -> bool:
    """验证交易所代码"""
    return exchange in [Exchange.SSE, Exchange.SZSE, Exchange.HKEX]


# 数据库工具函数
def get_current_date():
    """获取当前日期（用于 as_of 字段）"""
    return datetime.now().date()


def generate_snapshot_id(entity_type: str, entity_id: str, timestamp: Optional[datetime] = None) -> str:
    """生成快照ID"""
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{entity_type}_{entity_id}_{timestamp_str}"


# 常用查询过滤器
class QueryFilters:
    """常用查询过滤器"""
    
    @staticmethod
    def active_only(query, model):
        """只查询激活状态的记录"""
        if hasattr(model, 'is_active'):
            return query.filter(model.is_active == True)
        return query
    
    @staticmethod
    def not_deleted(query, model):
        """只查询未删除的记录"""
        if hasattr(model, 'is_deleted'):
            return query.filter(model.is_deleted == False)
        return query
    
    @staticmethod
    def current_version(query, model):
        """只查询当前版本的记录"""
        if hasattr(model, 'is_current'):
            return query.filter(model.is_current == True)
        return query
    
    @staticmethod
    def by_date_range(query, model, start_date, end_date, date_field='created_at'):
        """按日期范围过滤"""
        date_column = getattr(model, date_field, None)
        if date_column is not None:
            if start_date:
                query = query.filter(date_column >= start_date)
            if end_date:
                query = query.filter(date_column <= end_date)
        return query