"""
InvestIQ Platform - 告警数据模型
告警、事件和通知模型
"""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Date, Text, JSON, Boolean, Index, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
import enum

from backend.app.core.database import Base


class AlertType(enum.Enum):
    """告警类型"""
    EVENT = "event"          # 事件型告警
    KPI = "kpi"             # KPI偏离告警
    TREND = "trend"         # 趋势型告警
    DRAWDOWN = "drawdown"   # 回撤型告警


class AlertSeverity(enum.Enum):
    """告警严重程度"""
    P1 = "P1"  # 严重 - 需要立即处理
    P2 = "P2"  # 重要 - 需要及时处理
    P3 = "P3"  # 一般 - 可以延后处理


class AlertStatus(enum.Enum):
    """告警状态"""
    NEW = "new"                    # 新建
    ACKNOWLEDGED = "acknowledged"  # 已确认
    IN_PROGRESS = "in_progress"   # 处理中
    RESOLVED = "resolved"         # 已解决
    CLOSED = "closed"             # 已关闭


class Alert(Base):
    """告警表"""
    
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 告警基本信息
    alert_type = Column(Enum(AlertType), nullable=False, comment="告警类型")
    severity = Column(Enum(AlertSeverity), nullable=False, comment="严重程度")
    status = Column(Enum(AlertStatus), default=AlertStatus.NEW, comment="告警状态")
    
    # 关联实体
    entity_type = Column(String(50), comment="实体类型 (industry/equity/portfolio)")
    entity_id = Column(UUID(as_uuid=True), comment="实体ID")
    ticker = Column(String(20), comment="股票代码 (如果适用)")
    
    # 告警内容
    title = Column(String(200), nullable=False, comment="告警标题")
    message = Column(Text, comment="告警消息")
    rule_name = Column(String(100), comment="触发规则名称")
    
    # 告警数据
    current_value = Column(Float, comment="当前值")
    threshold_value = Column(Float, comment="阈值")
    previous_value = Column(Float, comment="前值")
    
    # 告警详情
    details = Column(JSON, comment="详细信息")
    context = Column(JSON, comment="上下文信息")
    
    # 节流控制
    throttle_key = Column(String(200), comment="节流键")
    hits_count = Column(Integer, default=1, comment="命中次数")
    last_hit_at = Column(DateTime(timezone=True), comment="最后命中时间")
    next_allowed_at = Column(DateTime(timezone=True), comment="下次允许时间")
    
    # 锁存状态 (用于回撤型告警)
    is_latched = Column(Boolean, default=False, comment="是否锁存")
    latch_condition = Column(JSON, comment="锁存条件")
    recovery_condition = Column(JSON, comment="恢复条件")
    
    # 处理信息
    assignee = Column(String(100), comment="处理人")
    acknowledged_at = Column(DateTime(timezone=True), comment="确认时间")
    resolved_at = Column(DateTime(timezone=True), comment="解决时间")
    resolution_note = Column(Text, comment="解决备注")
    
    # 时间信息
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), comment="触发时间")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_alert_type_severity', 'alert_type', 'severity'),
        Index('idx_alert_status', 'status'),
        Index('idx_alert_entity', 'entity_type', 'entity_id'),
        Index('idx_alert_ticker', 'ticker'),
        Index('idx_alert_triggered', 'triggered_at'),
        Index('idx_alert_throttle', 'throttle_key'),
        Index('idx_alert_assignee', 'assignee'),
    )
    
    def __repr__(self):
        return f"<Alert(type='{self.alert_type.value}', severity='{self.severity.value}', title='{self.title}')>"
    
    def can_trigger_now(self) -> bool:
        """检查是否可以触发"""
        if self.next_allowed_at is None:
            return True
        return datetime.utcnow() >= self.next_allowed_at
    
    def should_escalate(self, p3_to_p2_hits: int = 3, p2_to_p1_hits: int = 2) -> Optional[AlertSeverity]:
        """检查是否应该升级"""
        if self.severity == AlertSeverity.P3 and self.hits_count >= p3_to_p2_hits:
            return AlertSeverity.P2
        elif self.severity == AlertSeverity.P2 and self.hits_count >= p2_to_p1_hits:
            return AlertSeverity.P1
        return None
    
    def is_overdue(self, sla_minutes: Dict[str, int] = None) -> bool:
        """检查是否超时"""
        if self.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
            return False
        
        sla_minutes = sla_minutes or {"P1": 30, "P2": 120, "P3": 1440}
        sla = sla_minutes.get(self.severity.value, 1440)
        
        deadline = self.triggered_at + timedelta(minutes=sla)
        return datetime.utcnow() > deadline


class AlertRule(Base):
    """告警规则表"""
    
    __tablename__ = "alert_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, comment="规则名称")
    description = Column(Text, comment="规则描述")
    
    # 规则配置
    alert_type = Column(Enum(AlertType), nullable=False, comment="告警类型")
    entity_type = Column(String(50), comment="适用实体类型")
    
    # 触发条件
    condition = Column(JSON, nullable=False, comment="触发条件")
    threshold_config = Column(JSON, comment="阈值配置")
    
    # 告警配置
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.P3, comment="默认严重程度")
    message_template = Column(Text, comment="消息模板")
    
    # 节流配置
    throttle_config = Column(JSON, comment="节流配置")
    
    # 状态信息
    is_enabled = Column(Boolean, default=True, comment="是否启用")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_alert_rule_name', 'name'),
        Index('idx_alert_rule_type', 'alert_type'),
        Index('idx_alert_rule_enabled', 'is_enabled'),
    )
    
    def __repr__(self):
        return f"<AlertRule(name='{self.name}', type='{self.alert_type.value}', enabled={self.is_enabled})>"


class EventCalendar(Base):
    """事件日历表"""
    
    __tablename__ = "event_calendar"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 事件基本信息
    event_type = Column(String(50), nullable=False, comment="事件类型")
    title = Column(String(200), nullable=False, comment="事件标题")
    description = Column(Text, comment="事件描述")
    
    # 关联实体
    entity_type = Column(String(50), comment="关联实体类型")
    entity_id = Column(UUID(as_uuid=True), comment="关联实体ID")
    ticker = Column(String(20), comment="股票代码")
    
    # 时间信息
    event_date = Column(Date, nullable=False, comment="事件日期")
    event_time = Column(DateTime(timezone=True), comment="具体时间")
    
    # 提醒配置
    lead_time_days = Column(Integer, comment="提前提醒天数")
    reminder_sent = Column(Boolean, default=False, comment="是否已发送提醒")
    
    # 事件数据
    event_data = Column(JSON, comment="事件相关数据")
    source = Column(String(100), comment="事件来源")
    source_url = Column(String(500), comment="来源链接")
    
    # 状态信息
    is_active = Column(Boolean, default=True, comment="是否活跃")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_event_calendar_date', 'event_date'),
        Index('idx_event_calendar_type', 'event_type'),
        Index('idx_event_calendar_entity', 'entity_type', 'entity_id'),
        Index('idx_event_calendar_ticker', 'ticker'),
        Index('idx_event_calendar_reminder', 'reminder_sent'),
    )
    
    def __repr__(self):
        return f"<EventCalendar(title='{self.title}', date='{self.event_date}', type='{self.event_type}')>"
    
    def should_send_reminder(self) -> bool:
        """检查是否应该发送提醒"""
        if self.reminder_sent or not self.lead_time_days:
            return False
        
        reminder_date = self.event_date - timedelta(days=self.lead_time_days)
        return date.today() >= reminder_date


class NotificationLog(Base):
    """通知日志表"""
    
    __tablename__ = "notification_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 通知信息
    notification_type = Column(String(50), nullable=False, comment="通知类型")
    channel = Column(String(50), comment="通知渠道 (email/sms/webhook)")
    recipient = Column(String(200), comment="接收者")
    
    # 关联告警或事件
    alert_id = Column(UUID(as_uuid=True), comment="关联告警ID")
    event_id = Column(UUID(as_uuid=True), comment="关联事件ID")
    
    # 通知内容
    subject = Column(String(200), comment="主题")
    content = Column(Text, comment="内容")
    
    # 发送状态
    status = Column(String(20), default="pending", comment="发送状态")
    sent_at = Column(DateTime(timezone=True), comment="发送时间")
    delivered_at = Column(DateTime(timezone=True), comment="送达时间")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    retry_count = Column(Integer, default=0, comment="重试次数")
    
    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_notification_log_alert', 'alert_id'),
        Index('idx_notification_log_event', 'event_id'),
        Index('idx_notification_log_status', 'status'),
        Index('idx_notification_log_sent', 'sent_at'),
        Index('idx_notification_log_recipient', 'recipient'),
    )
    
    def __repr__(self):
        return f"<NotificationLog(type='{self.notification_type}', status='{self.status}', recipient='{self.recipient}')>"


class AlertMetrics(Base):
    """告警指标统计表"""
    
    __tablename__ = "alert_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 统计维度
    metric_date = Column(Date, nullable=False, comment="统计日期")
    alert_type = Column(Enum(AlertType), comment="告警类型")
    severity = Column(Enum(AlertSeverity), comment="严重程度")
    
    # 统计指标
    total_alerts = Column(Integer, default=0, comment="总告警数")
    new_alerts = Column(Integer, default=0, comment="新增告警数")
    resolved_alerts = Column(Integer, default=0, comment="解决告警数")
    overdue_alerts = Column(Integer, default=0, comment="超时告警数")
    
    # 处理指标
    avg_resolution_time_minutes = Column(Float, comment="平均解决时间(分钟)")
    avg_acknowledgment_time_minutes = Column(Float, comment="平均确认时间(分钟)")
    
    # 节流指标
    throttled_alerts = Column(Integer, default=0, comment="被节流的告警数")
    throttle_hit_ratio = Column(Float, comment="节流命中率")
    
    # 升级指标
    escalated_alerts = Column(Integer, default=0, comment="升级告警数")
    escalation_ratio = Column(Float, comment="升级比例")
    
    # 时间信息
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 索引
    __table_args__ = (
        Index('idx_alert_metrics_date', 'metric_date'),
        Index('idx_alert_metrics_type_severity', 'alert_type', 'severity'),
    )
    
    def __repr__(self):
        return f"<AlertMetrics(date='{self.metric_date}', type='{self.alert_type}', total={self.total_alerts})>"
