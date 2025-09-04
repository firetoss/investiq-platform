"""
InvestIQ Platform - 告警系统API端点
告警管理、事件日历和通知服务的REST API
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.core.config import settings
from backend.app.models.alert import (
    Alert, AlertRule, EventCalendar, NotificationLog, AlertMetrics,
    AlertType, AlertSeverity, AlertStatus
)
from backend.app.services.alert_service import alert_service
from backend.app.api import API_DESCRIPTIONS

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class AlertCreateRequest(BaseModel):
    """创建告警请求模型"""
    alert_type: str = Field(..., description="告警类型")
    entity_type: Optional[str] = Field(None, description="实体类型")
    entity_id: Optional[str] = Field(None, description="实体ID")
    ticker: Optional[str] = Field(None, description="股票代码")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    rule_name: Optional[str] = Field(None, description="规则名称")
    severity: str = Field("P3", description="严重程度")
    current_value: Optional[float] = Field(None, description="当前值")
    threshold_value: Optional[float] = Field(None, description="阈值")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")

    @validator("alert_type")
    def validate_alert_type(cls, v):
        valid_types = ["event", "kpi", "trend", "drawdown"]
        if v not in valid_types:
            raise ValueError(f"告警类型必须为: {valid_types}")
        return v

    @validator("severity")
    def validate_severity(cls, v):
        valid_severities = ["P1", "P2", "P3"]
        if v not in valid_severities:
            raise ValueError(f"严重程度必须为: {valid_severities}")
        return v


class AlertUpdateRequest(BaseModel):
    """更新告警请求模型"""
    status: Optional[str] = Field(None, description="告警状态")
    assignee: Optional[str] = Field(None, description="处理人")
    resolution_note: Optional[str] = Field(None, description="解决备注")

    @validator("status")
    def validate_status(cls, v):
        if v is not None:
            valid_statuses = ["new", "acknowledged", "in_progress", "resolved", "closed"]
            if v not in valid_statuses:
                raise ValueError(f"状态必须为: {valid_statuses}")
        return v


class AlertRuleCreateRequest(BaseModel):
    """创建告警规则请求模型"""
    name: str = Field(..., description="规则名称")
    description: Optional[str] = Field(None, description="规则描述")
    alert_type: str = Field(..., description="告警类型")
    entity_type: Optional[str] = Field(None, description="实体类型")
    condition: Dict[str, Any] = Field(..., description="触发条件")
    threshold_config: Optional[Dict[str, Any]] = Field(None, description="阈值配置")
    severity: str = Field("P3", description="默认严重程度")
    message_template: Optional[str] = Field(None, description="消息模板")
    throttle_config: Optional[Dict[str, Any]] = Field(None, description="节流配置")

    @validator("alert_type")
    def validate_alert_type(cls, v):
        valid_types = ["event", "kpi", "trend", "drawdown"]
        if v not in valid_types:
            raise ValueError(f"告警类型必须为: {valid_types}")
        return v


class EventCreateRequest(BaseModel):
    """创建事件请求模型"""
    event_type: str = Field(..., description="事件类型")
    title: str = Field(..., description="事件标题")
    description: Optional[str] = Field(None, description="事件描述")
    entity_type: Optional[str] = Field(None, description="关联实体类型")
    entity_id: Optional[str] = Field(None, description="关联实体ID")
    ticker: Optional[str] = Field(None, description="股票代码")
    event_date: date = Field(..., description="事件日期")
    event_time: Optional[datetime] = Field(None, description="具体时间")
    lead_time_days: Optional[int] = Field(None, description="提前提醒天数")
    event_data: Optional[Dict[str, Any]] = Field(None, description="事件数据")
    source: Optional[str] = Field(None, description="事件来源")
    source_url: Optional[str] = Field(None, description="来源链接")


class AlertResponse(BaseModel):
    """告警响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.get(
    "/list",
    response_model=AlertResponse,
    summary="获取告警列表",
    description="获取告警列表，支持分页和过滤"
)
async def list_alerts(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    alert_type: Optional[str] = Query(None, description="告警类型过滤"),
    severity: Optional[str] = Query(None, description="严重程度过滤"),
    status: Optional[str] = Query(None, description="状态过滤"),
    assignee: Optional[str] = Query(None, description="处理人过滤"),
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    ticker: Optional[str] = Query(None, description="股票代码过滤"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期")
):
    """获取告警列表"""
    try:
        result = await alert_service.list_alerts(
            page=page,
            page_size=page_size,
            alert_type=alert_type,
            severity=severity,
            status=status,
            assignee=assignee,
            entity_type=entity_type,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"List alerts failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/create",
    response_model=AlertResponse,
    summary="创建告警",
    description="创建新的告警记录"
)
async def create_alert(
    request: AlertCreateRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """创建告警"""
    try:
        result = await alert_service.create_alert(
            alert_type=request.alert_type,
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            ticker=request.ticker,
            title=request.title,
            message=request.message,
            rule_name=request.rule_name,
            severity=request.severity,
            current_value=request.current_value,
            threshold_value=request.threshold_value,
            details=request.details,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Create alert failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.put(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="更新告警",
    description="更新告警状态和处理信息"
)
async def update_alert(
    alert_id: str,
    request: AlertUpdateRequest,
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """更新告警"""
    try:
        result = await alert_service.update_alert(
            alert_id=alert_id,
            status=request.status,
            assignee=request.assignee,
            resolution_note=request.resolution_note,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Update alert failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/{alert_id}/acknowledge",
    response_model=AlertResponse,
    summary="确认告警",
    description="确认告警并可选择分配处理人"
)
async def acknowledge_alert(
    alert_id: str,
    db: AsyncSession = Depends(get_db),
    assignee: Optional[str] = Query(None, description="分配处理人"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """确认告警"""
    try:
        result = await alert_service.acknowledge_alert(
            alert_id=alert_id,
            assignee=assignee,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Acknowledge alert failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/batch-acknowledge",
    response_model=AlertResponse,
    summary="批量确认告警",
    description="批量确认多个告警"
)
async def batch_acknowledge_alerts(
    alert_ids: List[str] = Field(..., description="告警ID列表"),
    assignee: Optional[str] = Field(None, description="分配处理人"),
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """批量确认告警"""
    try:
        result = await alert_service.batch_acknowledge_alerts(
            alert_ids=alert_ids,
            assignee=assignee,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch acknowledge alerts failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/rules",
    response_model=AlertResponse,
    summary="获取告警规则",
    description="获取告警规则列表"
)
async def list_alert_rules(
    db: AsyncSession = Depends(get_db),
    enabled_only: bool = Query(True, description="仅显示启用的规则"),
    alert_type: Optional[str] = Query(None, description="告警类型过滤")
):
    """获取告警规则"""
    try:
        result = await alert_service.list_alert_rules(
            enabled_only=enabled_only,
            alert_type=alert_type,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"List alert rules failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/rules",
    response_model=AlertResponse,
    summary="创建告警规则",
    description="创建新的告警规则"
)
async def create_alert_rule(
    request: AlertRuleCreateRequest,
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """创建告警规则"""
    try:
        result = await alert_service.create_alert_rule(
            name=request.name,
            description=request.description,
            alert_type=request.alert_type,
            entity_type=request.entity_type,
            condition=request.condition,
            threshold_config=request.threshold_config,
            severity=request.severity,
            message_template=request.message_template,
            throttle_config=request.throttle_config,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Create alert rule failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/events",
    response_model=AlertResponse,
    summary="获取事件日历",
    description="获取事件日历列表"
)
async def list_events(
    db: AsyncSession = Depends(get_db),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    event_type: Optional[str] = Query(None, description="事件类型过滤"),
    ticker: Optional[str] = Query(None, description="股票代码过滤"),
    pending_reminder_only: bool = Query(False, description="仅显示待提醒事件")
):
    """获取事件日历"""
    try:
        result = await alert_service.list_events(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            ticker=ticker,
            pending_reminder_only=pending_reminder_only,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"List events failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/events",
    response_model=AlertResponse,
    summary="创建事件",
    description="创建新的事件日历记录"
)
async def create_event(
    request: EventCreateRequest,
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """创建事件"""
    try:
        result = await alert_service.create_event(
            event_type=request.event_type,
            title=request.title,
            description=request.description,
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            ticker=request.ticker,
            event_date=request.event_date,
            event_time=request.event_time,
            lead_time_days=request.lead_time_days,
            event_data=request.event_data,
            source=request.source,
            source_url=request.source_url,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Create event failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/dashboard",
    response_model=AlertResponse,
    summary="告警仪表板",
    description="获取告警系统的仪表板数据"
)
async def get_alert_dashboard(
    db: AsyncSession = Depends(get_db),
    days: int = Query(7, ge=1, le=30, description="统计天数")
):
    """告警仪表板"""
    try:
        result = await alert_service.get_alert_dashboard(days=days, db=db)
        
        return AlertResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get alert dashboard failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/metrics",
    response_model=AlertResponse,
    summary="告警指标统计",
    description="获取告警系统的统计指标"
)
async def get_alert_metrics(
    db: AsyncSession = Depends(get_db),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    group_by: str = Query("date", description="分组方式 (date/type/severity)")
):
    """告警指标统计"""
    try:
        result = await alert_service.get_alert_metrics(
            start_date=start_date,
            end_date=end_date,
            group_by=group_by,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get alert metrics failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/test/{rule_name}",
    response_model=AlertResponse,
    summary="测试告警规则",
    description="测试告警规则是否正常工作"
)
async def test_alert_rule(
    rule_name: str,
    test_data: Dict[str, Any] = Field(..., description="测试数据"),
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """测试告警规则"""
    try:
        result = await alert_service.test_alert_rule(
            rule_name=rule_name,
            test_data=test_data,
            db=db
        )
        
        return AlertResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Test alert rule failed: {e}", exc_info=True)
        return AlertResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/config/default",
    summary="获取默认配置",
    description="获取告警系统的默认配置"
)
async def get_default_config():
    """获取默认配置"""
    return {
        "alert_types": {
            "event": {"name": "事件型告警", "description": "政策发布、财报公告等事件驱动"},
            "kpi": {"name": "KPI偏离告警", "description": "关键指标偏离预设阈值"},
            "trend": {"name": "趋势型告警", "description": "技术指标、均线交叉等趋势变化"},
            "drawdown": {"name": "回撤型告警", "description": "组合或个股回撤超过阈值"}
        },
        "severities": {
            "P1": {"name": "严重", "sla_minutes": 30, "description": "需要立即处理"},
            "P2": {"name": "重要", "sla_minutes": 120, "description": "需要及时处理"},
            "P3": {"name": "一般", "sla_minutes": 1440, "description": "可以延后处理"}
        },
        "statuses": [
            {"value": "new", "name": "新建"},
            {"value": "acknowledged", "name": "已确认"},
            {"value": "in_progress", "name": "处理中"},
            {"value": "resolved", "name": "已解决"},
            {"value": "closed", "name": "已关闭"}
        ],
        "throttle_config": {
            "event": {"min_interval_minutes": settings.ALERT_THROTTLE_EVENT_MINUTES},
            "kpi": {"min_interval_minutes": settings.ALERT_THROTTLE_KPI_MINUTES},
            "trend": {"min_interval_days": settings.ALERT_THROTTLE_TREND_DAYS},
            "drawdown": {"latch_enabled": True}
        },
        "escalation": {
            "p3_to_p2_hits": settings.ALERT_ESCALATION_P3_TO_P2_HITS,
            "p2_to_p1_hits": settings.ALERT_ESCALATION_P2_TO_P1_HITS
        },
        "notification_channels": ["email", "sms", "webhook", "in_app"],
        "event_types": [
            "policy_release",
            "earnings_announcement", 
            "dividend_announcement",
            "insider_trading",
            "analyst_rating_change",
            "major_contract",
            "regulatory_filing",
            "market_event"
        ]
    }


@router.get(
    "/health",
    summary="告警系统健康检查",
    description="检查告警系统状态"
)
async def alert_health_check(
    db: AsyncSession = Depends(get_db)
):
    """告警系统健康检查"""
    try:
        # 检查数据库连接
        await db.execute("SELECT 1")
        
        # 检查告警服务状态
        health_status = await alert_service.health_check(db)
        
        return {
            "status": "healthy",
            "service_status": health_status,
            "features": {
                "alert_creation": True,
                "rule_management": True,
                "event_calendar": True,
                "notification": True,
                "throttling": True,
                "escalation": True,
                "metrics": True
            },
            "config": {
                "throttle_event_minutes": settings.ALERT_THROTTLE_EVENT_MINUTES,
                "throttle_kpi_minutes": settings.ALERT_THROTTLE_KPI_MINUTES,
                "throttle_trend_days": settings.ALERT_THROTTLE_TREND_DAYS,
                "escalation_p3_to_p2": settings.ALERT_ESCALATION_P3_TO_P2_HITS,
                "escalation_p2_to_p1": settings.ALERT_ESCALATION_P2_TO_P1_HITS
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alert health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"告警系统健康检查失败: {e}"
        )