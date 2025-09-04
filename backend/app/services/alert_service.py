"""
InvestIQ Platform - 告警服务
实现告警管理、事件日历和通知系统的核心业务逻辑
符合PRD要求的节流优先级、趋势确认、回撤锁存机制
"""

import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func, case, update, text
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.exceptions import ValidationException, AlertException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.alert import (
    Alert, AlertRule, EventCalendar, NotificationLog, AlertMetrics,
    AlertType, AlertSeverity, AlertStatus
)

logger = get_logger(__name__)


class ThrottleController:
    """告警节流控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.by_type_config = config.get("by_type", {})
        self.global_fallback = config.get("min_interval_minutes", 120)
    
    def get_throttle_config(self, alert_type: str) -> Dict[str, Any]:
        """
        获取节流配置 (PRD要求: by_type优先于全局回退)
        
        配置项 alerts.throttle.by_type 的每类配置优先于全局 min_interval_minutes。
        当两者同时存在时，以 by_type 为准；仅当 by_type 缺失时回退到全局值。
        """
        # 优先使用by_type配置
        type_config = self.by_type_config.get(alert_type)
        if type_config:
            return type_config
        
        # 回退到全局配置
        return {"min_interval_minutes": self.global_fallback}
    
    async def should_throttle(
        self, 
        alert_type: str,
        entity_type: Optional[str],
        entity_id: Optional[str],
        db_session: AsyncSession
    ) -> Tuple[bool, Optional[datetime]]:
        """
        检查是否应该节流
        
        Returns:
            (should_throttle, next_allowed_time)
        """
        throttle_config = self.get_throttle_config(alert_type)
        
        # 获取最近的同类告警
        stmt = select(Alert).where(
            and_(
                Alert.alert_type == alert_type,
                Alert.entity_type == entity_type,
                Alert.entity_id == entity_id,
                Alert.status.in_([AlertStatus.NEW, AlertStatus.ACKNOWLEDGED, AlertStatus.IN_PROGRESS])
            )
        ).order_by(desc(Alert.triggered_at)).limit(1)
        
        result = await db_session.execute(stmt)
        last_alert = result.scalar_one_or_none()
        
        if not last_alert:
            return False, None  # 没有历史告警，不节流
        
        # 计算节流时间
        now = datetime.utcnow()
        
        if alert_type == "trend":
            # 趋势告警：特殊处理，仅"收盘后"确认双均线交叉，最小间隔5个交易日
            if "confirm_on_cross_rule" in throttle_config:
                min_days = throttle_config.get("min_interval_days", 5)
                next_allowed = last_alert.triggered_at + timedelta(days=min_days)
                return now < next_allowed, next_allowed
        
        elif alert_type == "drawdown":
            # 回撤告警：latch锁存机制
            if throttle_config.get("latch", False):
                # 检查是否已锁存且未恢复
                if hasattr(last_alert, 'is_latched') and last_alert.is_latched:
                    return True, None  # 锁存状态，直到满足恢复条件
        
        # 标准时间节流
        min_interval_minutes = throttle_config.get("min_interval_minutes", self.global_fallback)
        next_allowed = last_alert.triggered_at + timedelta(minutes=min_interval_minutes)
        
        return now < next_allowed, next_allowed


class DrawdownLatchController:
    """回撤锁存控制器"""
    
    def __init__(self):
        self.latch_levels = {
            -0.10: "L1",  # 一级预警
            -0.20: "L2",  # 二级断路  
            -0.30: "L3"   # 三级断路
        }
    
    async def check_latch_status(
        self,
        portfolio_id: str,
        current_drawdown: float,
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """
        检查回撤锁存状态
        
        PRD规定: 
        - latch锁存；恢复条件为"最大回撤回阈值以内并连续3个交易日满足，或创出本轮新高"
        """
        # 获取当前锁存的回撤告警
        stmt = select(Alert).where(
            and_(
                Alert.alert_type == "drawdown",
                Alert.entity_type == "portfolio",
                Alert.entity_id == portfolio_id,
                Alert.status.in_([AlertStatus.NEW, AlertStatus.ACKNOWLEDGED, AlertStatus.IN_PROGRESS])
            )
        ).order_by(desc(Alert.triggered_at)).limit(1)
        
        result = await db_session.execute(stmt)
        active_alert = result.scalar_one_or_none()
        
        if not active_alert:
            return {
                "is_latched": False,
                "latch_level": None,
                "can_recover": False,
                "recovery_conditions": None
            }
        
        # 检查是否触发新的锁存级别
        triggered_level = None
        for threshold, level in self.latch_levels.items():
            if current_drawdown <= threshold:
                triggered_level = level
                break  # 取最严重的级别
        
        # 检查恢复条件
        can_recover = await self._check_recovery_conditions(
            portfolio_id, current_drawdown, active_alert, db_session
        )
        
        return {
            "is_latched": True,
            "latch_level": triggered_level,
            "can_recover": can_recover,
            "current_drawdown": current_drawdown,
            "alert_id": active_alert.alert_id,
            "latched_since": active_alert.triggered_at
        }
    
    async def _check_recovery_conditions(
        self,
        portfolio_id: str,
        current_drawdown: float,
        active_alert: Alert,
        db_session: AsyncSession
    ) -> bool:
        """
        检查回撤锁存恢复条件
        
        恢复条件：最大回撤回阈值以内并连续3个交易日满足，或创出本轮新高
        """
        try:
            # 获取原始触发阈值
            trigger_threshold = active_alert.threshold_value or -0.10
            
            # 条件1: 回撤已回到阈值以内
            if current_drawdown > trigger_threshold:
                # TODO: 检查是否连续3个交易日满足
                # 这里需要查询历史组合净值数据
                return True
            
            # 条件2: 创出本轮新高
            # TODO: 检查是否创出新高
            # 这里需要查询历史最高净值
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check recovery conditions: {e}")
            return False
    
    async def resolve_latch(
        self,
        alert_id: str,
        resolution_reason: str,
        db_session: AsyncSession
    ):
        """解除回撤锁存"""
        try:
            # 更新告警状态
            stmt = update(Alert).where(
                Alert.alert_id == alert_id
            ).values(
                status=AlertStatus.RESOLVED,
                resolved_at=datetime.utcnow(),
                resolution_notes=resolution_reason,
                is_latched=False
            )
            
            await db_session.execute(stmt)
            await db_session.commit()
            
            logger.info(f"Drawdown latch resolved for alert {alert_id}: {resolution_reason}")
            
        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to resolve drawdown latch: {e}")
            raise AlertException(f"解除回撤锁存失败: {e}")


class TrendConfirmationService:
    """趋势确认服务"""
    
    def __init__(self):
        self.confirmation_rules = {
            "cross_confirmation": True,  # 双均线交叉确认
            "eod_only": True,           # 仅收盘后确认
            "min_interval_days": 5      # 最小间隔5个交易日
        }
    
    async def confirm_trend_signal(
        self,
        ticker: str,
        signal_type: str,
        market_close_time: datetime,
        technical_data: Dict[str, Any],
        db_session: AsyncSession
    ) -> bool:
        """
        确认趋势信号 (PRD要求: 仅"收盘后"确认双均线交叉)
        """
        # 检查是否为收盘后
        if not self._is_after_market_close(market_close_time):
            logger.debug(f"Trend confirmation skipped - not after market close: {ticker}")
            return False
        
        # 检查双均线交叉
        if not self._validate_ma_cross(technical_data, signal_type):
            logger.debug(f"Trend confirmation failed - invalid MA cross: {ticker}")
            return False
        
        # 检查最小间隔
        can_trigger = await self._check_min_interval(ticker, signal_type, db_session)
        if not can_trigger:
            logger.debug(f"Trend confirmation throttled - min interval not met: {ticker}")
            return False
        
        return True
    
    def _is_after_market_close(self, market_close_time: datetime) -> bool:
        """检查是否为收盘后"""
        now = datetime.utcnow()
        # 假设收盘时间为15:00 (北京时间)
        # 这里简化处理，实际应该根据市场日历
        return now.hour >= 15  # UTC时间，需要调整为实际市场时间
    
    def _validate_ma_cross(self, technical_data: Dict[str, Any], signal_type: str) -> bool:
        """验证双均线交叉"""
        try:
            ma_20 = technical_data.get("ma_20")
            ma_200 = technical_data.get("ma_200") 
            prev_ma_20 = technical_data.get("prev_ma_20")
            prev_ma_200 = technical_data.get("prev_ma_200")
            
            if None in [ma_20, ma_200, prev_ma_20, prev_ma_200]:
                return False
            
            if signal_type == "golden_cross":
                # 金叉：20日线上穿200日线
                current_above = ma_20 > ma_200
                prev_below = prev_ma_20 <= prev_ma_200
                return current_above and prev_below
                
            elif signal_type == "death_cross":
                # 死叉：20日线下穿200日线
                current_below = ma_20 < ma_200
                prev_above = prev_ma_20 >= prev_ma_200
                return current_below and prev_above
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate MA cross: {e}")
            return False
    
    async def _check_min_interval(
        self, 
        ticker: str, 
        signal_type: str, 
        db_session: AsyncSession
    ) -> bool:
        """检查最小间隔 (5个交易日)"""
        try:
            # 查询最近的趋势告警
            stmt = select(Alert).where(
                and_(
                    Alert.alert_type == "trend",
                    Alert.ticker == ticker,
                    Alert.additional_info.op("->>")(text("'signal_type'")) == signal_type
                )
            ).order_by(desc(Alert.triggered_at)).limit(1)
            
            result = await db_session.execute(stmt)
            last_alert = result.scalar_one_or_none()
            
            if not last_alert:
                return True  # 没有历史记录，可以触发
            
            # 检查间隔 (简化为日历日，实际应该使用交易日)
            min_days = self.confirmation_rules["min_interval_days"]
            min_date = last_alert.triggered_at + timedelta(days=min_days)
            
            return datetime.utcnow() >= min_date
            
        except Exception as e:
            logger.error(f"Failed to check min interval: {e}")
            return True  # 检查失败时允许触发


class AlertService:
    """告警服务核心类 (已更新支持PRD要求)"""
    
    def __init__(self):
        # 初始化节流配置 (PRD格式)
        self.throttle_config = {
            "min_interval_minutes": 120,  # 全局回退
            "aggregation_window_minutes": 15,
            "by_type": {
                "event":    {"min_interval_minutes": 60,   "aggregation_window_minutes": 15},
                "kpi":      {"min_interval_minutes": 1440, "aggregation_window_minutes": 60},
                "trend":    {"confirm_on_cross_rule": True, "min_interval_days": 5},
                "drawdown": {"latch": True}
            }
        }
        
        # 初始化控制器
        self.throttle_controller = ThrottleController(self.throttle_config)
        self.drawdown_controller = DrawdownLatchController()
        self.trend_service = TrendConfirmationService()
        
        self.escalation_config = {
            "p3_to_p2_hits": settings.ALERT_ESCALATION_P3_TO_P2_HITS,
            "p2_to_p1_hits": settings.ALERT_ESCALATION_P2_TO_P1_HITS
        }
    
    @log_performance("alert_creation_enhanced")
    async def create_alert_with_throttling(
        self,
        alert_type: str,
        entity_type: Optional[str],
        entity_id: Optional[str],
        ticker: Optional[str],
        title: str,
        message: str,
        severity: str = "P3",
        rule_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        technical_data: Optional[Dict[str, Any]] = None,
        db_session: Optional[AsyncSession] = None
    ) -> Optional[Alert]:
        """
        创建告警 (带节流和特殊处理)
        
        Args:
            alert_type: 告警类型 (event/kpi/trend/drawdown)
            其他参数: 标准告警参数
        
        Returns:
            创建的告警对象，如果被节流则返回None
        """
        try:
            # 1. 检查节流
            should_throttle, next_allowed = await self.throttle_controller.should_throttle(
                alert_type, entity_type, entity_id, db_session
            )
            
            if should_throttle:
                logger.info(f"Alert throttled: {alert_type} for {entity_type}:{entity_id}")
                if next_allowed:
                    logger.info(f"Next allowed time: {next_allowed}")
                return None
            
            # 2. 特殊处理：趋势告警需要确认
            if alert_type == "trend" and technical_data:
                signal_type = additional_info.get("signal_type", "unknown") if additional_info else "unknown"
                confirmed = await self.trend_service.confirm_trend_signal(
                    ticker or entity_id or "unknown",
                    signal_type,
                    datetime.utcnow(),  # 简化：实际应该传入市场收盘时间
                    technical_data,
                    db_session
                )
                
                if not confirmed:
                    logger.info(f"Trend alert not confirmed: {ticker}")
                    return None
            
            # 3. 特殊处理：回撤告警检查锁存状态
            if alert_type == "drawdown":
                latch_status = await self.drawdown_controller.check_latch_status(
                    entity_id or "unknown",
                    current_value or 0.0,
                    db_session
                )
                
                # 如果已锁存且未达到恢复条件，不创建新告警
                if latch_status["is_latched"] and not latch_status["can_recover"]:
                    logger.info(f"Drawdown alert latched, skipping: {entity_id}")
                    return None
                
                # 添加锁存信息到附加信息
                if additional_info is None:
                    additional_info = {}
                additional_info.update({
                    "latch_status": latch_status,
                    "is_latched": True
                })
            
            # 4. 创建告警
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                alert_type=alert_type,
                entity_type=entity_type,
                entity_id=entity_id,
                ticker=ticker,
                title=title,
                message=message,
                severity=severity,
                status=AlertStatus.NEW,
                rule_name=rule_name,
                current_value=current_value,
                threshold_value=threshold_value,
                additional_info=additional_info,
                triggered_at=datetime.utcnow(),
                is_latched=(alert_type == "drawdown"),  # 回撤告警默认锁存
                hits_count=1
            )
            
            if db_session:
                db_session.add(alert)
                await db_session.commit()
                await db_session.refresh(alert)
            
            logger.info(
                f"Alert created: {alert_type} for {entity_type}:{entity_id} "
                f"(severity={severity}, latched={alert.is_latched})"
            )
            
            return alert
            
        except Exception as e:
            if db_session:
                await db_session.rollback()
            logger.error(f"Failed to create alert with throttling: {e}")
            raise AlertException(f"创建告警失败: {e}")
    
    async def check_and_resolve_drawdown_latch(
        self,
        portfolio_id: str,
        current_drawdown: float,
        db_session: AsyncSession
    ) -> bool:
        """检查并解除回撤锁存"""
        try:
            latch_status = await self.drawdown_controller.check_latch_status(
                portfolio_id, current_drawdown, db_session
            )
            
            if latch_status["is_latched"] and latch_status["can_recover"]:
                await self.drawdown_controller.resolve_latch(
                    latch_status["alert_id"],
                    f"回撤从{latch_status['current_drawdown']:.2%}恢复",
                    db_session
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check drawdown latch recovery: {e}")
            return False
    
    async def get_throttle_metrics(self, db_session: AsyncSession) -> Dict[str, Any]:
        """
        获取节流指标
        
        Returns:
            节流命中率和统计信息
        """
        try:
            # 计算各类型告警的节流命中率
            throttle_metrics = {}
            
            for alert_type in ["event", "kpi", "trend", "drawdown"]:
                # 查询最近24小时的告警
                start_time = datetime.utcnow() - timedelta(hours=24)
                
                stmt = select(func.count(Alert.alert_id)).where(
                    and_(
                        Alert.alert_type == alert_type,
                        Alert.triggered_at >= start_time
                    )
                )
                
                result = await db_session.execute(stmt)
                total_attempts = result.scalar() or 0
                
                # TODO: 实际统计节流次数需要额外的记录机制
                # 这里简化处理
                throttled_count = max(0, total_attempts - 10)  # 假设超过10次的为节流
                hit_ratio = throttled_count / max(1, total_attempts)
                
                throttle_metrics[alert_type] = {
                    "total_attempts": total_attempts,
                    "throttled_count": throttled_count,
                    "hit_ratio": round(hit_ratio, 3),
                    "config": self.throttle_controller.get_throttle_config(alert_type)
                }
            
            return {
                "metrics": throttle_metrics,
                "overall_health": {
                    "avg_hit_ratio": sum(m["hit_ratio"] for m in throttle_metrics.values()) / len(throttle_metrics),
                    "alert_storm_detected": any(m["hit_ratio"] > 0.9 for m in throttle_metrics.values()),
                    "config_anomaly_detected": any(m["hit_ratio"] < 0.1 or m["hit_ratio"] > 0.9 for m in throttle_metrics.values())
                },
                "calculated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get throttle metrics: {e}")
            return {
                "error": str(e),
                "calculated_at": datetime.utcnow().isoformat()
            }
    
    async def create_alert(
        self,
        alert_type: str,
        entity_type: Optional[str],
        entity_id: Optional[str],
        ticker: Optional[str],
        title: str,
        message: str,
        rule_name: Optional[str],
        severity: str,
        current_value: Optional[float],
        threshold_value: Optional[float],
        details: Optional[Dict[str, Any]],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """创建告警 (兼容旧API，转发到新的增强版本)"""
        return await self.create_alert_with_throttling(
            alert_type=alert_type,
            entity_type=entity_type,
            entity_id=entity_id,
            ticker=ticker,
            title=title,
            message=message,
            severity=severity,
            rule_name=rule_name,
            current_value=current_value,
            threshold_value=threshold_value,
            additional_info=details,
            db_session=db
        )
    
    async def update_alert(
        self,
        alert_id: str,
        status: Optional[str],
        assignee: Optional[str],
        resolution_note: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """更新告警"""
        try:
            # 获取告警记录
            alert = await self._get_alert_by_id(alert_id, db)
            
            # 记录状态变更历史
            old_status = alert.status.value if alert.status else None
            
            # 更新字段
            if status:
                alert.status = AlertStatus(status)
                if status == "acknowledged" and not alert.acknowledged_at:
                    alert.acknowledged_at = datetime.utcnow()
                elif status == "resolved" and not alert.resolved_at:
                    alert.resolved_at = datetime.utcnow()
            
            if assignee:
                alert.assignee = assignee
            
            if resolution_note:
                alert.resolution_note = resolution_note
            
            await db.commit()
            
            # 记录状态变更事件
            if old_status != status:
                logger.log_alert_event(
                    event_type="alert_status_changed",
                    alert_id=alert_id,
                    old_status=old_status,
                    new_status=status,
                    assignee=assignee
                )
            
            result = {
                "alert_id": alert_id,
                "old_status": old_status,
                "new_status": status,
                "assignee": assignee,
                "updated_at": alert.updated_at,
                "resolution_note": resolution_note
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Update alert failed: {e}", exc_info=True)
            raise ValidationException(f"更新告警失败: {e}")
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        assignee: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """确认告警"""
        try:
            result = await self.update_alert(
                alert_id=alert_id,
                status="acknowledged",
                assignee=assignee,
                resolution_note=None,
                db=db
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Acknowledge alert failed: {e}", exc_info=True)
            raise ValidationException(f"确认告警失败: {e}")
    
    async def batch_acknowledge_alerts(
        self,
        alert_ids: List[str],
        assignee: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """批量确认告警"""
        try:
            results = []
            for alert_id in alert_ids:
                try:
                    result = await self.acknowledge_alert(alert_id, assignee, db)
                    results.append({"alert_id": alert_id, "success": True, "result": result})
                except Exception as e:
                    results.append({"alert_id": alert_id, "success": False, "error": str(e)})
            
            successful_count = len([r for r in results if r["success"]])
            
            return {
                "total_requested": len(alert_ids),
                "successful_count": successful_count,
                "failed_count": len(alert_ids) - successful_count,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Batch acknowledge alerts failed: {e}", exc_info=True)
            raise ValidationException(f"批量确认告警失败: {e}")
    
    async def list_alerts(
        self,
        page: int,
        page_size: int,
        alert_type: Optional[str],
        severity: Optional[str],
        status: Optional[str],
        assignee: Optional[str],
        entity_type: Optional[str],
        ticker: Optional[str],
        start_date: Optional[date],
        end_date: Optional[date],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取告警列表"""
        try:
            # 构建查询
            query = select(Alert)
            
            # 添加过滤条件
            if alert_type:
                query = query.where(Alert.alert_type == AlertType(alert_type))
            if severity:
                query = query.where(Alert.severity == AlertSeverity(severity))
            if status:
                query = query.where(Alert.status == AlertStatus(status))
            if assignee:
                query = query.where(Alert.assignee == assignee)
            if entity_type:
                query = query.where(Alert.entity_type == entity_type)
            if ticker:
                query = query.where(Alert.ticker == ticker)
            if start_date:
                query = query.where(Alert.triggered_at >= datetime.combine(start_date, datetime.min.time()))
            if end_date:
                query = query.where(Alert.triggered_at <= datetime.combine(end_date, datetime.max.time()))
            
            # 获取总数
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await db.execute(count_query)
            total_count = total_result.scalar()
            
            # 分页和排序
            offset = (page - 1) * page_size
            query = query.order_by(desc(Alert.triggered_at)).offset(offset).limit(page_size)
            
            # 执行查询
            result = await db.execute(query)
            alerts = result.scalars().all()
            
            # 构建响应
            alert_list = []
            for alert in alerts:
                alert_data = {
                    "alert_id": str(alert.id),
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "entity_type": alert.entity_type,
                    "entity_id": str(alert.entity_id) if alert.entity_id else None,
                    "ticker": alert.ticker,
                    "title": alert.title,
                    "message": alert.message,
                    "rule_name": alert.rule_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "assignee": alert.assignee,
                    "hits_count": alert.hits_count,
                    "is_latched": alert.is_latched,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "next_allowed_at": alert.next_allowed_at.isoformat() if alert.next_allowed_at else None,
                    "is_overdue": alert.is_overdue(),
                    "can_trigger_now": alert.can_trigger_now()
                }
                alert_list.append(alert_data)
            
            return {
                "alerts": alert_list,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size
                },
                "filters_applied": {
                    "alert_type": alert_type,
                    "severity": severity,
                    "status": status,
                    "assignee": assignee,
                    "entity_type": entity_type,
                    "ticker": ticker,
                    "date_range": [start_date, end_date] if start_date or end_date else None
                }
            }
            
        except Exception as e:
            logger.error(f"List alerts failed: {e}", exc_info=True)
            raise ValidationException(f"获取告警列表失败: {e}")
    
    async def create_alert_rule(
        self,
        name: str,
        description: Optional[str],
        alert_type: str,
        entity_type: Optional[str],
        condition: Dict[str, Any],
        threshold_config: Optional[Dict[str, Any]],
        severity: str,
        message_template: Optional[str],
        throttle_config: Optional[Dict[str, Any]],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """创建告警规则"""
        try:
            # 检查规则名称是否已存在
            existing_rule = await db.execute(
                select(AlertRule).where(AlertRule.name == name)
            )
            if existing_rule.scalar_one_or_none():
                raise ValidationException(f"告警规则 '{name}' 已存在")
            
            # 创建规则
            rule = AlertRule(
                name=name,
                description=description,
                alert_type=AlertType(alert_type),
                entity_type=entity_type,
                condition=condition,
                threshold_config=threshold_config or {},
                severity=AlertSeverity(severity),
                message_template=message_template,
                throttle_config=throttle_config or {}
            )
            
            db.add(rule)
            await db.commit()
            await db.refresh(rule)
            
            result = {
                "rule_id": str(rule.id),
                "name": rule.name,
                "alert_type": alert_type,
                "severity": severity,
                "is_enabled": rule.is_enabled,
                "created_at": rule.created_at
            }
            
            logger.info(f"Alert rule created: {name}")
            return result
            
        except Exception as e:
            logger.error(f"Create alert rule failed: {e}", exc_info=True)
            raise ValidationException(f"创建告警规则失败: {e}")
    
    async def list_alert_rules(
        self,
        enabled_only: bool,
        alert_type: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取告警规则列表"""
        try:
            query = select(AlertRule)
            
            if enabled_only:
                query = query.where(AlertRule.is_enabled == True)
            if alert_type:
                query = query.where(AlertRule.alert_type == AlertType(alert_type))
            
            query = query.order_by(desc(AlertRule.updated_at))
            
            result = await db.execute(query)
            rules = result.scalars().all()
            
            rule_list = []
            for rule in rules:
                rule_data = {
                    "rule_id": str(rule.id),
                    "name": rule.name,
                    "description": rule.description,
                    "alert_type": rule.alert_type.value,
                    "entity_type": rule.entity_type,
                    "condition": rule.condition,
                    "threshold_config": rule.threshold_config,
                    "severity": rule.severity.value,
                    "message_template": rule.message_template,
                    "throttle_config": rule.throttle_config,
                    "is_enabled": rule.is_enabled,
                    "created_at": rule.created_at.isoformat(),
                    "updated_at": rule.updated_at.isoformat()
                }
                rule_list.append(rule_data)
            
            return {
                "rules": rule_list,
                "total_count": len(rule_list),
                "enabled_count": len([r for r in rule_list if r["is_enabled"]])
            }
            
        except Exception as e:
            logger.error(f"List alert rules failed: {e}", exc_info=True)
            raise ValidationException(f"获取告警规则失败: {e}")
    
    async def create_event(
        self,
        event_type: str,
        title: str,
        description: Optional[str],
        entity_type: Optional[str],
        entity_id: Optional[str],
        ticker: Optional[str],
        event_date: date,
        event_time: Optional[datetime],
        lead_time_days: Optional[int],
        event_data: Optional[Dict[str, Any]],
        source: Optional[str],
        source_url: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """创建事件"""
        try:
            event = EventCalendar(
                event_type=event_type,
                title=title,
                description=description,
                entity_type=entity_type,
                entity_id=uuid.UUID(entity_id) if entity_id else None,
                ticker=ticker,
                event_date=event_date,
                event_time=event_time,
                lead_time_days=lead_time_days,
                event_data=event_data or {},
                source=source,
                source_url=source_url
            )
            
            db.add(event)
            await db.commit()
            await db.refresh(event)
            
            result = {
                "event_id": str(event.id),
                "event_type": event_type,
                "title": title,
                "event_date": event_date.isoformat(),
                "lead_time_days": lead_time_days,
                "should_send_reminder": event.should_send_reminder(),
                "created_at": event.created_at
            }
            
            logger.info(f"Event created: {title} on {event_date}")
            return result
            
        except Exception as e:
            logger.error(f"Create event failed: {e}", exc_info=True)
            raise ValidationException(f"创建事件失败: {e}")
    
    async def list_events(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        event_type: Optional[str],
        ticker: Optional[str],
        pending_reminder_only: bool,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取事件列表"""
        try:
            query = select(EventCalendar)
            
            # 默认日期范围
            if not start_date:
                start_date = date.today()
            if not end_date:
                end_date = date.today() + timedelta(days=30)
            
            query = query.where(
                and_(
                    EventCalendar.event_date >= start_date,
                    EventCalendar.event_date <= end_date,
                    EventCalendar.is_active == True
                )
            )
            
            if event_type:
                query = query.where(EventCalendar.event_type == event_type)
            if ticker:
                query = query.where(EventCalendar.ticker == ticker)
            if pending_reminder_only:
                query = query.where(EventCalendar.reminder_sent == False)
            
            query = query.order_by(EventCalendar.event_date)
            
            result = await db.execute(query)
            events = result.scalars().all()
            
            event_list = []
            for event in events:
                event_data = {
                    "event_id": str(event.id),
                    "event_type": event.event_type,
                    "title": event.title,
                    "description": event.description,
                    "entity_type": event.entity_type,
                    "entity_id": str(event.entity_id) if event.entity_id else None,
                    "ticker": event.ticker,
                    "event_date": event.event_date.isoformat(),
                    "event_time": event.event_time.isoformat() if event.event_time else None,
                    "lead_time_days": event.lead_time_days,
                    "reminder_sent": event.reminder_sent,
                    "should_send_reminder": event.should_send_reminder(),
                    "event_data": event.event_data,
                    "source": event.source,
                    "source_url": event.source_url
                }
                event_list.append(event_data)
            
            return {
                "events": event_list,
                "total_count": len(event_list),
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "pending_reminders": len([e for e in event_list if e["should_send_reminder"]])
            }
            
        except Exception as e:
            logger.error(f"List events failed: {e}", exc_info=True)
            raise ValidationException(f"获取事件列表失败: {e}")
    
    async def get_alert_dashboard(
        self, days: int, db: AsyncSession
    ) -> Dict[str, Any]:
        """获取告警仪表板数据"""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 基础统计
            basic_stats_query = select(
                func.count(Alert.id).label("total_alerts"),
                func.count(case((Alert.status == AlertStatus.NEW, 1))).label("new_alerts"),
                func.count(case((Alert.status == AlertStatus.ACKNOWLEDGED, 1))).label("acknowledged_alerts"),
                func.count(case((Alert.status == AlertStatus.RESOLVED, 1))).label("resolved_alerts"),
                func.count(case((Alert.severity == AlertSeverity.P1, 1))).label("p1_alerts"),
                func.count(case((Alert.severity == AlertSeverity.P2, 1))).label("p2_alerts"),
                func.count(case((Alert.severity == AlertSeverity.P3, 1))).label("p3_alerts")
            ).where(Alert.triggered_at >= start_date)
            
            basic_result = await db.execute(basic_stats_query)
            basic_stats = basic_result.fetchone()
            
            # 按类型统计
            type_stats_query = select(
                Alert.alert_type,
                func.count(Alert.id).label("count")
            ).where(Alert.triggered_at >= start_date).group_by(Alert.alert_type)
            
            type_result = await db.execute(type_stats_query)
            type_stats = {row.alert_type.value: row.count for row in type_result}
            
            # 按日期统计
            daily_stats_query = select(
                func.date(Alert.triggered_at).label("alert_date"),
                func.count(Alert.id).label("count")
            ).where(Alert.triggered_at >= start_date).group_by(func.date(Alert.triggered_at)).order_by(func.date(Alert.triggered_at))
            
            daily_result = await db.execute(daily_stats_query)
            daily_stats = [{"date": row.alert_date.isoformat(), "count": row.count} for row in daily_result]
            
            # 超时告警
            overdue_alerts_query = select(Alert).where(
                and_(
                    Alert.triggered_at >= start_date,
                    Alert.status.in_([AlertStatus.NEW, AlertStatus.ACKNOWLEDGED])
                )
            )
            
            overdue_result = await db.execute(overdue_alerts_query)
            overdue_alerts = [alert for alert in overdue_result.scalars().all() if alert.is_overdue()]
            
            return {
                "dashboard_period_days": days,
                "basic_stats": {
                    "total_alerts": basic_stats.total_alerts,
                    "new_alerts": basic_stats.new_alerts,
                    "acknowledged_alerts": basic_stats.acknowledged_alerts,
                    "resolved_alerts": basic_stats.resolved_alerts,
                    "resolution_rate": (basic_stats.resolved_alerts / basic_stats.total_alerts * 100) if basic_stats.total_alerts > 0 else 0
                },
                "severity_distribution": {
                    "P1": basic_stats.p1_alerts,
                    "P2": basic_stats.p2_alerts,
                    "P3": basic_stats.p3_alerts
                },
                "type_distribution": type_stats,
                "daily_trend": daily_stats,
                "overdue_alerts": {
                    "count": len(overdue_alerts),
                    "alerts": [
                        {
                            "alert_id": str(alert.id),
                            "title": alert.title,
                            "severity": alert.severity.value,
                            "triggered_at": alert.triggered_at.isoformat(),
                            "overdue_hours": int((datetime.utcnow() - alert.triggered_at).total_seconds() / 3600)
                        } for alert in overdue_alerts[:10]  # 仅返回前10个
                    ]
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get alert dashboard failed: {e}", exc_info=True)
            raise ValidationException(f"获取告警仪表板失败: {e}")
    
    async def get_alert_metrics(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        group_by: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取告警指标统计"""
        try:
            # 默认日期范围
            if not start_date:
                start_date = date.today() - timedelta(days=7)
            if not end_date:
                end_date = date.today()
            
            # 基础查询
            base_query = select(Alert).where(
                Alert.triggered_at >= datetime.combine(start_date, datetime.min.time()),
                Alert.triggered_at <= datetime.combine(end_date, datetime.max.time())
            )
            
            metrics = {}
            
            if group_by == "date":
                # 按日期分组
                query = select(
                    func.date(Alert.triggered_at).label("date"),
                    func.count(Alert.id).label("total"),
                    func.avg(
                        case((Alert.resolved_at.is_not(None), 
                             func.extract('epoch', Alert.resolved_at - Alert.triggered_at) / 60))
                    ).label("avg_resolution_minutes")
                ).where(
                    Alert.triggered_at >= datetime.combine(start_date, datetime.min.time()),
                    Alert.triggered_at <= datetime.combine(end_date, datetime.max.time())
                ).group_by(func.date(Alert.triggered_at)).order_by(func.date(Alert.triggered_at))
                
                result = await db.execute(query)
                metrics["daily_metrics"] = [
                    {
                        "date": row.date.isoformat(),
                        "total_alerts": row.total,
                        "avg_resolution_minutes": float(row.avg_resolution_minutes) if row.avg_resolution_minutes else None
                    } for row in result
                ]
                
            elif group_by == "type":
                # 按类型分组
                query = select(
                    Alert.alert_type,
                    func.count(Alert.id).label("total"),
                    func.count(case((Alert.status == AlertStatus.RESOLVED, 1))).label("resolved"),
                    func.avg(
                        case((Alert.resolved_at.is_not(None), 
                             func.extract('epoch', Alert.resolved_at - Alert.triggered_at) / 60))
                    ).label("avg_resolution_minutes")
                ).where(
                    Alert.triggered_at >= datetime.combine(start_date, datetime.min.time()),
                    Alert.triggered_at <= datetime.combine(end_date, datetime.max.time())
                ).group_by(Alert.alert_type)
                
                result = await db.execute(query)
                metrics["type_metrics"] = [
                    {
                        "alert_type": row.alert_type.value,
                        "total_alerts": row.total,
                        "resolved_alerts": row.resolved,
                        "resolution_rate": (row.resolved / row.total * 100) if row.total > 0 else 0,
                        "avg_resolution_minutes": float(row.avg_resolution_minutes) if row.avg_resolution_minutes else None
                    } for row in result
                ]
                
            elif group_by == "severity":
                # 按严重程度分组
                query = select(
                    Alert.severity,
                    func.count(Alert.id).label("total"),
                    func.count(case((Alert.status == AlertStatus.RESOLVED, 1))).label("resolved"),
                    func.avg(
                        case((Alert.acknowledged_at.is_not(None), 
                             func.extract('epoch', Alert.acknowledged_at - Alert.triggered_at) / 60))
                    ).label("avg_ack_minutes")
                ).where(
                    Alert.triggered_at >= datetime.combine(start_date, datetime.min.time()),
                    Alert.triggered_at <= datetime.combine(end_date, datetime.max.time())
                ).group_by(Alert.severity)
                
                result = await db.execute(query)
                metrics["severity_metrics"] = [
                    {
                        "severity": row.severity.value,
                        "total_alerts": row.total,
                        "resolved_alerts": row.resolved,
                        "resolution_rate": (row.resolved / row.total * 100) if row.total > 0 else 0,
                        "avg_acknowledgment_minutes": float(row.avg_ack_minutes) if row.avg_ack_minutes else None
                    } for row in result
                ]
            
            return {
                "metrics": metrics,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "group_by": group_by
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get alert metrics failed: {e}", exc_info=True)
            raise ValidationException(f"获取告警指标失败: {e}")
    
    async def test_alert_rule(
        self,
        rule_name: str,
        test_data: Dict[str, Any],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """测试告警规则"""
        try:
            # 获取规则
            rule_result = await db.execute(
                select(AlertRule).where(AlertRule.name == rule_name)
            )
            rule = rule_result.scalar_one_or_none()
            
            if not rule:
                raise ValidationException(f"告警规则 '{rule_name}' 不存在")
            
            # 模拟条件评估
            condition_result = self._evaluate_condition(rule.condition, test_data)
            
            # 生成测试结果
            result = {
                "rule_name": rule_name,
                "rule_enabled": rule.is_enabled,
                "test_data": test_data,
                "condition_met": condition_result["met"],
                "condition_details": condition_result,
                "would_create_alert": condition_result["met"] and rule.is_enabled,
                "expected_severity": rule.severity.value,
                "expected_message": self._render_message_template(
                    rule.message_template, test_data
                ) if rule.message_template else None,
                "throttle_config": rule.throttle_config,
                "tested_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Test alert rule failed: {e}", exc_info=True)
            raise ValidationException(f"测试告警规则失败: {e}")
    
    async def health_check(self, db: AsyncSession) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查数据库连接
            await db.execute("SELECT 1")
            
            # 基本统计
            alert_count = await db.execute(select(func.count(Alert.id)))
            rule_count = await db.execute(select(func.count(AlertRule.id)))
            event_count = await db.execute(select(func.count(EventCalendar.id)))
            
            # 检查最近24小时的告警活动
            recent_alerts = await db.execute(
                select(func.count(Alert.id)).where(
                    Alert.triggered_at >= datetime.utcnow() - timedelta(hours=24)
                )
            )
            
            return {
                "database_connected": True,
                "total_alerts": alert_count.scalar(),
                "total_rules": rule_count.scalar(),
                "total_events": event_count.scalar(),
                "recent_24h_alerts": recent_alerts.scalar(),
                "features_available": [
                    "alert_creation",
                    "rule_management", 
                    "event_calendar",
                    "throttling",
                    "escalation",
                    "metrics_reporting"
                ]
            }
            
        except Exception as e:
            logger.error(f"Alert service health check failed: {e}", exc_info=True)
            return {
                "database_connected": False,
                "error": str(e)
            }
    
    # 辅助方法
    def _generate_throttle_key(
        self, alert_type: str, entity_type: Optional[str], 
        entity_id: Optional[str], ticker: Optional[str], rule_name: Optional[str]
    ) -> str:
        """生成节流键"""
        parts = [alert_type]
        if entity_type:
            parts.append(entity_type)
        if entity_id:
            parts.append(entity_id)
        if ticker:
            parts.append(ticker)
        if rule_name:
            parts.append(rule_name)
        return ":".join(parts)
    
    async def _is_throttled(self, throttle_key: str, alert_type: str, db: AsyncSession) -> bool:
        """检查是否被节流 (兼容方法，使用新的节流控制器)"""
        # 解析节流键获取参数
        parts = throttle_key.split(":")
        if len(parts) >= 3:
            entity_type = parts[1] if len(parts) > 1 else None
            entity_id = parts[2] if len(parts) > 2 else None
            
            should_throttle, _ = await self.throttle_controller.should_throttle(
                alert_type, entity_type, entity_id, db
            )
            return should_throttle
        return False
    
    def _calculate_next_allowed_time(self, alert_type: str) -> datetime:
        """计算下次允许时间"""
        throttle_config = self.throttle_controller.get_throttle_config(alert_type)
        
        if alert_type == "trend":
            # 趋势型告警按天节流
            days = throttle_config.get("min_interval_days", 5)
            return datetime.utcnow() + timedelta(days=days)
        else:
            # 其他类型按分钟节流
            minutes = throttle_config.get("min_interval_minutes", 120)
            return datetime.utcnow() + timedelta(minutes=minutes)
    
    async def _check_escalation(self, throttle_key: str, db: AsyncSession) -> Optional[str]:
        """检查是否需要升级 (返回字符串格式的严重程度)"""
        # 获取相同throttle_key的告警数量
        alerts = await db.execute(
            select(Alert).where(Alert.alert_id.like(f"%{throttle_key.split(':')[-1]}%"))
        )
        alert_list = alerts.scalars().all()
        
        if not alert_list:
            return None
        
        hits = len(alert_list)
        
        # 检查升级条件 (简化版本)
        if hits >= self.escalation_config["p2_to_p1_hits"]:
            return "P1"
        elif hits >= self.escalation_config["p3_to_p2_hits"]:
            return "P2"
        
        return None
    
    async def _send_notification(self, alert: Alert, db: AsyncSession) -> bool:
        """发送通知"""
        try:
            # 创建通知日志记录
            notification = NotificationLog(
                notification_type="alert",
                channel="in_app",  # 默认应用内通知
                recipient="system",
                alert_id=alert.alert_id,  # 使用新模型的alert_id字段
                subject=f"[{alert.severity}] {alert.title}",
                content=alert.message,
                status="sent",
                sent_at=datetime.utcnow()
            )
            
            db.add(notification)
            await db.commit()
            
            # 这里可以集成实际的通知系统 (邮件、短信、Webhook等)
            logger.info(f"Notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Send notification failed: {e}")
            return False
    
    async def _get_alert_by_id(self, alert_id: str, db: AsyncSession) -> Alert:
        """根据ID获取告警"""
        result = await db.execute(
            select(Alert).where(Alert.alert_id == alert_id)
        )
        alert = result.scalar_one_or_none()
        if not alert:
            raise ValidationException(f"告警不存在: {alert_id}")
        return alert
    
    def _evaluate_condition(self, condition: Dict[str, Any], test_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估条件"""
        # 简化的条件评估逻辑，生产环境需要更复杂的规则引擎
        try:
            operator = condition.get("operator", "gt")
            field = condition.get("field")
            threshold = condition.get("threshold")
            
            if not field or threshold is None:
                return {"met": False, "reason": "条件配置不完整"}
            
            value = test_data.get(field)
            if value is None:
                return {"met": False, "reason": f"测试数据中缺少字段: {field}"}
            
            if operator == "gt":
                met = value > threshold
            elif operator == "lt":
                met = value < threshold
            elif operator == "eq":
                met = value == threshold
            elif operator == "ge":
                met = value >= threshold
            elif operator == "le":
                met = value <= threshold
            else:
                return {"met": False, "reason": f"不支持的操作符: {operator}"}
            
            return {
                "met": met,
                "field": field,
                "operator": operator,
                "threshold": threshold,
                "actual_value": value,
                "comparison": f"{value} {operator} {threshold} = {met}"
            }
            
        except Exception as e:
            return {"met": False, "reason": f"条件评估失败: {e}"}
    
    def _render_message_template(self, template: Optional[str], data: Dict[str, Any]) -> Optional[str]:
        """渲染消息模板"""
        if not template:
            return None
        
        try:
            # 简单的模板替换，生产环境可使用Jinja2等模板引擎
            message = template
            for key, value in data.items():
                message = message.replace(f"{{{key}}}", str(value))
            return message
        except Exception as e:
            logger.error(f"Render message template failed: {e}")
            return template


# 创建全局服务实例
alert_service = AlertService()