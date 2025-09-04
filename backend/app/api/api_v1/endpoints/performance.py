"""
性能监控API端点
提供系统性能监控、模型性能统计和优化建议
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
import logging

from pydantic import BaseModel, Field

from backend.app.services.performance_monitor import performance_monitor
from backend.app.core.deps import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class MonitoringConfig(BaseModel):
    """监控配置"""
    interval: float = Field(default=5.0, description="监控间隔(秒)", ge=1.0, le=60.0)
    enable_detailed_metrics: bool = Field(default=True, description="是否启用详细指标")


@router.get("/system/status")
async def get_system_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取系统状态
    
    Args:
        current_user: 当前用户
        
    Returns:
        系统状态信息
    """
    try:
        logger.info(f"User {current_user.id} requesting system status")
        
        status = performance_monitor.get_system_status()
        
        if not status:
            return {
                "status": "no_data",
                "message": "性能监控尚未启动或无数据",
                "monitoring_active": performance_monitor.monitoring
            }
        
        return {
            "status": "success",
            "data": status.dict(),
            "monitoring_active": performance_monitor.monitoring
        }
        
    except Exception as e:
        logger.error(f"Get system status failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@router.get("/models/performance")
async def get_model_performance(
    model_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取模型性能统计
    
    Args:
        model_name: 模型名称（可选）
        current_user: 当前用户
        
    Returns:
        模型性能统计
    """
    try:
        logger.info(f"User {current_user.id} requesting model performance for {model_name or 'all models'}")
        
        performance_data = performance_monitor.get_model_performance_summary(model_name)
        
        return {
            "status": "success",
            "data": performance_data,
            "model_name": model_name,
            "total_models": len(performance_data) if not model_name else 1
        }
        
    except Exception as e:
        logger.error(f"Get model performance failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型性能失败: {str(e)}")


@router.get("/models/{model_name}/trends")
async def get_performance_trends(
    model_name: str,
    hours: int = 1,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取模型性能趋势
    
    Args:
        model_name: 模型名称
        hours: 时间范围(小时)
        current_user: 当前用户
        
    Returns:
        性能趋势数据
    """
    try:
        logger.info(f"User {current_user.id} requesting performance trends for {model_name}")
        
        if hours < 1 or hours > 24:
            raise HTTPException(status_code=400, detail="时间范围必须在1-24小时之间")
        
        trends = performance_monitor.get_performance_trends(model_name, hours)
        
        return {
            "status": "success",
            "model_name": model_name,
            "time_range_hours": hours,
            "data": trends,
            "data_points": len(trends.get("timestamps", []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get performance trends failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能趋势失败: {str(e)}")


@router.get("/hardware/utilization")
async def get_hardware_utilization(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取硬件利用率报告
    
    Args:
        current_user: 当前用户
        
    Returns:
        硬件利用率报告
    """
    try:
        logger.info(f"User {current_user.id} requesting hardware utilization report")
        
        report = performance_monitor.get_hardware_utilization_report()
        
        return {
            "status": "success",
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Get hardware utilization failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取硬件利用率失败: {str(e)}")


@router.get("/analysis/bottlenecks")
async def analyze_bottlenecks(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    分析性能瓶颈
    
    Args:
        current_user: 当前用户
        
    Returns:
        瓶颈分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting bottleneck analysis")
        
        analysis = performance_monitor.analyze_performance_bottlenecks()
        
        return {
            "status": "success",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Bottleneck analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"瓶颈分析失败: {str(e)}")


@router.get("/models/{model_name}/optimization")
async def get_optimization_suggestions(
    model_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取模型优化建议
    
    Args:
        model_name: 模型名称
        current_user: 当前用户
        
    Returns:
        优化建议
    """
    try:
        logger.info(f"User {current_user.id} requesting optimization suggestions for {model_name}")
        
        suggestions = performance_monitor.get_optimization_suggestions(model_name)
        
        return {
            "status": "success",
            "model_name": model_name,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Get optimization suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取优化建议失败: {str(e)}")


@router.post("/monitoring/start")
async def start_monitoring(
    config: MonitoringConfig,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    启动性能监控
    
    Args:
        config: 监控配置
        current_user: 当前用户
        
    Returns:
        启动结果
    """
    try:
        logger.info(f"User {current_user.id} starting performance monitoring")
        
        performance_monitor.start_monitoring(config.interval)
        
        return {
            "status": "success",
            "message": "性能监控已启动",
            "config": config.dict(),
            "monitoring_active": performance_monitor.monitoring
        }
        
    except Exception as e:
        logger.error(f"Start monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"启动监控失败: {str(e)}")


@router.post("/monitoring/stop")
async def stop_monitoring(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    停止性能监控
    
    Args:
        current_user: 当前用户
        
    Returns:
        停止结果
    """
    try:
        logger.info(f"User {current_user.id} stopping performance monitoring")
        
        performance_monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "性能监控已停止",
            "monitoring_active": performance_monitor.monitoring
        }
        
    except Exception as e:
        logger.error(f"Stop monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"停止监控失败: {str(e)}")


@router.get("/dashboard")
async def get_performance_dashboard(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取性能仪表板数据
    
    Args:
        current_user: 当前用户
        
    Returns:
        仪表板数据
    """
    try:
        logger.info(f"User {current_user.id} requesting performance dashboard")
        
        dashboard_data = performance_monitor.get_performance_dashboard_data()
        
        return {
            "status": "success",
            "dashboard": dashboard_data,
            "timestamp": "2025-01-04T16:57:00+08:00"
        }
        
    except Exception as e:
        logger.error(f"Get performance dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能仪表板失败: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    性能监控服务健康检查
    
    Returns:
        健康状态信息
    """
    try:
        # 检查监控状态
        monitoring_active = performance_monitor.monitoring
        metrics_count = len(performance_monitor.metrics_history)
        status_count = len(performance_monitor.system_status_history)
        
        # 检查最近的系统状态
        recent_status = performance_monitor.get_system_status()
        
        health_score = 100
        issues = []
        
        if not monitoring_active:
            health_score -= 30
            issues.append("性能监控未启动")
        
        if metrics_count == 0:
            health_score -= 20
            issues.append("无性能指标数据")
        
        if recent_status and recent_status.cpu_percent > 90:
            health_score -= 25
            issues.append("CPU使用率过高")
        
        if recent_status and recent_status.memory_percent > 90:
            health_score -= 25
            issues.append("内存使用率过高")
        
        return {
            "status": "healthy" if health_score >= 70 else "degraded",
            "health_score": health_score,
            "monitoring_active": monitoring_active,
            "metrics_collected": metrics_count,
            "status_points": status_count,
            "issues": issues,
            "timestamp": "2025-01-04T16:57:00+08:00"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-04T16:57:00+08:00"
        }
