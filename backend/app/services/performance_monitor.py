"""
性能监控和调优模块
监控AI模型性能、资源使用情况，并提供优化建议
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque
import threading

import psutil
import torch
from pydantic import BaseModel

# 条件导入NVIDIA监控库
try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    model_name: str
    inference_time: float  # 推理时间(秒)
    throughput: float  # 吞吐量(requests/s)
    memory_usage: int  # 内存使用(MB)
    gpu_utilization: float  # GPU使用率(%)
    dla_utilization: float  # DLA使用率(%)
    cpu_utilization: float  # CPU使用率(%)
    batch_size: int  # 批处理大小
    hardware_target: str  # 硬件目标


class SystemStatus(BaseModel):
    """系统状态"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    dla_percent: float = 0.0
    temperature: Dict[str, float] = {}
    power_usage: Dict[str, float] = {}


class PerformanceMonitor:
    """
    性能监控器
    
    功能：
    1. 实时性能监控
    2. 资源使用统计
    3. 性能瓶颈识别
    4. 优化建议生成
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.system_status_history: deque = deque(maxlen=history_size)
        
        # 性能统计
        self.model_stats: Dict[str, Dict[str, Any]] = {}
        
        # 监控线程
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 初始化NVIDIA监控
        self.nvml_available = NVML_AVAILABLE
        if self.nvml_available:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                logger.info(f"NVML initialized, found {self.gpu_count} GPU(s)")
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}")
                self.nvml_available = False
    
    def start_monitoring(self, interval: float = 5.0):
        """
        启动性能监控
        
        Args:
            interval: 监控间隔(秒)
        """
        if self.monitoring:
            logger.warning("Performance monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统状态
                status = self._collect_system_status()
                self.system_status_history.append(status)
                
                # 等待下次监控
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_system_status(self) -> SystemStatus:
        """收集系统状态"""
        # CPU和内存
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU状态
        gpu_percent = 0.0
        gpu_memory_percent = 0.0
        temperature = {}
        power_usage = {}
        
        if self.nvml_available:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU使用率
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = max(gpu_percent, utilization.gpu)
                    
                    # GPU内存
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_percent = max(
                        gpu_memory_percent,
                        (memory_info.used / memory_info.total) * 100
                    )
                    
                    # 温度
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        temperature[f"gpu_{i}"] = temp
                    except:
                        pass
                    
                    # 功耗
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换为瓦特
                        power_usage[f"gpu_{i}"] = power
                    except:
                        pass
                        
            except Exception as e:
                logger.debug(f"NVML monitoring error: {e}")
        
        # DLA使用率（简化实现）
        dla_percent = 0.0  # 需要专门的DLA监控API
        
        return SystemStatus(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            dla_percent=dla_percent,
            temperature=temperature,
            power_usage=power_usage
        )
    
    def record_inference_metrics(
        self,
        model_name: str,
        inference_time: float,
        batch_size: int = 1,
        hardware_target: str = "gpu"
    ):
        """
        记录推理性能指标
        
        Args:
            model_name: 模型名称
            inference_time: 推理时间
            batch_size: 批处理大小
            hardware_target: 硬件目标
        """
        try:
            # 计算吞吐量
            throughput = batch_size / inference_time if inference_time > 0 else 0
            
            # 获取当前资源使用
            current_status = self._collect_system_status()
            
            # 创建性能指标
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                model_name=model_name,
                inference_time=inference_time,
                throughput=throughput,
                memory_usage=int(current_status.memory_percent * 
                               psutil.virtual_memory().total / (100 * 1024**2)),
                gpu_utilization=current_status.gpu_percent,
                dla_utilization=current_status.dla_percent,
                cpu_utilization=current_status.cpu_percent,
                batch_size=batch_size,
                hardware_target=hardware_target
            )
            
            # 添加到历史记录
            self.metrics_history.append(metrics)
            
            # 更新模型统计
            self._update_model_stats(model_name, metrics)
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
    
    def _update_model_stats(self, model_name: str, metrics: PerformanceMetrics):
        """更新模型统计信息"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "total_requests": 0,
                "total_inference_time": 0.0,
                "avg_inference_time": 0.0,
                "avg_throughput": 0.0,
                "max_throughput": 0.0,
                "min_inference_time": float('inf'),
                "max_inference_time": 0.0,
                "last_updated": datetime.now()
            }
        
        stats = self.model_stats[model_name]
        stats["total_requests"] += 1
        stats["total_inference_time"] += metrics.inference_time
        stats["avg_inference_time"] = stats["total_inference_time"] / stats["total_requests"]
        stats["avg_throughput"] = (stats["avg_throughput"] * (stats["total_requests"] - 1) + 
                                  metrics.throughput) / stats["total_requests"]
        stats["max_throughput"] = max(stats["max_throughput"], metrics.throughput)
        stats["min_inference_time"] = min(stats["min_inference_time"], metrics.inference_time)
        stats["max_inference_time"] = max(stats["max_inference_time"], metrics.inference_time)
        stats["last_updated"] = datetime.now()
    
    def get_model_performance_summary(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型性能摘要
        
        Args:
            model_name: 模型名称，为空则返回所有模型
            
        Returns:
            性能摘要
        """
        if model_name:
            return self.model_stats.get(model_name, {})
        
        return self.model_stats.copy()
    
    def get_system_status(self) -> Optional[SystemStatus]:
        """获取最新系统状态"""
        if self.system_status_history:
            return self.system_status_history[-1]
        return None
    
    def get_performance_trends(self, model_name: str, hours: int = 1) -> Dict[str, List[float]]:
        """
        获取性能趋势
        
        Args:
            model_name: 模型名称
            hours: 时间范围(小时)
            
        Returns:
            性能趋势数据
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # 过滤指定时间范围和模型的指标
        filtered_metrics = [
            m for m in self.metrics_history
            if m.model_name == model_name and m.timestamp >= cutoff_time
        ]
        
        if not filtered_metrics:
            return {}
        
        return {
            "timestamps": [m.timestamp.isoformat() for m in filtered_metrics],
            "inference_times": [m.inference_time for m in filtered_metrics],
            "throughputs": [m.throughput for m in filtered_metrics],
            "gpu_utilizations": [m.gpu_utilization for m in filtered_metrics],
            "memory_usages": [m.memory_usage for m in filtered_metrics]
        }
    
    def analyze_performance_bottlenecks(self) -> Dict[str, Any]:
        """
        分析性能瓶颈
        
        Returns:
            瓶颈分析结果
        """
        analysis = {
            "bottlenecks": [],
            "recommendations": [],
            "overall_health": "good"
        }
        
        if not self.system_status_history:
            return analysis
        
        # 分析最近的系统状态
        recent_status = list(self.system_status_history)[-10:]  # 最近10个状态
        
        # CPU瓶颈检测
        avg_cpu = sum(s.cpu_percent for s in recent_status) / len(recent_status)
        if avg_cpu > 80:
            analysis["bottlenecks"].append("CPU使用率过高")
            analysis["recommendations"].append("考虑增加CPU核心数或优化CPU密集型操作")
        
        # 内存瓶颈检测
        avg_memory = sum(s.memory_percent for s in recent_status) / len(recent_status)
        if avg_memory > 85:
            analysis["bottlenecks"].append("内存使用率过高")
            analysis["recommendations"].append("考虑卸载不常用模型或增加内存")
        
        # GPU瓶颈检测
        avg_gpu = sum(s.gpu_percent for s in recent_status) / len(recent_status)
        if avg_gpu > 90:
            analysis["bottlenecks"].append("GPU使用率过高")
            analysis["recommendations"].append("考虑使用DLA分担部分计算负载")
        elif avg_gpu < 30:
            analysis["recommendations"].append("GPU使用率较低，可以考虑增加批处理大小")
        
        # 温度检测
        for status in recent_status:
            for device, temp in status.temperature.items():
                if temp > 80:
                    analysis["bottlenecks"].append(f"{device}温度过高: {temp}°C")
                    analysis["recommendations"].append("检查散热系统，考虑降低性能设置")
        
        # 确定整体健康状态
        if len(analysis["bottlenecks"]) == 0:
            analysis["overall_health"] = "excellent"
        elif len(analysis["bottlenecks"]) <= 2:
            analysis["overall_health"] = "good"
        elif len(analysis["bottlenecks"]) <= 4:
            analysis["overall_health"] = "fair"
        else:
            analysis["overall_health"] = "poor"
        
        return analysis
    
    def get_optimization_suggestions(self, model_name: str) -> List[str]:
        """
        获取模型优化建议
        
        Args:
            model_name: 模型名称
            
        Returns:
            优化建议列表
        """
        suggestions = []
        
        if model_name not in self.model_stats:
            return ["模型尚未运行，无法提供优化建议"]
        
        stats = self.model_stats[model_name]
        
        # 基于推理时间的建议
        if stats["avg_inference_time"] > 2.0:
            suggestions.append("推理时间较长，考虑使用更高精度的量化或硬件加速")
        
        if stats["avg_inference_time"] < 0.1:
            suggestions.append("推理时间很短，可以考虑增加批处理大小提高吞吐量")
        
        # 基于吞吐量的建议
        if stats["avg_throughput"] < 10:
            suggestions.append("吞吐量较低，考虑启用批处理或使用更快的硬件")
        
        # 基于硬件使用情况的建议
        recent_metrics = [m for m in self.metrics_history if m.model_name == model_name][-10:]
        
        if recent_metrics:
            avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
            
            if avg_gpu_util < 50:
                suggestions.append("GPU使用率较低，考虑增加并发请求或批处理大小")
            elif avg_gpu_util > 95:
                suggestions.append("GPU使用率过高，考虑使用DLA分担计算或降低批处理大小")
        
        return suggestions if suggestions else ["当前性能表现良好，无需特别优化"]
    
    def get_hardware_utilization_report(self) -> Dict[str, Any]:
        """
        获取硬件利用率报告
        
        Returns:
            硬件利用率报告
        """
        if not self.system_status_history:
            return {"error": "No monitoring data available"}
        
        recent_status = list(self.system_status_history)[-60:]  # 最近60个状态点
        
        report = {
            "monitoring_period": {
                "start": recent_status[0].timestamp.isoformat(),
                "end": recent_status[-1].timestamp.isoformat(),
                "duration_minutes": len(recent_status) * 5 / 60  # 假设5秒间隔
            },
            "cpu": {
                "avg_utilization": sum(s.cpu_percent for s in recent_status) / len(recent_status),
                "max_utilization": max(s.cpu_percent for s in recent_status),
                "min_utilization": min(s.cpu_percent for s in recent_status)
            },
            "memory": {
                "avg_utilization": sum(s.memory_percent for s in recent_status) / len(recent_status),
                "max_utilization": max(s.memory_percent for s in recent_status),
                "min_utilization": min(s.memory_percent for s in recent_status)
            },
            "gpu": {
                "avg_utilization": sum(s.gpu_percent for s in recent_status) / len(recent_status),
                "max_utilization": max(s.gpu_percent for s in recent_status),
                "min_utilization": min(s.gpu_percent for s in recent_status),
                "avg_memory_utilization": sum(s.gpu_memory_percent for s in recent_status) / len(recent_status)
            },
            "dla": {
                "avg_utilization": sum(s.dla_percent for s in recent_status) / len(recent_status),
                "max_utilization": max(s.dla_percent for s in recent_status),
                "min_utilization": min(s.dla_percent for s in recent_status)
            }
        }
        
        # 添加温度和功耗统计
        if recent_status[0].temperature:
            temp_data = {}
            for device in recent_status[0].temperature.keys():
                temps = [s.temperature.get(device, 0) for s in recent_status if device in s.temperature]
                if temps:
                    temp_data[device] = {
                        "avg": sum(temps) / len(temps),
                        "max": max(temps),
                        "min": min(temps)
                    }
            report["temperature"] = temp_data
        
        if recent_status[0].power_usage:
            power_data = {}
            for device in recent_status[0].power_usage.keys():
                powers = [s.power_usage.get(device, 0) for s in recent_status if device in s.power_usage]
                if powers:
                    power_data[device] = {
                        "avg": sum(powers) / len(powers),
                        "max": max(powers),
                        "min": min(powers)
                    }
            report["power_usage"] = power_data
        
        return report
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        获取性能仪表板数据
        
        Returns:
            仪表板数据
        """
        return {
            "system_status": self.get_system_status().dict() if self.get_system_status() else None,
            "model_performance": self.get_model_performance_summary(),
            "hardware_utilization": self.get_hardware_utilization_report(),
            "bottleneck_analysis": self.analyze_performance_bottlenecks(),
            "total_metrics_collected": len(self.metrics_history),
            "monitoring_active": self.monitoring
        }


# 全局性能监控器实例
performance_monitor = PerformanceMonitor()
