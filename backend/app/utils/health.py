"""
InvestIQ Platform - 健康检查工具模块
系统健康状态监控和检查
"""

import asyncio
import time
from typing import Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.database import db_manager
from backend.app.core.redis import redis_client
from backend.app.utils.gpu import check_gpu_availability, get_gpu_info


class HealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.checks = {
            "database": self._check_database,
            "redis": self._check_redis,
            "gpu": self._check_gpu,
            "disk": self._check_disk,
            "memory": self._check_memory,
        }
    
    async def check_all(self) -> Dict:
        """执行所有健康检查"""
        start_time = time.time()
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[check_name] = result
                
                if not result.get("healthy", False):
                    overall_status = "unhealthy"
                    
            except Exception as e:
                results[check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": time.time(),
                }
                overall_status = "unhealthy"
        
        duration = time.time() - start_time
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "duration_ms": round(duration * 1000, 2),
            "checks": results,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }
    
    async def check_single(self, check_name: str) -> Dict:
        """执行单个健康检查"""
        if check_name not in self.checks:
            return {
                "healthy": False,
                "error": f"Unknown health check: {check_name}",
                "timestamp": time.time(),
            }
        
        try:
            return await self.checks[check_name]()
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def _check_database(self) -> Dict:
        """检查数据库连接"""
        start_time = time.time()
        
        try:
            # 检查数据库连接
            is_healthy = await db_manager.health_check()
            
            if is_healthy:
                # 获取数据库信息
                db_info = await db_manager.get_connection_info()
                duration = time.time() - start_time
                
                return {
                    "healthy": True,
                    "response_time_ms": round(duration * 1000, 2),
                    "details": db_info,
                    "timestamp": time.time(),
                }
            else:
                return {
                    "healthy": False,
                    "error": "Database connection failed",
                    "timestamp": time.time(),
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def _check_redis(self) -> Dict:
        """检查Redis连接"""
        start_time = time.time()
        
        try:
            # 检查Redis连接
            is_healthy = await redis_client.ping()
            
            if is_healthy:
                # 获取Redis信息
                redis_info = await redis_client.info()
                duration = time.time() - start_time
                
                return {
                    "healthy": True,
                    "response_time_ms": round(duration * 1000, 2),
                    "details": {
                        "version": redis_info.get("redis_version"),
                        "used_memory": redis_info.get("used_memory_human"),
                        "connected_clients": redis_info.get("connected_clients"),
                        "uptime_seconds": redis_info.get("uptime_in_seconds"),
                    },
                    "timestamp": time.time(),
                }
            else:
                return {
                    "healthy": False,
                    "error": "Redis connection failed",
                    "timestamp": time.time(),
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def _check_gpu(self) -> Dict:
        """检查GPU状态"""
        try:
            gpu_available = check_gpu_availability()
            
            if gpu_available:
                gpu_info = get_gpu_info()
                
                # 检查GPU内存使用率
                memory_utilization = gpu_info.get("memory_utilization", 0)
                is_healthy = memory_utilization < 0.95  # 内存使用率低于95%
                
                return {
                    "healthy": is_healthy,
                    "details": {
                        "device_id": gpu_info.get("device_id"),
                        "memory_total_mb": gpu_info.get("memory_total", 0) // 1024**2,
                        "memory_used_mb": gpu_info.get("memory_used", 0) // 1024**2,
                        "memory_utilization": round(memory_utilization, 3),
                        "compute_capability": gpu_info.get("compute_capability"),
                    },
                    "warning": "High GPU memory usage" if memory_utilization > 0.8 else None,
                    "timestamp": time.time(),
                }
            else:
                return {
                    "healthy": not settings.ENABLE_GPU_ACCELERATION,  # 如果禁用GPU则认为健康
                    "details": {"available": False},
                    "warning": "GPU not available" if settings.ENABLE_GPU_ACCELERATION else None,
                    "timestamp": time.time(),
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def _check_disk(self) -> Dict:
        """检查磁盘空间"""
        try:
            import shutil
            
            # 检查根目录磁盘空间
            total, used, free = shutil.disk_usage("/")
            
            # 计算使用率
            usage_percent = used / total
            is_healthy = usage_percent < 0.9  # 磁盘使用率低于90%
            
            return {
                "healthy": is_healthy,
                "details": {
                    "total_gb": round(total / 1024**3, 2),
                    "used_gb": round(used / 1024**3, 2),
                    "free_gb": round(free / 1024**3, 2),
                    "usage_percent": round(usage_percent, 3),
                },
                "warning": "High disk usage" if usage_percent > 0.8 else None,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def _check_memory(self) -> Dict:
        """检查内存使用"""
        try:
            import psutil
            
            # 获取内存信息
            memory = psutil.virtual_memory()
            
            # 检查内存使用率
            is_healthy = memory.percent < 90  # 内存使用率低于90%
            
            return {
                "healthy": is_healthy,
                "details": {
                    "total_gb": round(memory.total / 1024**3, 2),
                    "used_gb": round(memory.used / 1024**3, 2),
                    "available_gb": round(memory.available / 1024**3, 2),
                    "usage_percent": memory.percent,
                },
                "warning": "High memory usage" if memory.percent > 80 else None,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }


class ServiceMonitor:
    """服务监控器"""
    
    def __init__(self):
        self.health_checker = HealthChecker()
        self.metrics = {}
    
    async def get_system_metrics(self) -> Dict:
        """获取系统指标"""
        try:
            import psutil
            
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage("/")
            
            # 网络信息
            network = psutil.net_io_counters()
            
            # 进程信息
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                },
                "memory": {
                    "total_gb": round(memory.total / 1024**3, 2),
                    "used_gb": round(memory.used / 1024**3, 2),
                    "usage_percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / 1024**3, 2),
                    "used_gb": round(disk.used / 1024**3, 2),
                    "usage_percent": round(disk.used / disk.total * 100, 2),
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv,
                },
                "process": {
                    "memory_mb": round(process_memory.rss / 1024**2, 2),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                },
                "timestamp": time.time(),
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def get_application_metrics(self) -> Dict:
        """获取应用指标"""
        try:
            # 数据库连接池状态
            db_info = await db_manager.get_connection_info()
            
            # Redis信息
            redis_info = await redis_client.info()
            
            # GPU信息（如果可用）
            gpu_info = None
            if check_gpu_availability():
                gpu_info = get_gpu_info()
            
            return {
                "database": {
                    "active_connections": db_info.get("active_connections", 0),
                    "pool_size": db_info.get("pool_size", 0),
                    "checked_out_connections": db_info.get("checked_out_connections", 0),
                },
                "redis": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_mb": redis_info.get("used_memory", 0) // 1024**2,
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                },
                "gpu": gpu_info if gpu_info and gpu_info.get("available") else None,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": time.time(),
            }
    
    async def check_dependencies(self) -> Dict:
        """检查外部依赖"""
        dependencies = {}
        
        # 检查数据库
        try:
            db_healthy = await db_manager.health_check()
            dependencies["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "url": settings.DATABASE_URL.split("@")[-1],  # 隐藏密码
            }
        except Exception as e:
            dependencies["database"] = {
                "status": "error",
                "error": str(e),
            }
        
        # 检查Redis
        try:
            redis_healthy = await redis_client.ping()
            dependencies["redis"] = {
                "status": "healthy" if redis_healthy else "unhealthy",
                "url": settings.REDIS_URL.split("@")[-1],  # 隐藏密码
            }
        except Exception as e:
            dependencies["redis"] = {
                "status": "error",
                "error": str(e),
            }
        
        return {
            "dependencies": dependencies,
            "timestamp": time.time(),
        }


# 创建全局健康检查器和监控器实例
health_checker = HealthChecker()
service_monitor = ServiceMonitor()


async def quick_health_check() -> bool:
    """快速健康检查，返回布尔值"""
    try:
        # 只检查关键服务
        db_healthy = await db_manager.health_check()
        redis_healthy = await redis_client.ping()
        
        return db_healthy and redis_healthy
    except Exception:
        return False


async def detailed_health_report() -> Dict:
    """详细健康报告"""
    health_status = await health_checker.check_all()
    system_metrics = await service_monitor.get_system_metrics()
    app_metrics = await service_monitor.get_application_metrics()
    dependencies = await service_monitor.check_dependencies()
    
    return {
        "health": health_status,
        "system": system_metrics,
        "application": app_metrics,
        "dependencies": dependencies,
        "generated_at": time.time(),
    }
