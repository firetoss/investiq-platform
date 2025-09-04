"""
InvestIQ Platform - GPU工具模块
GPU可用性检查和管理工具
"""

import logging
from typing import Dict, Optional, Tuple
from backend.app.core.config import settings
from backend.app.core.exceptions import GPUException


def check_gpu_availability() -> bool:
    """检查GPU是否可用"""
    try:
        import cupy as cp
        # 尝试创建一个简单的数组来测试GPU
        test_array = cp.array([1, 2, 3])
        return True
    except Exception as e:
        logging.warning(f"GPU not available: {e}")
        return False


def get_gpu_info() -> Dict:
    """获取GPU信息"""
    try:
        import cupy as cp
        
        device = cp.cuda.Device()
        mem_info = device.mem_info
        
        return {
            "available": True,
            "device_id": device.id,
            "device_count": cp.cuda.runtime.getDeviceCount(),
            "memory_total": mem_info[1],
            "memory_free": mem_info[0],
            "memory_used": mem_info[1] - mem_info[0],
            "memory_utilization": (mem_info[1] - mem_info[0]) / mem_info[1],
            "compute_capability": device.compute_capability,
        }
    except Exception as e:
        logging.error(f"Failed to get GPU info: {e}")
        return {"available": False, "error": str(e)}


def set_gpu_memory_fraction(fraction: float = None) -> bool:
    """设置GPU内存使用比例"""
    try:
        import cupy as cp
        
        fraction = fraction or settings.GPU_MEMORY_FRACTION
        
        # CuPy会自动管理内存，这里主要是设置内存池
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=int(get_gpu_info()["memory_total"] * fraction))
        
        logging.info(f"GPU memory fraction set to {fraction}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to set GPU memory fraction: {e}")
        return False


def clear_gpu_memory() -> bool:
    """清理GPU内存"""
    try:
        import cupy as cp
        import gc
        
        # 清理CuPy内存池
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        
        # 强制垃圾回收
        gc.collect()
        
        logging.info("GPU memory cleared")
        return True
        
    except Exception as e:
        logging.error(f"Failed to clear GPU memory: {e}")
        return False


class GPUContext:
    """GPU上下文管理器"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.previous_device = None
        
    def __enter__(self):
        try:
            import cupy as cp
            self.previous_device = cp.cuda.Device().id
            cp.cuda.Device(self.device_id).use()
            return self
        except Exception as e:
            raise GPUException(f"Failed to set GPU device {self.device_id}: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import cupy as cp
            if self.previous_device is not None:
                cp.cuda.Device(self.previous_device).use()
        except Exception as e:
            logging.error(f"Failed to restore GPU device: {e}")


class GPUMemoryMonitor:
    """GPU内存监控器"""
    
    def __init__(self):
        self.initial_memory = None
        
    def __enter__(self):
        if check_gpu_availability():
            gpu_info = get_gpu_info()
            self.initial_memory = gpu_info.get("memory_used", 0)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if check_gpu_availability() and self.initial_memory is not None:
            gpu_info = get_gpu_info()
            current_memory = gpu_info.get("memory_used", 0)
            memory_diff = current_memory - self.initial_memory
            
            if memory_diff > 0:
                logging.info(f"GPU memory usage increased by {memory_diff // 1024**2} MB")
    
    def get_current_usage(self) -> Dict:
        """获取当前内存使用情况"""
        if not check_gpu_availability():
            return {"available": False}
        
        gpu_info = get_gpu_info()
        return {
            "available": True,
            "memory_used_mb": gpu_info["memory_used"] // 1024**2,
            "memory_free_mb": gpu_info["memory_free"] // 1024**2,
            "memory_total_mb": gpu_info["memory_total"] // 1024**2,
            "utilization": gpu_info["memory_utilization"],
        }


def ensure_gpu_available():
    """确保GPU可用的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not settings.ENABLE_GPU_ACCELERATION:
                raise GPUException("GPU acceleration is disabled")
            
            if not check_gpu_availability():
                raise GPUException("GPU is not available")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def gpu_fallback(cpu_func):
    """GPU计算失败时回退到CPU的装饰器"""
    def decorator(gpu_func):
        def wrapper(*args, **kwargs):
            if not settings.ENABLE_GPU_ACCELERATION or not check_gpu_availability():
                logging.info(f"Using CPU fallback for {gpu_func.__name__}")
                return cpu_func(*args, **kwargs)
            
            try:
                return gpu_func(*args, **kwargs)
            except Exception as e:
                logging.warning(f"GPU computation failed, falling back to CPU: {e}")
                return cpu_func(*args, **kwargs)
        return wrapper
    return decorator


def optimize_gpu_performance():
    """优化GPU性能设置"""
    try:
        import cupy as cp
        
        # 设置内存池
        set_gpu_memory_fraction()
        
        # 启用内存池
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=None)  # 移除限制，让CuPy自动管理
        
        # 设置CUDA流
        stream = cp.cuda.Stream(non_blocking=True)
        
        logging.info("GPU performance optimized")
        return True
        
    except Exception as e:
        logging.error(f"Failed to optimize GPU performance: {e}")
        return False


def benchmark_gpu() -> Dict:
    """GPU性能基准测试"""
    try:
        import cupy as cp
        import time
        
        # 矩阵乘法基准测试
        size = 1000
        a = cp.random.random((size, size), dtype=cp.float32)
        b = cp.random.random((size, size), dtype=cp.float32)
        
        # 预热
        for _ in range(3):
            cp.dot(a, b)
        
        # 基准测试
        start_time = time.time()
        for _ in range(10):
            result = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        
        gpu_time = (end_time - start_time) / 10
        
        # CPU对比
        import numpy as np
        a_cpu = cp.asnumpy(a)
        b_cpu = cp.asnumpy(b)
        
        start_time = time.time()
        for _ in range(10):
            result_cpu = np.dot(a_cpu, b_cpu)
        end_time = time.time()
        
        cpu_time = (end_time - start_time) / 10
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        return {
            "gpu_time_ms": gpu_time * 1000,
            "cpu_time_ms": cpu_time * 1000,
            "speedup": speedup,
            "matrix_size": size,
        }
        
    except Exception as e:
        logging.error(f"GPU benchmark failed: {e}")
        return {"error": str(e)}


def get_cuda_version() -> Optional[str]:
    """获取CUDA版本"""
    try:
        import cupy as cp
        return cp.cuda.runtime.runtimeGetVersion()
    except Exception:
        return None


def get_cudnn_version() -> Optional[str]:
    """获取cuDNN版本"""
    try:
        import cupy as cp
        return cp.cuda.cudnn.getVersion()
    except Exception:
        return None


def diagnose_gpu() -> Dict:
    """GPU诊断信息"""
    diagnosis = {
        "gpu_available": check_gpu_availability(),
        "gpu_acceleration_enabled": settings.ENABLE_GPU_ACCELERATION,
        "cuda_version": get_cuda_version(),
        "cudnn_version": get_cudnn_version(),
    }
    
    if diagnosis["gpu_available"]:
        diagnosis.update(get_gpu_info())
        diagnosis["benchmark"] = benchmark_gpu()
    
    return diagnosis
