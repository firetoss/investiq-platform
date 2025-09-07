"""
通用工具函数 - 日志、配置、验证等
"""

import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps

import structlog
from pydantic import BaseModel


class LogConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "json"  # json | human
    add_timestamp: bool = True
    add_caller: bool = False


def setup_logging(config: LogConfig = LogConfig()) -> None:
    """配置结构化日志"""
    
    # 配置Python标准logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.level.upper())
    )
    
    # 配置structlog
    processors = [
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.add_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="ISO"))
    
    if config.add_caller:
        processors.append(structlog.processors.CallsiteParameterAdder())
    
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """获取结构化日志器"""
    return structlog.get_logger(name or __name__)


def performance_timer(logger: Optional[structlog.BoundLogger] = None):
    """性能计时装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            log = logger or get_logger(func.__module__)
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                log.info(
                    "Function completed",
                    function=func.__name__,
                    duration_seconds=round(duration, 4),
                    args_count=len(args),
                    kwargs_count=len(kwargs)
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    "Function failed",
                    function=func.__name__,
                    duration_seconds=round(duration, 4),
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            log = logger or get_logger(func.__module__)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log.info(
                    "Function completed",
                    function=func.__name__,
                    duration_seconds=round(duration, 4)
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                log.error(
                    "Function failed", 
                    function=func.__name__,
                    duration_seconds=round(duration, 4),
                    error=str(e)
                )
                raise
        
        # 检查是否是协程函数
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ConfigManager:
    """配置管理器 - 环境变量和默认值"""
    
    def __init__(self, prefix: str = "INVESTIQ"):
        self.prefix = prefix
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """获取配置值"""
        env_key = f"{self.prefix}_{key.upper()}"
        
        if env_key in self._cache:
            return self._cache[env_key]
        
        import os
        value = os.getenv(env_key, default)
        
        if value is not None and cast_type != str:
            try:
                if cast_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                else:
                    value = cast_type(value)
            except (ValueError, TypeError):
                value = default
        
        self._cache[env_key] = value
        return value
    
    def get_service_url(self, service_name: str) -> str:
        """获取服务URL"""
        return self.get(f"{service_name.upper()}_SERVICE_URL", f"http://{service_name}-service:8000")


def validate_symbol(symbol: str) -> bool:
    """验证股票代码格式"""
    if not symbol or not isinstance(symbol, str):
        return False
    
    # 基础格式检查
    symbol = symbol.strip().upper()
    
    # 长度检查
    if not (1 <= len(symbol) <= 10):
        return False
    
    # 字符检查 - 只允许字母、数字、点号
    if not all(c.isalnum() or c == '.' for c in symbol):
        return False
    
    return True


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """文本清理和截断"""
    if not text:
        return ""
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # 截断长度
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()


def generate_request_id() -> str:
    """生成请求ID"""
    import uuid
    return f"req_{int(time.time())}_{uuid.uuid4().hex[:8]}"


class RateLimiter:
    """简单的内存速率限制器"""
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
    
    def allow(self, identifier: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # 清理过期请求
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # 检查是否超限
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # 记录请求
        self.requests[identifier].append(now)
        return True


def create_error_response(message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
    """创建标准错误响应"""
    return {
        "success": False,
        "message": message,
        "error_code": error_code,
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """创建标准成功响应"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }