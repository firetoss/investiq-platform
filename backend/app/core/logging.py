"""
InvestIQ Platform - 日志配置模块
配置结构化日志和日志处理
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from backend.app.core.config import settings


def setup_logging():
    """设置应用日志配置"""
    
    # 配置标准库logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(message)s",
        handlers=[],
    )
    
    # 创建日志目录
    if settings.LOG_FILE:
        log_path = Path(settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _add_request_context,
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # 获取根logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    if settings.LOG_FORMAT == "json":
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果配置了日志文件）
    if settings.LOG_FILE:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=settings.LOG_FILE,
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 设置第三方库日志级别
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    
    # GPU相关库日志级别
    logging.getLogger("cupy").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def _add_request_context(logger, method_name, event_dict):
    """添加请求上下文到日志"""
    import contextvars
    
    # 尝试获取请求上下文
    try:
        request_id = contextvars.copy_context().get("request_id", None)
        if request_id:
            event_dict["request_id"] = request_id
    except Exception:
        pass
    
    return event_dict


class InvestIQLogger:
    """InvestIQ专用日志器"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def log_api_request(
        self, 
        method: str, 
        path: str, 
        status_code: int, 
        duration: float,
        **kwargs
    ):
        """记录API请求日志"""
        self.logger.info(
            "API request",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            **kwargs
        )
    
    def log_scoring_event(
        self, 
        event_type: str, 
        entity_type: str, 
        entity_id: str,
        score: float = None,
        **kwargs
    ):
        """记录评分事件日志"""
        self.logger.info(
            "Scoring event",
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            score=score,
            **kwargs
        )
    
    def log_gpu_operation(
        self, 
        operation: str, 
        duration: float = None,
        memory_used: int = None,
        **kwargs
    ):
        """记录GPU操作日志"""
        self.logger.info(
            "GPU operation",
            operation=operation,
            duration_ms=round(duration * 1000, 2) if duration else None,
            memory_used_mb=memory_used // 1024**2 if memory_used else None,
            **kwargs
        )
    
    def log_portfolio_event(
        self, 
        event_type: str, 
        portfolio_id: str = None,
        ticker: str = None,
        action: str = None,
        **kwargs
    ):
        """记录投资组合事件日志"""
        self.logger.info(
            "Portfolio event",
            event_type=event_type,
            portfolio_id=portfolio_id,
            ticker=ticker,
            action=action,
            **kwargs
        )
    
    def log_alert_event(
        self, 
        alert_type: str, 
        severity: str,
        entity_type: str = None,
        entity_id: str = None,
        **kwargs
    ):
        """记录告警事件日志"""
        self.logger.warning(
            "Alert triggered",
            alert_type=alert_type,
            severity=severity,
            entity_type=entity_type,
            entity_id=entity_id,
            **kwargs
        )
    
    def log_data_ingestion(
        self, 
        source: str, 
        data_type: str,
        records_count: int = None,
        success: bool = True,
        **kwargs
    ):
        """记录数据摄取日志"""
        level = "info" if success else "error"
        getattr(self.logger, level)(
            "Data ingestion",
            source=source,
            data_type=data_type,
            records_count=records_count,
            success=success,
            **kwargs
        )
    
    def log_ai_inference(
        self, 
        model_name: str, 
        input_tokens: int = None,
        output_tokens: int = None,
        duration: float = None,
        **kwargs
    ):
        """记录AI推理日志"""
        self.logger.info(
            "AI inference",
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=round(duration * 1000, 2) if duration else None,
            **kwargs
        )


def get_logger(name: str) -> InvestIQLogger:
    """获取InvestIQ日志器实例"""
    return InvestIQLogger(name)


# 性能监控装饰器
def log_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        import time
        import asyncio
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Performance: {operation_name}",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    success=True,
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Performance: {operation_name} failed",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                    success=False,
                )
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Performance: {operation_name}",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    success=True,
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"Performance: {operation_name} failed",
                    function=func.__name__,
                    duration_ms=round(duration * 1000, 2),
                    error=str(e),
                    success=False,
                )
                
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 审计日志
class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def log_user_action(
        self, 
        user_id: str, 
        action: str, 
        resource: str,
        resource_id: str = None,
        details: Dict[str, Any] = None
    ):
        """记录用户操作审计日志"""
        self.logger.info(
            "User action",
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            details=details or {},
        )
    
    def log_system_event(
        self, 
        event_type: str, 
        component: str,
        details: Dict[str, Any] = None
    ):
        """记录系统事件审计日志"""
        self.logger.info(
            "System event",
            event_type=event_type,
            component=component,
            details=details or {},
        )
    
    def log_security_event(
        self, 
        event_type: str, 
        user_id: str = None,
        ip_address: str = None,
        details: Dict[str, Any] = None
    ):
        """记录安全事件审计日志"""
        self.logger.warning(
            "Security event",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
        )


# 创建全局审计日志器
audit_logger = AuditLogger()
