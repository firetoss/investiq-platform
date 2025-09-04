"""
InvestIQ Platform - 异常处理模块
定义自定义异常类和错误处理
"""

from typing import Any, Dict, Optional


class InvestIQException(Exception):
    """InvestIQ平台基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "INVESTIQ_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationException(InvestIQException):
    """数据验证异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class DatabaseException(InvestIQException):
    """数据库操作异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=503,
            details=details,
        )


class GPUException(InvestIQException):
    """GPU计算异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="GPU_ERROR",
            status_code=503,
            details=details,
        )


class ScoringException(InvestIQException):
    """评分计算异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SCORING_ERROR",
            status_code=422,
            details=details,
        )


class GatekeeperException(InvestIQException):
    """四闸门校验异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="GATEKEEPER_ERROR",
            status_code=422,
            details=details,
        )


class LiquidityException(InvestIQException):
    """流动性检查异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="LIQUIDITY_ERROR",
            status_code=422,
            details=details,
        )


class PortfolioException(InvestIQException):
    """投资组合异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="PORTFOLIO_ERROR",
            status_code=422,
            details=details,
        )


class AIException(InvestIQException):
    """AI模型异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AI_ERROR",
            status_code=503,
            details=details,
        )


class DataSourceException(InvestIQException):
    """数据源异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATA_SOURCE_ERROR",
            status_code=503,
            details=details,
        )


class AuthenticationException(InvestIQException):
    """认证异常"""
    
    def __init__(self, message: str = "认证失败", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class AuthorizationException(InvestIQException):
    """授权异常"""
    
    def __init__(self, message: str = "权限不足", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
        )


class RateLimitException(InvestIQException):
    """速率限制异常"""
    
    def __init__(self, message: str = "请求过于频繁", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            status_code=429,
            details=details,
        )


class ConfigurationException(InvestIQException):
    """配置异常"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class ExternalServiceException(InvestIQException):
    """外部服务异常"""
    
    def __init__(self, message: str, service_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service_name"] = service_name
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=503,
            details=details,
        )


# 错误码映射
ERROR_CODE_MAPPING = {
    "VALIDATION_ERROR": "数据验证失败",
    "DATABASE_ERROR": "数据库操作失败",
    "GPU_ERROR": "GPU计算失败",
    "SCORING_ERROR": "评分计算失败",
    "GATEKEEPER_ERROR": "四闸门校验失败",
    "LIQUIDITY_ERROR": "流动性检查失败",
    "PORTFOLIO_ERROR": "投资组合操作失败",
    "AI_ERROR": "AI模型处理失败",
    "DATA_SOURCE_ERROR": "数据源访问失败",
    "AUTHENTICATION_ERROR": "认证失败",
    "AUTHORIZATION_ERROR": "权限不足",
    "RATE_LIMIT_ERROR": "请求频率超限",
    "CONFIGURATION_ERROR": "配置错误",
    "EXTERNAL_SERVICE_ERROR": "外部服务不可用",
}


def get_error_message(error_code: str) -> str:
    """根据错误码获取错误消息"""
    return ERROR_CODE_MAPPING.get(error_code, "未知错误")
