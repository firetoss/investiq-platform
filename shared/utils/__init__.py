"""
InvestIQ Platform - 共享工具模块
"""

from .api import (
    create_app,
    setup_exception_handlers,
    get_request_id,
    create_response,
    create_error_response,
    HealthCheck,
    setup_health_check,
    RequestIDMiddleware,
    LoggingMiddleware
)

from .database import (
    DatabaseManager,
    db_manager,
    init_database,
    get_db,
    get_db_session,
    Repository,
    TransactionalService,
    check_database_health,
    database_health_check
)

from .redis import (
    RedisManager,
    redis_manager,
    init_redis,
    get_redis,
    redis_health_check
)

__all__ = [
    # API utilities
    "create_app",
    "setup_exception_handlers", 
    "get_request_id",
    "create_response",
    "create_error_response",
    "HealthCheck",
    "setup_health_check",
    "RequestIDMiddleware",
    "LoggingMiddleware",
    
    # Database utilities
    "DatabaseManager",
    "db_manager", 
    "init_database",
    "get_db",
    "get_db_session",
    "Repository",
    "TransactionalService",
    "check_database_health",
    "database_health_check",
    
    # Redis utilities  
    "RedisManager",
    "redis_manager",
    "init_redis", 
    "get_redis",
    "redis_health_check"
]