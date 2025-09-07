"""
主API服务配置 - 精简版本，只包含必要配置
"""

import os
from typing import List
from pydantic import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    PROJECT_NAME: str = "InvestIQ Main API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("FASTAPI_ENV", "production")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # API配置
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # 数据库配置
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://investiq:investiq123@postgres:5432/investiq"
    )
    
    # Redis配置
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # 微服务URLs
    LLM_SERVICE_URL: str = os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
    SENTIMENT_SERVICE_URL: str = os.getenv("SENTIMENT_SERVICE_URL", "http://sentiment-service:8002")
    TIMESERIES_SERVICE_URL: str = os.getenv("TIMESERIES_SERVICE_URL", "http://timeseries-service:8003")
    
    # JWT配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()