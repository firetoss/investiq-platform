"""
InvestIQ Platform - 核心配置模块
管理应用的所有配置参数
"""

import os
from typing import List, Optional, Union
from pydantic import BaseSettings, validator, Field


class Settings(BaseSettings):
    """应用配置类"""
    
    # 基础应用配置
    APP_NAME: str = Field(default="InvestIQ Platform", env="APP_NAME")
    APP_VERSION: str = Field(default="0.1.0", env="APP_VERSION")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    SECRET_KEY: str = Field(default="dev-secret-key", env="SECRET_KEY")
    API_V1_STR: str = Field(default="/api/v1", env="API_V1_STR")
    
    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE: Optional[str] = Field(default=None, env="LOG_FILE")
    LOG_ROTATION: str = Field(default="1 day", env="LOG_ROTATION")
    LOG_RETENTION: str = Field(default="30 days", env="LOG_RETENTION")
    
    # 数据库配置
    DATABASE_URL: str = Field(
        default="postgresql://investiq:investiq123@localhost:5432/investiq",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis配置
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_CACHE_TTL: int = Field(default=3600, env="REDIS_CACHE_TTL")
    
    # MinIO对象存储配置
    MINIO_URL: str = Field(default="http://localhost:9000", env="MINIO_URL")
    MINIO_ACCESS_KEY: str = Field(default="investiq", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="investiq123", env="MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME: str = Field(default="investiq-data", env="MINIO_BUCKET_NAME")
    
    # GPU和CUDA配置
    CUDA_VISIBLE_DEVICES: str = Field(default="0", env="CUDA_VISIBLE_DEVICES")
    GPU_MEMORY_FRACTION: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    ENABLE_GPU_ACCELERATION: bool = Field(default=True, env="ENABLE_GPU_ACCELERATION")
    
    # AI模型配置
    MODEL_CACHE_DIR: str = Field(default="/app/models", env="MODEL_CACHE_DIR")
    HUGGINGFACE_CACHE_DIR: str = Field(
        default="/app/models/huggingface", 
        env="HUGGINGFACE_CACHE_DIR"
    )
    TRANSFORMERS_CACHE: str = Field(
        default="/app/models/transformers", 
        env="TRANSFORMERS_CACHE"
    )
    
    # LLM配置
    LLM_MODEL_NAME: str = Field(default="Qwen3-8B-INT8", env="LLM_MODEL_NAME")
    LLM_MODEL_PATH: str = Field(default="/models/Qwen3-8B-INT8.gguf", env="LLM_MODEL_PATH")
    LLM_MAX_TOKENS: int = Field(default=2048, env="LLM_MAX_TOKENS")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LLM_TEMPERATURE")
    LLM_TOP_P: float = Field(default=0.9, env="LLM_TOP_P")
    ENABLE_LOCAL_LLM: bool = Field(default=True, env="ENABLE_LOCAL_LLM")
    
    # 金融数据源配置
    ENABLE_YFINANCE: bool = Field(default=True, env="ENABLE_YFINANCE")
    ENABLE_AKSHARE: bool = Field(default=True, env="ENABLE_AKSHARE")
    DATA_UPDATE_INTERVAL: int = Field(default=300, env="DATA_UPDATE_INTERVAL")  # 5分钟
    
    # 评分引擎配置
    INDUSTRY_SCORE_WEIGHTS_P: float = Field(default=0.35, env="INDUSTRY_SCORE_WEIGHTS_P")
    INDUSTRY_SCORE_WEIGHTS_E: float = Field(default=0.25, env="INDUSTRY_SCORE_WEIGHTS_E")
    INDUSTRY_SCORE_WEIGHTS_M: float = Field(default=0.25, env="INDUSTRY_SCORE_WEIGHTS_M")
    INDUSTRY_SCORE_WEIGHTS_R: float = Field(default=0.15, env="INDUSTRY_SCORE_WEIGHTS_R")
    
    EQUITY_SCORE_WEIGHTS_Q: float = Field(default=0.30, env="EQUITY_SCORE_WEIGHTS_Q")
    EQUITY_SCORE_WEIGHTS_V: float = Field(default=0.20, env="EQUITY_SCORE_WEIGHTS_V")
    EQUITY_SCORE_WEIGHTS_M: float = Field(default=0.25, env="EQUITY_SCORE_WEIGHTS_M")
    EQUITY_SCORE_WEIGHTS_C: float = Field(default=0.15, env="EQUITY_SCORE_WEIGHTS_C")
    EQUITY_SCORE_WEIGHTS_S: float = Field(default=0.10, env="EQUITY_SCORE_WEIGHTS_S")
    
    # 四闸门阈值配置
    GATE_INDUSTRY_THRESHOLD: float = Field(default=70.0, env="GATE_INDUSTRY_THRESHOLD")
    GATE_EQUITY_THRESHOLD: float = Field(default=70.0, env="GATE_EQUITY_THRESHOLD")
    GATE_VALUATION_PERCENTILE_MAX: float = Field(default=0.7, env="GATE_VALUATION_PERCENTILE_MAX")
    GATE_GROWTH_PERCENTILE_MAX: float = Field(default=0.8, env="GATE_GROWTH_PERCENTILE_MAX")
    GATE_PEG_MAX: float = Field(default=1.5, env="GATE_PEG_MAX")
    
    # 流动性配置
    LIQUIDITY_PARTICIPATION_RATE_A: float = Field(default=0.10, env="LIQUIDITY_PARTICIPATION_RATE_A")
    LIQUIDITY_PARTICIPATION_RATE_H: float = Field(default=0.08, env="LIQUIDITY_PARTICIPATION_RATE_H")
    LIQUIDITY_EXIT_DAYS_CORE: int = Field(default=5, env="LIQUIDITY_EXIT_DAYS_CORE")
    LIQUIDITY_EXIT_DAYS_TACTICAL: int = Field(default=3, env="LIQUIDITY_EXIT_DAYS_TACTICAL")
    LIQUIDITY_ADV_MIN_A: int = Field(default=30000000, env="LIQUIDITY_ADV_MIN_A")
    LIQUIDITY_ADV_MIN_H: int = Field(default=20000000, env="LIQUIDITY_ADV_MIN_H")
    LIQUIDITY_TURNOVER_MIN_A: float = Field(default=0.005, env="LIQUIDITY_TURNOVER_MIN_A")
    LIQUIDITY_TURNOVER_MIN_H: float = Field(default=0.003, env="LIQUIDITY_TURNOVER_MIN_H")
    
    # 投资组合配置
    PORTFOLIO_LEVERAGE_MAX: float = Field(default=1.10, env="PORTFOLIO_LEVERAGE_MAX")
    PORTFOLIO_TIER_A_MIN: float = Field(default=0.12, env="PORTFOLIO_TIER_A_MIN")
    PORTFOLIO_TIER_A_MAX: float = Field(default=0.15, env="PORTFOLIO_TIER_A_MAX")
    PORTFOLIO_TIER_B_MIN: float = Field(default=0.08, env="PORTFOLIO_TIER_B_MIN")
    PORTFOLIO_TIER_B_MAX: float = Field(default=0.10, env="PORTFOLIO_TIER_B_MAX")
    PORTFOLIO_TIER_C_MIN: float = Field(default=0.03, env="PORTFOLIO_TIER_C_MIN")
    PORTFOLIO_TIER_C_MAX: float = Field(default=0.05, env="PORTFOLIO_TIER_C_MAX")
    
    # 回撤断路器配置
    CIRCUIT_BREAKER_LEVEL_1: float = Field(default=-0.10, env="CIRCUIT_BREAKER_LEVEL_1")
    CIRCUIT_BREAKER_LEVEL_2: float = Field(default=-0.20, env="CIRCUIT_BREAKER_LEVEL_2")
    CIRCUIT_BREAKER_LEVEL_3: float = Field(default=-0.30, env="CIRCUIT_BREAKER_LEVEL_3")
    
    # 告警配置
    ALERT_THROTTLE_EVENT_MINUTES: int = Field(default=60, env="ALERT_THROTTLE_EVENT_MINUTES")
    ALERT_THROTTLE_KPI_MINUTES: int = Field(default=1440, env="ALERT_THROTTLE_KPI_MINUTES")
    ALERT_THROTTLE_TREND_DAYS: int = Field(default=5, env="ALERT_THROTTLE_TREND_DAYS")
    ALERT_ESCALATION_P3_TO_P2_HITS: int = Field(default=3, env="ALERT_ESCALATION_P3_TO_P2_HITS")
    ALERT_ESCALATION_P2_TO_P1_HITS: int = Field(default=2, env="ALERT_ESCALATION_P2_TO_P1_HITS")
    
    # 监控配置
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=9090, env="PROMETHEUS_PORT")
    GRAFANA_ENABLED: bool = Field(default=True, env="GRAFANA_ENABLED")
    GRAFANA_PORT: int = Field(default=3001, env="GRAFANA_PORT")
    
    # Celery配置
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    CELERY_TASK_SERIALIZER: str = Field(default="json", env="CELERY_TASK_SERIALIZER")
    CELERY_RESULT_SERIALIZER: str = Field(default="json", env="CELERY_RESULT_SERIALIZER")
    CELERY_ACCEPT_CONTENT: List[str] = Field(default=["json"], env="CELERY_ACCEPT_CONTENT")
    CELERY_TIMEZONE: str = Field(default="Asia/Shanghai", env="CELERY_TIMEZONE")
    
    # 安全配置
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"],
        env="ALLOWED_HOSTS"
    )
    
    # 开发工具配置
    ENABLE_DOCS: bool = Field(default=True, env="ENABLE_DOCS")
    ENABLE_REDOC: bool = Field(default=True, env="ENABLE_REDOC")
    ENABLE_OPENAPI: bool = Field(default=True, env="ENABLE_OPENAPI")
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """解析CORS origins"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """解析允许的主机"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("CELERY_ACCEPT_CONTENT", pre=True)
    def assemble_celery_accept_content(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """解析Celery接受的内容类型"""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v: str, values: dict) -> str:
        """验证密钥安全性"""
        if values.get("ENVIRONMENT") == "production" and v == "dev-secret-key":
            raise ValueError("生产环境必须设置安全的SECRET_KEY")
        if len(v) < 32:
            raise ValueError("SECRET_KEY长度至少32个字符")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v: str) -> str:
        """验证数据库URL"""
        if not v.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise ValueError("仅支持PostgreSQL数据库")
        return v
    
    @property
    def industry_score_weights(self) -> dict:
        """行业评分权重"""
        return {
            "P": self.INDUSTRY_SCORE_WEIGHTS_P,
            "E": self.INDUSTRY_SCORE_WEIGHTS_E,
            "M": self.INDUSTRY_SCORE_WEIGHTS_M,
            "R": self.INDUSTRY_SCORE_WEIGHTS_R,
        }
    
    @property
    def equity_score_weights(self) -> dict:
        """个股评分权重"""
        return {
            "Q": self.EQUITY_SCORE_WEIGHTS_Q,
            "V": self.EQUITY_SCORE_WEIGHTS_V,
            "M": self.EQUITY_SCORE_WEIGHTS_M,
            "C": self.EQUITY_SCORE_WEIGHTS_C,
            "S": self.EQUITY_SCORE_WEIGHTS_S,
        }
    
    @property
    def gate_thresholds(self) -> dict:
        """四闸门阈值"""
        return {
            "industry": self.GATE_INDUSTRY_THRESHOLD,
            "equity": self.GATE_EQUITY_THRESHOLD,
            "valuation_percentile_max": self.GATE_VALUATION_PERCENTILE_MAX,
            "growth_percentile_max": self.GATE_GROWTH_PERCENTILE_MAX,
            "peg_max": self.GATE_PEG_MAX,
        }
    
    @property
    def liquidity_config(self) -> dict:
        """流动性配置"""
        return {
            "participation_rate": {
                "A": self.LIQUIDITY_PARTICIPATION_RATE_A,
                "H": self.LIQUIDITY_PARTICIPATION_RATE_H,
            },
            "exit_days": {
                "core": self.LIQUIDITY_EXIT_DAYS_CORE,
                "tactical": self.LIQUIDITY_EXIT_DAYS_TACTICAL,
            },
            "adv_min": {
                "A": self.LIQUIDITY_ADV_MIN_A,
                "H": self.LIQUIDITY_ADV_MIN_H,
            },
            "turnover_min": {
                "A": self.LIQUIDITY_TURNOVER_MIN_A,
                "H": self.LIQUIDITY_TURNOVER_MIN_H,
            },
        }
    
    @property
    def portfolio_tiers(self) -> dict:
        """投资组合分层配置"""
        return {
            "A": {"min": self.PORTFOLIO_TIER_A_MIN, "max": self.PORTFOLIO_TIER_A_MAX},
            "B": {"min": self.PORTFOLIO_TIER_B_MIN, "max": self.PORTFOLIO_TIER_B_MAX},
            "C": {"min": self.PORTFOLIO_TIER_C_MIN, "max": self.PORTFOLIO_TIER_C_MAX},
        }
    
    @property
    def circuit_breaker_levels(self) -> dict:
        """回撤断路器级别"""
        return {
            "level_1": self.CIRCUIT_BREAKER_LEVEL_1,
            "level_2": self.CIRCUIT_BREAKER_LEVEL_2,
            "level_3": self.CIRCUIT_BREAKER_LEVEL_3,
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 创建全局设置实例
settings = Settings()

# 设置环境变量 (用于其他库)
os.environ["CUDA_VISIBLE_DEVICES"] = settings.CUDA_VISIBLE_DEVICES
os.environ["TRANSFORMERS_CACHE"] = settings.TRANSFORMERS_CACHE
os.environ["HF_HOME"] = settings.HUGGINGFACE_CACHE_DIR
