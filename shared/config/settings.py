"""
InvestIQ Platform - 配置管理
统一的应用配置管理，支持环境变量覆盖
"""

from pydantic import BaseSettings, Field
from typing import Optional, Dict, Any
import os


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    host: str = Field(default="postgres", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    name: str = Field(default="investiq", env="DATABASE_NAME")
    username: str = Field(default="investiq", env="DATABASE_USER")
    password: str = Field(default="password", env="DATABASE_PASSWORD")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis配置"""
    host: str = Field(default="redis", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    db: int = Field(default=0, env="REDIS_DB")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class MinIOSettings(BaseSettings):
    """MinIO对象存储配置"""
    host: str = Field(default="minio", env="MINIO_HOST")
    port: int = Field(default=9000, env="MINIO_PORT")
    access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin123", env="MINIO_SECRET_KEY")
    bucket_name: str = Field(default="investiq-evidence", env="MINIO_BUCKET")
    secure: bool = Field(default=False, env="MINIO_SECURE")
    
    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"


class AppSettings(BaseSettings):
    """应用基础配置"""
    # 应用信息
    name: str = "InvestIQ Platform"
    version: str = "1.0.0"
    description: str = "Industry Selection & Portfolio Support Platform"
    
    # 运行环境
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # 时区和货币
    timezone: str = Field(default="Asia/Shanghai", env="APP_TIMEZONE")
    currency_default: str = Field(default="CNY", env="APP_CURRENCY_DEFAULT")
    
    # API配置
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    
    # 安全配置
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24 hours
    
    # 跨域配置
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]


class ScoringSettings(BaseSettings):
    """评分系统配置"""
    # 行业评分权重
    industry_weights: Dict[str, float] = {
        "P": 0.35,  # 政策强度
        "E": 0.25,  # 落地证据
        "M": 0.25,  # 市场确认
        "R_neg": 0.15  # 风险扣分
    }
    
    # 个股评分权重
    equity_weights: Dict[str, float] = {
        "Q": 0.30,  # 质量
        "V": 0.20,  # 估值
        "M": 0.25,  # 动量
        "C": 0.15,  # 政策契合
        "S": 0.10   # 份额护城河
    }
    
    # 评分阈值
    industry_threshold_in_pool: float = 70.0
    industry_threshold_core: float = 75.0
    equity_threshold_observe: float = 65.0
    equity_threshold_build: float = 70.0


class GatekeeperSettings(BaseSettings):
    """闸门系统配置"""
    # 四闸门阈值
    industry_score_min: float = 70.0
    equity_score_min: float = 70.0
    valuation_percentile_max: float = 0.7  # 70%
    growth_valuation_percentile_max: float = 0.8  # 80%
    peg_max: float = 1.5
    above_200dma_required: bool = True
    
    # 估值分位计算
    valuation_window_years: int = 5


class LiquiditySettings(BaseSettings):
    """流动性配置"""
    # 参与率设置
    participation_rate: Dict[str, float] = {
        "A": 0.10,  # A股 10%
        "H": 0.08   # H股 8%
    }
    
    # 退出天数设置
    exit_days: Dict[str, int] = {
        "core": 5,     # 核心持仓 5 天
        "tactical": 3  # 战术持仓 3 天
    }
    
    # 绝对底线
    floors: Dict[str, Dict[str, Any]] = {
        "A": {
            "adv20_min": 30_000_000,  # 3000万人民币
            "turnover_min": 0.005     # 0.5%
        },
        "H": {
            "adv20_min": 20_000_000,  # 2000万港币
            "turnover_min": 0.003     # 0.3%
        }
    }
    
    # 自由流通占用上限
    free_float_cap_pct_max: float = 0.02  # 2%


class PortfolioSettings(BaseSettings):
    """组合管理配置"""
    # A/B/C 分层目标仓位
    tiers: Dict[str, list] = {
        "A": [0.12, 0.15],  # 12%-15%
        "B": [0.08, 0.10],  # 8%-10%
        "C": [0.03, 0.05]   # 3%-5%
    }
    
    # 杠杆配置
    leverage_max: float = 1.10  # 最大 110%
    
    # 回撤断路器
    circuit_breaker_levels: list = [
        {
            "drawdown": -0.10,
            "actions": ["remove_leverage", "pause_tactical"]
        },
        {
            "drawdown": -0.20,
            "actions": ["halve_tactical", "clear_watch", "tighten_entry"]
        },
        {
            "drawdown": -0.30,
            "actions": ["keep_top2_3", "cash_rest"]
        }
    ]


class AlertSettings(BaseSettings):
    """告警系统配置"""
    # 节流设置
    throttle: Dict[str, Any] = {
        "min_interval_minutes": 120,
        "aggregation_window_minutes": 15,
        "by_type": {
            "event": {
                "min_interval_minutes": 60,
                "aggregation_window_minutes": 15
            },
            "kpi": {
                "min_interval_minutes": 1440,  # 24小时
                "aggregation_window_minutes": 60
            },
            "trend": {
                "confirm_on_cross_rule": True,
                "min_interval_days": 5
            },
            "drawdown": {
                "latch": True
            }
        }
    }
    
    # 升级规则
    escalation: Dict[str, int] = {
        "p3_to_p2_hits": 3,
        "p2_to_p1_hits": 2
    }


class Settings(BaseSettings):
    """主配置类，包含所有子配置"""
    
    # 基础配置
    app: AppSettings = AppSettings()
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    minio: MinIOSettings = MinIOSettings()
    
    # 业务配置
    scoring: ScoringSettings = ScoringSettings()
    gatekeeper: GatekeeperSettings = GatekeeperSettings()
    liquidity: LiquiditySettings = LiquiditySettings()
    portfolio: PortfolioSettings = PortfolioSettings()
    alerts: AlertSettings = AlertSettings()
    
    # 服务端口配置
    service_port: int = Field(default=8000, env="SERVICE_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置实例
settings = Settings()


# 配置验证函数
def validate_settings() -> bool:
    """验证配置的有效性"""
    try:
        # 验证评分权重总和
        industry_weight_sum = sum(settings.scoring.industry_weights.values())
        if not (0.99 <= industry_weight_sum <= 1.01):  # 允许小的浮点误差
            raise ValueError(f"Industry scoring weights sum to {industry_weight_sum}, should be 1.0")
        
        equity_weight_sum = sum(settings.scoring.equity_weights.values())
        if not (0.99 <= equity_weight_sum <= 1.01):
            raise ValueError(f"Equity scoring weights sum to {equity_weight_sum}, should be 1.0")
        
        # 验证阈值范围
        if not (0 <= settings.gatekeeper.valuation_percentile_max <= 1):
            raise ValueError("Valuation percentile max should be between 0 and 1")
        
        # 验证参与率范围
        for market, rate in settings.liquidity.participation_rate.items():
            if not (0 < rate <= 1):
                raise ValueError(f"Participation rate for {market} should be between 0 and 1")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    # 测试配置加载
    print("Loading InvestIQ Platform configuration...")
    print(f"Database URL: {settings.database.url}")
    print(f"Redis URL: {settings.redis.url}")
    print(f"Service Port: {settings.service_port}")
    print(f"Debug Mode: {settings.app.debug}")
    
    # 验证配置
    if validate_settings():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
        exit(1)