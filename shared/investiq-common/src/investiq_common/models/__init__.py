"""
共享数据模型 - 基于现有backend/app/models重构
只包含核心数据结构，去除数据库依赖
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from decimal import Decimal

from pydantic import BaseModel, Field


class ScoreLevel(str, Enum):
    """评分等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    AVERAGE = "average"
    POOR = "poor"
    TERRIBLE = "terrible"


class SentimentType(str, Enum):
    """情感类型枚举"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


# =============== 核心业务模型 ===============

class EquityBase(BaseModel):
    """股票基础信息"""
    symbol: str = Field(..., description="股票代码")
    name: str = Field(..., description="股票名称")
    market: str = Field(..., description="市场")
    industry: Optional[str] = Field(None, description="行业")
    
    class Config:
        from_attributes = True


class EquityScore(BaseModel):
    """股票评分信息"""
    symbol: str
    total_score: float = Field(..., ge=0, le=100, description="总分")
    level: ScoreLevel = Field(..., description="评分等级")
    technical_score: Optional[float] = Field(None, description="技术分析分数")
    fundamental_score: Optional[float] = Field(None, description="基本面分数")
    sentiment_score: Optional[float] = Field(None, description="情感分析分数")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SentimentAnalysis(BaseModel):
    """情感分析结果"""
    text: str = Field(..., description="分析文本")
    sentiment: SentimentType = Field(..., description="情感类型")
    confidence: float = Field(..., ge=0, le=1, description="置信度")
    score: float = Field(..., ge=-1, le=1, description="情感分数")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TimeSeriesPrediction(BaseModel):
    """时序预测结果"""
    symbol: str = Field(..., description="股票代码")
    prediction_type: str = Field(..., description="预测类型: price/return/volatility")
    predictions: List[float] = Field(..., description="预测值序列")
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(None, description="置信区间")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="模型信息")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LLMAnalysis(BaseModel):
    """LLM分析结果"""
    query: str = Field(..., description="查询内容")
    response: str = Field(..., description="分析结果")
    model_name: str = Field(..., description="使用的模型")
    tokens_used: Optional[int] = Field(None, description="使用的token数")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="置信度")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============== 请求/响应模型 ===============

class SentimentRequest(BaseModel):
    """情感分析请求"""
    texts: List[str] = Field(..., min_items=1, description="待分析文本列表")
    batch_size: Optional[int] = Field(32, ge=1, description="批处理大小")


class SentimentResponse(BaseModel):
    """情感分析响应"""
    results: List[SentimentAnalysis] = Field(..., description="分析结果列表")
    processing_time: float = Field(..., description="处理时间(秒)")


class LLMRequest(BaseModel):
    """LLM推理请求"""
    prompt: str = Field(..., description="输入提示")
    max_tokens: Optional[int] = Field(2048, ge=1, description="最大token数")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="温度参数")
    top_p: Optional[float] = Field(0.9, ge=0, le=1, description="Top-p采样")
    system_prompt: Optional[str] = Field(None, description="系统提示")


class LLMResponse(BaseModel):
    """LLM推理响应"""
    response: str = Field(..., description="生成的回复")
    model: str = Field(..., description="模型名称")
    tokens_used: int = Field(..., description="使用的token数")
    processing_time: float = Field(..., description="处理时间(秒)")


class TimeSeriesRequest(BaseModel):
    """时序预测请求"""
    symbol: str = Field(..., description="股票代码")
    data: List[float] = Field(..., min_items=10, description="历史数据")
    prediction_steps: int = Field(5, ge=1, le=30, description="预测步数")
    model_type: str = Field("arima", description="模型类型: arima/garch/lstm")


class TimeSeriesResponse(BaseModel):
    """时序预测响应"""
    prediction: TimeSeriesPrediction = Field(..., description="预测结果")
    model_metrics: Dict[str, float] = Field(default_factory=dict, description="模型指标")
    processing_time: float = Field(..., description="处理时间(秒)")


# =============== 健康检查和状态模型 ===============

class ServiceStatus(str, Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """健康检查响应"""
    status: ServiceStatus = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: Dict[str, bool] = Field(default_factory=dict, description="详细检查项")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外信息")


class APIResponse(BaseModel):
    """通用API响应格式"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field("", description="响应消息")
    data: Optional[Any] = Field(None, description="响应数据")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="请求ID")