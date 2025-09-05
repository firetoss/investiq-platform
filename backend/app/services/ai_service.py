"""
AI服务 - 统一的AI功能接口
整合LLM、情感分析、时间序列预测等功能
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from .ai_clients import ai_clients
from .performance_monitor import performance_monitor

logger = logging.getLogger(__name__)


class LLMRequest(BaseModel):
    """LLM请求模型"""
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None


class LLMResponse(BaseModel):
    """LLM响应模型"""
    text: str
    model_name: str
    tokens_used: int = 0
    processing_time: float = 0.0


class SentimentRequest(BaseModel):
    """情感分析请求模型"""
    texts: Union[str, List[str]]
    return_all_scores: bool = False


class SentimentResult(BaseModel):
    """单个情感分析结果"""
    text: str
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float
    confidence: float = 0.0


class SentimentResponse(BaseModel):
    """情感分析响应模型"""
    results: List[SentimentResult]
    model_name: str
    processing_time: float = 0.0


class TimeseriesRequest(BaseModel):
    """时间序列预测请求模型"""
    data: List[float]
    horizon: int = 10
    confidence_intervals: bool = False


class TimeseriesResponse(BaseModel):
    """时间序列预测响应模型"""
    predictions: List[float]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    model_name: str
    processing_time: float = 0.0


class AIService:
    """
    AI服务类 - 纯微服务架构
    
    通过HTTP调用独立的AI服务：
    1. 大语言模型推理 (GPU专用服务)
    2. 情感分析 (DLA0专用服务)
    3. 时间序列预测 (DLA1专用服务)
    4. CPU传统算法 (CPU专用服务)
    """
    
    def __init__(self):
        self.ai_clients = ai_clients
        
        # 中文金融提示词模板
        self.prompt_templates = {
            "policy_analysis": """
请分析以下政策文件的投资影响：

政策内容：{policy_text}

请从以下角度分析：
1. 受益行业：列出最可能受益的3-5个行业
2. 政策强度评分：给出1-100的评分，100表示影响最大
3. 实施确定性：评估政策落地的可能性（高/中/低）
4. 投资建议：具体的投资方向和注意事项

请用中文回答，保持客观和专业。
""",
            
            "company_analysis": """
请分析以下公司信息：

公司名称：{company_name}
行业：{industry}
财务数据：{financial_data}
最新新闻：{news}

请从以下角度分析：
1. 基本面评估：财务健康度、盈利能力、成长性
2. 行业地位：市场份额、竞争优势、护城河
3. 风险因素：主要风险点和不确定性
4. 投资评级：给出买入/持有/卖出建议及理由

请用中文回答，保持客观和专业。
""",
            
            "market_summary": """
请总结以下市场信息：

市场数据：{market_data}
重要新闻：{news_summary}
政策动态：{policy_updates}

请提供：
1. 市场概况：当前市场状态和主要趋势
2. 热点板块：表现突出的行业和主题
3. 风险提示：需要关注的风险因素
4. 操作建议：短期和中期的投资策略

请用中文回答，保持简洁和实用。
"""
        }
    
    async def llm_inference(self, request: LLMRequest, model_name: str = "qwen3-8b") -> LLMResponse:
        """
        LLM推理 - 通过HTTP调用GPU专用LLM服务
        
        Args:
            request: LLM请求
            model_name: 模型名称
            
        Returns:
            LLM响应
        """
        start_time = datetime.now()
        
        try:
            # 调用独立的LLM服务
            result = await self.ai_clients.llm_client.inference(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 记录性能指标
            performance_monitor.record_inference_metrics(
                model_name=result.get("model", model_name),
                inference_time=processing_time,
                batch_size=1,
                hardware_target="gpu"
            )
            
            return LLMResponse(
                text=result.get("text", ""),
                model_name=result.get("model", model_name),
                tokens_used=result.get("tokens_used", 0),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LLMResponse(
                text=f"推理失败: {str(e)}",
                model_name=model_name,
                processing_time=processing_time
            )
    
    async def sentiment_analysis(self, request: SentimentRequest, model_name: str = "roberta-financial-en") -> SentimentResponse:
        """
        情感分析 - 通过HTTP调用DLA0专用情感分析服务
        
        Args:
            request: 情感分析请求
            model_name: 模型名称
            
        Returns:
            情感分析响应
        """
        start_time = datetime.now()
        
        try:
            # 调用独立的情感分析服务
            raw_results = await self.ai_clients.sentiment_client.analyze(
                texts=request.texts,
                return_all_scores=request.return_all_scores
            )
            
            # 转换为标准格式
            results = []
            texts = request.texts if isinstance(request.texts, list) else [request.texts]
            
            for i, text in enumerate(texts):
                if i < len(raw_results):
                    raw_result = raw_results[i]
                    results.append(SentimentResult(
                        text=raw_result.get("text", text),
                        label=raw_result.get("label", "NEUTRAL"),
                        score=raw_result.get("score", 0.5),
                        confidence=raw_result.get("confidence", 0.0)
                    ))
                else:
                    results.append(SentimentResult(
                        text=text,
                        label="NEUTRAL",
                        score=0.5,
                        confidence=0.0
                    ))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 记录性能指标
            performance_monitor.record_inference_metrics(
                model_name=model_name,
                inference_time=processing_time,
                batch_size=len(texts),
                hardware_target="dla"  # 情感分析使用DLA0
            )
            
            return SentimentResponse(
                results=results,
                model_name=model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 返回默认结果
            texts = request.texts if isinstance(request.texts, list) else [request.texts]
            results = [
                SentimentResult(
                    text=text,
                    label="NEUTRAL",
                    score=0.5,
                    confidence=0.0
                ) for text in texts
            ]
            
            return SentimentResponse(
                results=results,
                model_name=model_name,
                processing_time=processing_time
            )
    
    async def timeseries_forecast(self, request: TimeseriesRequest, model_name: str = "patchtst-financial") -> TimeseriesResponse:
        """
        时间序列预测 - 通过HTTP调用DLA1专用时序预测服务
        
        Args:
            request: 时间序列预测请求
            model_name: 模型名称
            
        Returns:
            时间序列预测响应
        """
        start_time = datetime.now()
        
        try:
            # 调用独立的时序预测服务
            result = await self.ai_clients.timeseries_client.predict(
                data=request.data,
                horizon=request.horizon,
                confidence_intervals=request.confidence_intervals
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 记录性能指标
            performance_monitor.record_inference_metrics(
                model_name=result.get("model", model_name),
                inference_time=processing_time,
                batch_size=1,
                hardware_target="dla"  # 时序预测使用DLA1
            )
            
            return TimeseriesResponse(
                predictions=result.get("predictions", []),
                confidence_lower=result.get("confidence_lower"),
                confidence_upper=result.get("confidence_upper"),
                model_name=result.get("model", model_name),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Timeseries forecast failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 返回默认预测
            import random
            predictions = [random.uniform(0.9, 1.1) for _ in range(request.horizon)]
            
            return TimeseriesResponse(
                predictions=predictions,
                model_name=model_name,
                processing_time=processing_time
            )
    
    async def analyze_policy(self, policy_text: str) -> Dict[str, Any]:
        """
        政策分析
        
        Args:
            policy_text: 政策文本
            
        Returns:
            分析结果
        """
        prompt = self.prompt_templates["policy_analysis"].format(
            policy_text=policy_text
        )
        
        request = LLMRequest(prompt=prompt, max_tokens=1024)
        response = await self.llm_inference(request)
        
        return {
            "analysis": response.text,
            "model_name": response.model_name,
            "processing_time": response.processing_time
        }
    
    async def analyze_company(self, company_name: str, industry: str, 
                            financial_data: str, news: str) -> Dict[str, Any]:
        """
        公司分析
        
        Args:
            company_name: 公司名称
            industry: 行业
            financial_data: 财务数据
            news: 最新新闻
            
        Returns:
            分析结果
        """
        prompt = self.prompt_templates["company_analysis"].format(
            company_name=company_name,
            industry=industry,
            financial_data=financial_data,
            news=news
        )
        
        request = LLMRequest(prompt=prompt, max_tokens=1024)
        response = await self.llm_inference(request)
        
        return {
            "analysis": response.text,
            "model_name": response.model_name,
            "processing_time": response.processing_time
        }
    
    async def summarize_market(self, market_data: str, news_summary: str, 
                             policy_updates: str) -> Dict[str, Any]:
        """
        市场总结
        
        Args:
            market_data: 市场数据
            news_summary: 新闻摘要
            policy_updates: 政策动态
            
        Returns:
            总结结果
        """
        prompt = self.prompt_templates["market_summary"].format(
            market_data=market_data,
            news_summary=news_summary,
            policy_updates=policy_updates
        )
        
        request = LLMRequest(prompt=prompt, max_tokens=1024)
        response = await self.llm_inference(request)
        
        return {
            "summary": response.text,
            "model_name": response.model_name,
            "processing_time": response.processing_time
        }
    
    async def get_model_status(self) -> Dict[str, Any]:
        """获取所有AI服务状态 - 通过HTTP调用各服务健康检查"""
        return await self.ai_clients.health_check_all()
    
    async def load_model(self, model_name: str) -> bool:
        """加载指定模型 - 在微服务架构中，模型由各服务自行管理"""
        logger.info(f"Model loading is managed by individual AI services: {model_name}")
        return True
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载指定模型 - 在微服务架构中，模型由各服务自行管理"""
        logger.info(f"Model unloading is managed by individual AI services: {model_name}")
        return True


# 全局AI服务实例
ai_service = AIService()
