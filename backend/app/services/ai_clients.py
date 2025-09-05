"""
AI服务客户端 - 纯微服务架构
通过HTTP REST调用独立的AI服务
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ServiceConfig(BaseModel):
    """服务配置"""
    base_url: str
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 1.0


class LLMServiceClient:
    """LLM服务客户端"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self.client
    
    async def inference(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        LLM推理
        
        Args:
            prompt: 输入提示
            max_tokens: 最大token数
            temperature: 温度参数
            top_p: top_p参数
            
        Returns:
            推理结果
        """
        # 使用 OpenAI Chat Completions 兼容端点: /v1/chat/completions
        # 消息格式: [{"role":"user","content": prompt}]
        # 兼容性：若响应不含 message.content，则回退到 text 字段。
        try:
            client = await self._get_client()

            payload = {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

            # 简单重试
            last_err = None
            for attempt in range(self.config.retries):
                try:
                    response = await client.post("/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    break
                except Exception as e:
                    last_err = e
                    if attempt < self.config.retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                    else:
                        raise

            result = response.json()
            text = ""
            if isinstance(result, dict) and result.get("choices"):
                choice = result["choices"][0]
                msg = choice.get("message") or {}
                text = (msg.get("content") or choice.get("text") or "").strip()
            return {
                "text": text,
                "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                "model": result.get("model", "llm")
            }

        except Exception as e:
            logger.error(f"LLM service call failed: {e}")
            return {
                "text": f"LLM服务调用失败: {str(e)}",
                "tokens_used": 0,
                "model": "llm",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()


class SentimentServiceClient:
    """情感分析服务客户端"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self.client
    
    async def analyze(
        self, 
        texts: Union[str, List[str]], 
        return_all_scores: bool = False
    ) -> List[Dict[str, Any]]:
        """
        情感分析
        
        Args:
            texts: 文本或文本列表
            return_all_scores: 是否返回所有分数
            
        Returns:
            情感分析结果
        """
        try:
            client = await self._get_client()
            
            # 标准化输入
            if isinstance(texts, str):
                texts = [texts]
            
            # 构建请求
            request_data = {
                "texts": texts,
                "return_all_scores": return_all_scores
            }
            
            # 发送请求
            response = await client.post("/analyze", json=request_data)
            response.raise_for_status()
            
            result = response.json()
            
            # 返回结果
            return result.get("results", [])
                
        except Exception as e:
            logger.error(f"Sentiment service call failed: {e}")
            # 返回默认结果
            return [
                {
                    "text": text,
                    "label": "NEUTRAL",
                    "score": 0.5,
                    "confidence": 0.0
                } for text in (texts if isinstance(texts, list) else [texts])
            ]
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Sentiment health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()


class TimeseriesServiceClient:
    """时序预测服务客户端"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self.client
    
    async def predict(
        self, 
        data: List[float], 
        horizon: int = 10,
        confidence_intervals: bool = False
    ) -> Dict[str, Any]:
        """
        时序预测
        
        Args:
            data: 历史数据
            horizon: 预测期数
            confidence_intervals: 是否计算置信区间
            
        Returns:
            预测结果
        """
        try:
            client = await self._get_client()
            
            # 构建请求
            request_data = {
                "data": data,
                "horizon": horizon,
                "confidence_intervals": confidence_intervals
            }
            
            # 发送请求
            response = await client.post("/predict", json=request_data)
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "predictions": result.get("predictions", []),
                "confidence_lower": result.get("confidence_lower"),
                "confidence_upper": result.get("confidence_upper"),
                "processing_time": result.get("processing_time", 0.0),
                "model": "cpu-traditional"
            }
                
        except Exception as e:
            logger.error(f"Timeseries service call failed: {e}")
            # 返回默认预测
            import random
            predictions = [random.uniform(0.9, 1.1) for _ in range(horizon)]
            return {
                "predictions": predictions,
                "confidence_lower": None,
                "confidence_upper": None,
                "processing_time": 0.0,
                "model": "patchtst-financial",
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Timeseries health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()


class CPUTimeseriesServiceClient:
    """CPU时序服务客户端"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """获取HTTP客户端"""
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
        return self.client
    
    async def predict(
        self, 
        data: List[float], 
        horizon: int = 10,
        algorithm: str = "auto",
        confidence_intervals: bool = False
    ) -> Dict[str, Any]:
        """
        CPU时序预测
        
        Args:
            data: 历史数据
            horizon: 预测期数
            algorithm: 算法类型
            confidence_intervals: 是否计算置信区间
            
        Returns:
            预测结果
        """
        try:
            client = await self._get_client()
            
            # 构建请求
            request_data = {
                "data": data,
                "horizon": horizon,
                "algorithm": algorithm,
                "confidence_intervals": confidence_intervals
            }
            
            # 发送请求
            response = await client.post("/predict", json=request_data)
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "predictions": result.get("predictions", []),
                "confidence_lower": result.get("confidence_lower"),
                "confidence_upper": result.get("confidence_upper"),
                "algorithm_used": result.get("algorithm_used", algorithm),
                "processing_time": result.get("processing_time", 0.0),
                "model": "cpu-traditional"
            }
                
        except Exception as e:
            logger.error(f"CPU timeseries service call failed: {e}")
            # 返回默认预测
            import random
            predictions = [random.uniform(0.9, 1.1) for _ in range(horizon)]
            return {
                "predictions": predictions,
                "confidence_lower": None,
                "confidence_upper": None,
                "algorithm_used": algorithm,
                "processing_time": 0.0,
                "model": "cpu-traditional",
                "error": str(e)
            }
    
    async def technical_analysis(self, data: List[float]) -> Dict[str, Any]:
        """技术分析"""
        try:
            client = await self._get_client()
            response = await client.post("/technical-analysis", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"CPU timeseries health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self):
        """关闭客户端"""
        if self.client:
            await self.client.aclose()


class AIServiceClients:
    """AI服务客户端管理器"""
    
    def __init__(self):
        # 从环境变量获取服务URL
        import os
        
        self.llm_client = LLMServiceClient(ServiceConfig(
            base_url=os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
        ))
        
        self.sentiment_client = SentimentServiceClient(ServiceConfig(
            base_url=os.getenv("SENTIMENT_SERVICE_URL", "http://sentiment-service:8002")
        ))
        
        # 为兼容旧字段，TIMESERIES_SERVICE_URL 默认指向 CPU 服务
        self.timeseries_client = TimeseriesServiceClient(ServiceConfig(
            base_url=os.getenv("TIMESERIES_SERVICE_URL", os.getenv("CPU_TIMESERIES_SERVICE_URL", "http://cpu-timeseries:8004"))
        ))
        
        self.cpu_timeseries_client = CPUTimeseriesServiceClient(ServiceConfig(
            base_url=os.getenv("CPU_TIMESERIES_SERVICE_URL", "http://cpu-timeseries:8004")
        ))
    
    async def health_check_all(self) -> Dict[str, Any]:
        """检查所有AI服务健康状态"""
        results = {}
        
        try:
            results["llm"] = await self.llm_client.health_check()
        except Exception as e:
            results["llm"] = {"status": "error", "error": str(e)}
        
        try:
            results["sentiment"] = await self.sentiment_client.health_check()
        except Exception as e:
            results["sentiment"] = {"status": "error", "error": str(e)}
        
        try:
            results["timeseries"] = await self.timeseries_client.health_check()
        except Exception as e:
            results["timeseries"] = {"status": "error", "error": str(e)}
        
        try:
            results["cpu_timeseries"] = await self.cpu_timeseries_client.health_check()
        except Exception as e:
            results["cpu_timeseries"] = {"status": "error", "error": str(e)}
        
        return results
    
    async def close_all(self):
        """关闭所有客户端"""
        await asyncio.gather(
            self.llm_client.close(),
            self.sentiment_client.close(),
            self.timeseries_client.close(),
            self.cpu_timeseries_client.close(),
            return_exceptions=True
        )


# 全局AI服务客户端实例
ai_clients = AIServiceClients()
