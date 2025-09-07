"""
微服务客户端 - 服务间HTTP通信封装
提供重试、错误处理、负载均衡等功能
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel

from ..models import (
    SentimentRequest, SentimentResponse,
    LLMRequest, LLMResponse, 
    TimeSeriesRequest, TimeSeriesResponse,
    HealthCheck, APIResponse
)


logger = logging.getLogger(__name__)


class ServiceConfig(BaseModel):
    """服务配置"""
    name: str
    base_url: str
    timeout: float = 30.0
    retries: int = 3
    retry_delay: float = 1.0
    health_check_path: str = "/health"


class BaseServiceClient:
    """基础服务客户端"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
    
    @asynccontextmanager
    async def get_client(self):
        """获取HTTP客户端 - 上下文管理器"""
        if not self._client:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers={"User-Agent": f"investiq-{self.config.name}-client/1.0"}
            )
        
        try:
            yield self._client
        finally:
            # 保持连接复用，不在这里关闭
            pass
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> HealthCheck:
        """健康检查"""
        async with self.get_client() as client:
            response = await client.get(self.config.health_check_path)
            response.raise_for_status()
            return HealthCheck(**response.json())
    
    async def _retry_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """带重试的请求"""
        last_exception = None
        
        for attempt in range(self.config.retries):
            try:
                async with self.get_client() as client:
                    response = await client.request(method, endpoint, **kwargs)
                    
                    if response.status_code < 500:  # 非服务器错误不重试
                        return response
                    
                    logger.warning(
                        f"Service {self.config.name} returned {response.status_code}, "
                        f"attempt {attempt + 1}/{self.config.retries}"
                    )
                    
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(
                    f"Request to {self.config.name} failed: {e}, "
                    f"attempt {attempt + 1}/{self.config.retries}"
                )
            
            if attempt < self.config.retries - 1:
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # 指数退避
        
        # 所有重试失败
        if last_exception:
            raise last_exception
        else:
            response.raise_for_status()  # 抛出HTTP错误
            return response


class LLMServiceClient(BaseServiceClient):
    """LLM服务客户端"""
    
    async def inference(self, request: LLMRequest) -> LLMResponse:
        """LLM推理"""
        response = await self._retry_request(
            "POST", 
            "/v1/inference",
            json=request.model_dump()
        )
        response.raise_for_status()
        return LLMResponse(**response.json())
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> LLMResponse:
        """对话补全（OpenAI兼容格式）"""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = await self._retry_request(
            "POST",
            "/v1/chat/completions", 
            json=payload
        )
        response.raise_for_status()
        return LLMResponse(**response.json())


class SentimentServiceClient(BaseServiceClient):
    """情感分析服务客户端"""
    
    async def analyze(self, request: SentimentRequest) -> SentimentResponse:
        """批量情感分析"""
        response = await self._retry_request(
            "POST",
            "/v1/analyze",
            json=request.model_dump()
        )
        response.raise_for_status()
        return SentimentResponse(**response.json())
    
    async def analyze_single(self, text: str) -> SentimentResponse:
        """单条文本分析"""
        request = SentimentRequest(texts=[text])
        return await self.analyze(request)


class TimeSeriesServiceClient(BaseServiceClient):
    """时序预测服务客户端"""
    
    async def predict(self, request: TimeSeriesRequest) -> TimeSeriesResponse:
        """时序预测"""
        response = await self._retry_request(
            "POST",
            "/v1/predict",
            json=request.model_dump()
        )
        response.raise_for_status()
        return TimeSeriesResponse(**response.json())
    
    async def batch_predict(
        self, 
        requests: List[TimeSeriesRequest]
    ) -> List[TimeSeriesResponse]:
        """批量预测"""
        tasks = [self.predict(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


class ServiceRegistry:
    """服务注册表 - 管理所有微服务客户端"""
    
    def __init__(self):
        self.clients: Dict[str, BaseServiceClient] = {}
    
    def register_llm_service(self, base_url: str, **kwargs) -> LLMServiceClient:
        """注册LLM服务"""
        config = ServiceConfig(name="llm", base_url=base_url, **kwargs)
        client = LLMServiceClient(config)
        self.clients["llm"] = client
        return client
    
    def register_sentiment_service(self, base_url: str, **kwargs) -> SentimentServiceClient:
        """注册情感分析服务"""
        config = ServiceConfig(name="sentiment", base_url=base_url, **kwargs)
        client = SentimentServiceClient(config)
        self.clients["sentiment"] = client
        return client
    
    def register_timeseries_service(self, base_url: str, **kwargs) -> TimeSeriesServiceClient:
        """注册时序服务"""
        config = ServiceConfig(name="timeseries", base_url=base_url, **kwargs)
        client = TimeSeriesServiceClient(config)
        self.clients["timeseries"] = client
        return client
    
    async def health_check_all(self) -> Dict[str, HealthCheck]:
        """检查所有服务健康状态"""
        results = {}
        
        for name, client in self.clients.items():
            try:
                health = await client.health_check()
                results[name] = health
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = HealthCheck(
                    status="unhealthy",
                    version="unknown",
                    checks={"connection": False},
                    metadata={"error": str(e)}
                )
        
        return results
    
    async def close_all(self):
        """关闭所有客户端"""
        for client in self.clients.values():
            await client.close()
        self.clients.clear()


# 全局服务注册表实例
registry = ServiceRegistry()