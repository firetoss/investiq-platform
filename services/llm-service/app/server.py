"""
InvestIQ LLM推理服务 - 基于llama.cpp的HTTP服务包装
提供OpenAI兼容的API接口
"""

import asyncio
import json
import time
import httpx
import uvicorn
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from investiq_common.models import LLMRequest, LLMResponse, HealthCheck, ServiceStatus
from investiq_common.utils import setup_logging, get_logger


logger = get_logger(__name__)

# 配置
LLAMA_CPP_URL = "http://localhost:8080"  # llama.cpp server地址
MODEL_NAME = "Qwen3-8B-Instruct"


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天补全请求 - OpenAI兼容格式"""
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: Optional[int] = Field(2048, description="最大token数")
    temperature: Optional[float] = Field(0.7, description="温度参数")
    top_p: Optional[float] = Field(0.9, description="Top-p采样")
    stream: Optional[bool] = Field(False, description="是否流式输出")


app = FastAPI(
    title="InvestIQ LLM Service",
    description="基于llama.cpp的LLM推理服务",
    version="1.0.0"
)

setup_logging()


class LLamaClient:
    """llama.cpp客户端封装"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def completion(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """调用llama.cpp completion接口"""
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["</s>", "\n\n"],
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/completion",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.RequestError as e:
            logger.error(f"Request to llama.cpp failed: {e}")
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM service returned error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail="LLM inference failed")
    
    async def health_check(self) -> bool:
        """检查llama.cpp服务健康状态"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False


# 全局llama.cpp客户端
llama_client = LLamaClient(LLAMA_CPP_URL)


@app.get("/health")
async def health_check():
    """健康检查"""
    llama_healthy = await llama_client.health_check()
    
    status = ServiceStatus.HEALTHY if llama_healthy else ServiceStatus.UNHEALTHY
    
    return HealthCheck(
        status=status,
        version="1.0.0",
        checks={
            "llama_cpp": llama_healthy,
        },
        metadata={
            "model": MODEL_NAME,
            "backend": "llama.cpp"
        }
    )


@app.post("/v1/inference", response_model=LLMResponse)
async def inference(request: LLMRequest) -> LLMResponse:
    """LLM推理 - InvestIQ格式"""
    start_time = time.time()
    
    try:
        # 构建提示
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"System: {request.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # 调用llama.cpp
        result = await llama_client.completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # 提取响应内容
        content = result.get("content", "").strip()
        tokens_predicted = result.get("tokens_predicted", 0)
        
        processing_time = time.time() - start_time
        
        return LLMResponse(
            response=content,
            model=MODEL_NAME,
            tokens_used=tokens_predicted,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """聊天补全 - OpenAI兼容接口"""
    start_time = time.time()
    
    try:
        # 构建对话提示
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        # 调用llama.cpp
        result = await llama_client.completion(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        content = result.get("content", "").strip()
        tokens_used = result.get("tokens_predicted", 0)
        processing_time = time.time() - start_time
        
        # OpenAI兼容格式响应
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),  # 简化计算
                "completion_tokens": tokens_used,
                "total_tokens": len(prompt.split()) + tokens_used
            },
            "processing_time": processing_time
        }
    
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail="Chat completion failed")


@app.get("/v1/models")
async def list_models():
    """列出可用模型 - OpenAI兼容"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "investiq"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )