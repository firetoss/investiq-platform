"""
情感分析服务器 - GPU 专用
基于小型中文BERT/RoBERTa模型，优化批处理性能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 条件导入
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="InvestIQ Sentiment Service", version="1.0.0")

# 全局模型变量
sentiment_model = None
tokenizer = None


class SentimentRequest(BaseModel):
    """情感分析请求"""
    texts: List[str] = Field(..., description="文本列表", min_items=1, max_items=128)
    return_all_scores: bool = Field(default=False, description="是否返回所有分数")


class SentimentResult(BaseModel):
    """情感分析结果"""
    text: str
    label: str
    score: float
    confidence: float


class SentimentResponse(BaseModel):
    """情感分析响应"""
    results: List[SentimentResult]
    processing_time: float
    batch_size: int
    model_info: Dict[str, Any]


async def load_model():
    """加载RoBERTa中文金融情感分析模型"""
    global sentiment_model, tokenizer
    
    try:
        model_name = os.getenv("MODEL_NAME", "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment")
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        
        logger.info(f"Loading sentiment model: {model_name} on GPU if available")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            return False
        
        # 设置设备
        if torch.cuda.is_available():
            device_id = 0
            torch_dtype = torch.float16
            device_arg = 0
        else:
            device_id = -1
            torch_dtype = torch.float32
            device_arg = -1
        
        logger.info(f"Using device_id: {device_id}, dtype: {torch_dtype}")
        
        # 加载模型和tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        
        # 移动到目标设备
        if torch.cuda.is_available():
            model = model.to("cuda:0")
        
        # 创建pipeline
        sentiment_model = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device_arg,
            batch_size=batch_size,
            return_all_scores=True
        )
        
        # GPU优化设置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 预热模型
            dummy_texts = ["这是一个测试文本用于模型预热"] * 4
            _ = sentiment_model(dummy_texts)
            logger.info("Model warmup completed")
        
        logger.info("Sentiment model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load sentiment model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    success = await load_model()
    if not success:
        logger.error("Failed to load model during startup")


@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    """
    批量情感分析
    
    Args:
        request: 情感分析请求
        
    Returns:
        情感分析结果
    """
    start_time = datetime.now()
    
    try:
        if sentiment_model is None:
            raise HTTPException(status_code=503, detail="Sentiment model not loaded")
        
        # 执行批量分析（开启截断/填充，控制序列长度）
        max_len = int(os.getenv("MAX_LENGTH", "512"))
        raw_results = sentiment_model(
            request.texts,
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        
        # 处理结果
        results = []
        for i, text in enumerate(request.texts):
            if i < len(raw_results):
                # 获取最高分数的标签
                best_result = max(raw_results[i], key=lambda x: x['score'])
                label = best_result['label']
                score = best_result['score']
                
                # 标准化标签
                if label.upper() in ["POSITIVE", "POS", "LABEL_1", "1"]:
                    label = "POSITIVE"
                elif label.upper() in ["NEGATIVE", "NEG", "LABEL_0", "0"]:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                
                results.append(SentimentResult(
                    text=text,
                    label=label,
                    score=score,
                    confidence=score
                ))
            else:
                results.append(SentimentResult(
                    text=text,
                    label="NEUTRAL",
                    score=0.5,
                    confidence=0.0
                ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SentimentResponse(
            results=results,
            processing_time=processing_time,
            batch_size=len(request.texts),
            model_info={
                "model_name": os.getenv("MODEL_NAME", "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"),
                "device": "DLA_CORE_0",
                "precision": "FP16",
                "batch_size": int(os.getenv("BATCH_SIZE", "32"))
            }
        )
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"情感分析失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        model_loaded = sentiment_model is not None
        
        # 检查DLA状态
        dla_available = torch.cuda.is_available()
        dla_core = int(os.getenv("DLA_CORE", "0"))
        
        # 内存使用检查
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
            memory_usage = f"{gpu_used:.1f}GB / {gpu_memory:.1f}GB"
        else:
            memory_usage = "N/A"
        
        return {
            "status": "healthy" if model_loaded else "degraded",
            "model_loaded": model_loaded,
            "dla_available": dla_available,
            "dla_core": dla_core,
            "memory_usage": memory_usage,
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/metrics")
async def get_metrics():
    """获取性能指标"""
    try:
        # 简化的指标收集
        metrics = {
            "model_loaded": sentiment_model is not None,
            "dla_core": int(os.getenv("DLA_CORE", "0")),
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "precision": "FP16",
            "hardware_target": "DLA"
        }
        
        if torch.cuda.is_available():
            metrics.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                "gpu_memory_reserved": torch.cuda.memory_reserved(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory
            })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 配置服务器
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting sentiment analysis service on {host}:{port}")
    logger.info(f"DLA Core: {os.getenv('DLA_CORE', '0')}")
    logger.info(f"Batch Size: {os.getenv('BATCH_SIZE', '32')}")
    logger.info(f"Model: {os.getenv('MODEL_NAME', 'IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment')}")
    
    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # DLA服务使用单进程
    )
