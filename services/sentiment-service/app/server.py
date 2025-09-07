"""
InvestIQ情感分析服务 - ONNX+TensorRT优化版本
支持批量处理，GPU加速推理
"""

import asyncio
import time
import os
from typing import List, Dict, Any
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
import onnxruntime as ort
from transformers import AutoTokenizer

from investiq_common.models import (
    SentimentRequest, SentimentResponse, SentimentAnalysis, 
    SentimentType, HealthCheck, ServiceStatus
)
from investiq_common.utils import setup_logging, get_logger, performance_timer


logger = get_logger(__name__)

# 配置
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
MODEL_PATH = "/models/sentiment_model.onnx"
MAX_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512


class SentimentAnalyzer:
    """ONNX+TensorRT优化的情感分析器"""
    
    def __init__(self):
        self.tokenizer = None
        self.session = None
        self.is_ready = False
    
    async def load_model(self):
        """加载模型和tokenizer"""
        try:
            logger.info("Loading sentiment analysis model...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir="/cache/transformers"
            )
            logger.info("Tokenizer loaded successfully")
            
            # 配置ONNX Runtime
            providers = []
            if ort.get_device() == 'GPU':
                # 优先使用TensorRT
                providers.append(('TensorrtExecutionProvider', {
                    'trt_max_workspace_size': 2147483648,  # 2GB
                    'trt_fp16_enable': True,
                    'trt_int8_enable': False,
                }))
                providers.append('CUDAExecutionProvider')
            
            providers.append('CPUExecutionProvider')
            
            # 加载ONNX模型
            if os.path.exists(MODEL_PATH):
                self.session = ort.InferenceSession(
                    MODEL_PATH,
                    providers=providers
                )
                logger.info(f"ONNX model loaded with providers: {self.session.get_providers()}")
            else:
                # 如果没有ONNX模型，使用transformers作为后备
                logger.warning(f"ONNX model not found at {MODEL_PATH}, using transformers backend")
                await self._load_transformers_backend()
            
            self.is_ready = True
            logger.info("Sentiment analyzer ready")
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    async def _load_transformers_backend(self):
        """加载transformers后备模型"""
        from transformers import AutoModelForSequenceClassification
        import torch
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            cache_dir="/cache/transformers"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
    
    @performance_timer(logger)
    async def analyze_batch(self, texts: List[str]) -> List[SentimentAnalysis]:
        """批量情感分析"""
        if not self.is_ready:
            raise RuntimeError("Model not loaded")
        
        if len(texts) == 0:
            return []
        
        # 限制批次大小
        if len(texts) > MAX_BATCH_SIZE:
            logger.warning(f"Batch size {len(texts)} exceeds maximum {MAX_BATCH_SIZE}, splitting")
            results = []
            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch = texts[i:i + MAX_BATCH_SIZE]
                batch_results = await self.analyze_batch(batch)
                results.extend(batch_results)
            return results
        
        try:
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            if self.session:  # ONNX推理
                return await self._onnx_inference(processed_texts)
            else:  # Transformers后备推理
                return await self._transformers_inference(processed_texts)
        
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # 返回中性结果作为后备
            return [
                SentimentAnalysis(
                    text=text,
                    sentiment=SentimentType.NEUTRAL,
                    confidence=0.0,
                    score=0.0
                )
                for text in texts
            ]
    
    async def _onnx_inference(self, texts: List[str]) -> List[SentimentAnalysis]:
        """ONNX推理"""
        # 分词
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="np"
        )
        
        # ONNX推理
        outputs = self.session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
        )
        
        # 处理输出
        logits = outputs[0]
        probabilities = self._softmax(logits)
        
        results = []
        for i, text in enumerate(texts):
            prob = probabilities[i]
            sentiment, confidence, score = self._interpret_output(prob)
            
            results.append(SentimentAnalysis(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                score=score
            ))
        
        return results
    
    async def _transformers_inference(self, texts: List[str]) -> List[SentimentAnalysis]:
        """Transformers后备推理"""
        import torch
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()
        
        probabilities = self._softmax(logits)
        
        results = []
        for i, text in enumerate(texts):
            prob = probabilities[i]
            sentiment, confidence, score = self._interpret_output(prob)
            
            results.append(SentimentAnalysis(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                score=score
            ))
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        if not text or not isinstance(text, str):
            return ""
        
        # 基础清理
        text = text.strip()
        
        # 长度限制
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _interpret_output(self, probabilities: np.ndarray) -> tuple:
        """解释模型输出"""
        # 假设模型输出3个类别：[negative, neutral, positive]
        neg_prob, neu_prob, pos_prob = probabilities
        
        # 确定情感类型
        max_prob = max(neg_prob, neu_prob, pos_prob)
        
        if max_prob == pos_prob:
            sentiment = SentimentType.POSITIVE
        elif max_prob == neg_prob:
            sentiment = SentimentType.NEGATIVE
        else:
            sentiment = SentimentType.NEUTRAL
        
        # 计算置信度和分数
        confidence = float(max_prob)
        score = float(pos_prob - neg_prob)  # -1到1之间
        
        return sentiment, confidence, score


# 创建应用和分析器
app = FastAPI(
    title="InvestIQ Sentiment Service",
    description="基于ONNX+TensorRT的高性能情感分析服务",
    version="1.0.0"
)

analyzer = SentimentAnalyzer()

# 配置日志
setup_logging()


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    await analyzer.load_model()


@app.get("/health")
async def health_check():
    """健康检查"""
    checks = {
        "model_loaded": analyzer.is_ready,
        "tokenizer_loaded": analyzer.tokenizer is not None,
    }
    
    # 检查GPU状态
    gpu_available = False
    try:
        if analyzer.session:
            gpu_available = "TensorrtExecutionProvider" in analyzer.session.get_providers() or \
                           "CUDAExecutionProvider" in analyzer.session.get_providers()
    except:
        pass
    
    checks["gpu_acceleration"] = gpu_available
    
    overall_healthy = all(checks.values())
    status = ServiceStatus.HEALTHY if overall_healthy else ServiceStatus.UNHEALTHY
    
    return HealthCheck(
        status=status,
        version="1.0.0",
        checks=checks,
        metadata={
            "model": MODEL_NAME,
            "backend": "ONNX+TensorRT" if analyzer.session else "Transformers",
            "max_batch_size": MAX_BATCH_SIZE
        }
    )


@app.post("/v1/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest) -> SentimentResponse:
    """批量情感分析"""
    if not analyzer.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not request.texts:
        return SentimentResponse(results=[], processing_time=0.0)
    
    start_time = time.time()
    
    try:
        results = await analyzer.analyze_batch(request.texts)
        processing_time = time.time() - start_time
        
        logger.info(f"Analyzed {len(request.texts)} texts in {processing_time:.3f}s")
        
        return SentimentResponse(
            results=results,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/v1/analyze/single")
async def analyze_single(text: str) -> SentimentAnalysis:
    """单条文本分析"""
    request = SentimentRequest(texts=[text])
    response = await analyze_sentiment(request)
    
    if response.results:
        return response.results[0]
    else:
        raise HTTPException(status_code=500, detail="Analysis failed")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )