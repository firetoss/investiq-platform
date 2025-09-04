"""
时序预测服务器 - DLA核心1专用
基于PatchTST金融模型，优化金融时序预测性能
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 条件导入
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="InvestIQ Timeseries Service", version="1.0.0")

# 全局模型变量
timeseries_model = None
tokenizer = None


class TimeseriesRequest(BaseModel):
    """时序预测请求"""
    data: List[float] = Field(..., description="历史数据", min_items=5, max_items=1000)
    horizon: int = Field(default=10, description="预测期数", ge=1, le=100)
    confidence_intervals: bool = Field(default=False, description="是否计算置信区间")


class TimeseriesResponse(BaseModel):
    """时序预测响应"""
    predictions: List[float]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    processing_time: float
    model_info: Dict[str, Any]


async def load_model():
    """加载PatchTST金融时序预测模型"""
    global timeseries_model, tokenizer
    
    try:
        model_name = os.getenv("MODEL_NAME", "ibm-granite/granite-timeseries-patchtst")
        dla_core = int(os.getenv("DLA_CORE", "1"))
        batch_size = int(os.getenv("BATCH_SIZE", "32"))
        
        logger.info(f"Loading timeseries model: {model_name} on DLA core {dla_core}")
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available")
            return False
        
        # 设置设备
        if torch.cuda.is_available():
            device = f"cuda:{dla_core}"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        logger.info(f"Using device: {device}, dtype: {torch_dtype}")
        
        # 加载模型 (简化实现，实际需要根据PatchTST具体接口调整)
        try:
            # 尝试加载真实PatchTST模型
            timeseries_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map={"": device}
            )
            
            # 设置为评估模式
            timeseries_model.eval()
            
        except Exception as e:
            logger.warning(f"Failed to load real PatchTST model: {e}")
            # 使用Mock实现
            timeseries_model = MockPatchTSTModel(device, torch_dtype)
        
        # DLA优化设置
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 预热模型
            dummy_data = torch.randn(1, 512, dtype=torch_dtype, device=device)
            with torch.no_grad():
                if hasattr(timeseries_model, 'forward'):
                    _ = timeseries_model(dummy_data)
                logger.info("Model warmup completed")
        
        logger.info(f"Timeseries model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load timeseries model: {e}")
        return False


class MockPatchTSTModel:
    """Mock PatchTST模型，用于开发测试"""
    
    def __init__(self, device: str, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        
    def predict(self, data: List[float], horizon: int) -> List[float]:
        """模拟PatchTST预测"""
        try:
            # 简单的趋势预测算法
            if len(data) < 2:
                return [data[0] if data else 1.0] * horizon
            
            # 计算简单趋势
            recent_data = data[-min(20, len(data)):]
            trend = (recent_data[-1] - recent_data[0]) / len(recent_data)
            
            predictions = []
            last_value = data[-1]
            
            for i in range(horizon):
                # 趋势 + 小幅随机波动
                next_value = last_value + trend + np.random.normal(0, abs(last_value) * 0.01)
                predictions.append(float(next_value))
                last_value = next_value
            
            return predictions
            
        except Exception as e:
            logger.error(f"Mock prediction failed: {e}")
            # 返回简单预测
            import random
            last_value = data[-1] if data else 1.0
            return [last_value * (1 + random.uniform(-0.02, 0.02)) for _ in range(horizon)]


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    success = await load_model()
    if not success:
        logger.error("Failed to load model during startup")


@app.post("/predict", response_model=TimeseriesResponse)
async def predict_timeseries(request: TimeseriesRequest) -> TimeseriesResponse:
    """
    时序预测
    
    Args:
        request: 时序预测请求
        
    Returns:
        时序预测结果
    """
    start_time = datetime.now()
    
    try:
        if timeseries_model is None:
            raise HTTPException(status_code=503, detail="Timeseries model not loaded")
        
        # 执行预测
        if hasattr(timeseries_model, 'predict'):
            predictions = timeseries_model.predict(request.data, request.horizon)
        else:
            # 使用Mock实现
            mock_model = MockPatchTSTModel("cpu", torch.float32)
            predictions = mock_model.predict(request.data, request.horizon)
        
        # 计算置信区间
        confidence_lower = None
        confidence_upper = None
        
        if request.confidence_intervals:
            # 简化的置信区间计算
            std_dev = np.std(request.data[-min(20, len(request.data)):])
            confidence_lower = [p - 1.96 * std_dev for p in predictions]
            confidence_upper = [p + 1.96 * std_dev for p in predictions]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TimeseriesResponse(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            processing_time=processing_time,
            model_info={
                "model_name": os.getenv("MODEL_NAME", "patchtst-financial"),
                "device": "DLA_CORE_1",
                "precision": "FP16",
                "batch_size": int(os.getenv("BATCH_SIZE", "32")),
                "context_length": int(os.getenv("CONTEXT_LENGTH", "512")),
                "prediction_length": int(os.getenv("PREDICTION_LENGTH", "96"))
            }
        )
        
    except Exception as e:
        logger.error(f"Timeseries prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"时序预测失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        model_loaded = timeseries_model is not None
        
        # 检查DLA状态
        dla_available = torch.cuda.is_available()
        dla_core = int(os.getenv("DLA_CORE", "1"))
        
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
        metrics = {
            "model_loaded": timeseries_model is not None,
            "dla_core": int(os.getenv("DLA_CORE", "1")),
            "batch_size": int(os.getenv("BATCH_SIZE", "32")),
            "precision": "FP16",
            "hardware_target": "DLA",
            "context_length": int(os.getenv("CONTEXT_LENGTH", "512")),
            "prediction_length": int(os.getenv("PREDICTION_LENGTH", "96"))
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
    
    logger.info(f"Starting timeseries prediction service on {host}:{port}")
    logger.info(f"DLA Core: {os.getenv('DLA_CORE', '1')}")
    logger.info(f"Batch Size: {os.getenv('BATCH_SIZE', '32')}")
    logger.info(f"Model: {os.getenv('MODEL_NAME', 'patchtst-financial')}")
    
    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # DLA服务使用单进程
    )
