"""
CPU时序服务器 - 传统金融算法专用
基于CPU并行计算，提供ARIMA、GARCH、技术指标等传统算法
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 条件导入科学计算库
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(title="InvestIQ CPU Timeseries Service", version="1.0.0")

# 线程池执行器
executor = ThreadPoolExecutor(max_workers=12)


class CPUTimeseriesRequest(BaseModel):
    """CPU时序预测请求"""
    data: List[float] = Field(..., description="历史数据", min_items=5, max_items=1000)
    horizon: int = Field(default=10, description="预测期数", ge=1, le=100)
    algorithm: str = Field(default="auto", description="算法类型", regex="^(auto|arima|garch|ma|ema|rsi|macd)$")
    confidence_intervals: bool = Field(default=False, description="是否计算置信区间")


class CPUTimeseriesResponse(BaseModel):
    """CPU时序预测响应"""
    predictions: List[float]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    algorithm_used: str
    processing_time: float
    model_info: Dict[str, Any]


class TraditionalTimeseriesAlgorithms:
    """传统时序算法集合"""
    
    @staticmethod
    def moving_average(data: List[float], window: int = 5, horizon: int = 10) -> List[float]:
        """移动平均预测"""
        if len(data) < window:
            return [data[-1]] * horizon if data else [0.0] * horizon
        
        # 计算移动平均
        ma_values = []
        for i in range(len(data) - window + 1):
            ma_values.append(np.mean(data[i:i + window]))
        
        # 预测未来值
        last_ma = ma_values[-1]
        trend = (ma_values[-1] - ma_values[-min(5, len(ma_values))]) / min(5, len(ma_values))
        
        predictions = []
        for i in range(horizon):
            next_value = last_ma + trend * (i + 1)
            predictions.append(float(next_value))
        
        return predictions
    
    @staticmethod
    def exponential_moving_average(data: List[float], alpha: float = 0.3, horizon: int = 10) -> List[float]:
        """指数移动平均预测"""
        if not data:
            return [0.0] * horizon
        
        # 计算EMA
        ema = data[0]
        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        # 简单预测（假设趋势延续）
        if len(data) >= 2:
            trend = data[-1] - data[-2]
        else:
            trend = 0
        
        predictions = []
        last_value = ema
        
        for i in range(horizon):
            next_value = last_value + trend * 0.1  # 趋势衰减
            predictions.append(float(next_value))
            last_value = next_value
        
        return predictions
    
    @staticmethod
    def simple_arima(data: List[float], horizon: int = 10) -> List[float]:
        """简化ARIMA模型"""
        if len(data) < 3:
            return [data[-1] if data else 0.0] * horizon
        
        # 简单的AR(1)模型
        data_array = np.array(data)
        
        # 计算一阶差分
        diff = np.diff(data_array)
        
        # 计算自回归系数（简化）
        if len(diff) > 1:
            ar_coef = np.corrcoef(diff[:-1], diff[1:])[0, 1]
            ar_coef = max(-0.9, min(0.9, ar_coef))  # 限制系数范围
        else:
            ar_coef = 0.1
        
        # 预测
        predictions = []
        last_value = data[-1]
        last_diff = diff[-1] if len(diff) > 0 else 0
        
        for i in range(horizon):
            next_diff = ar_coef * last_diff
            next_value = last_value + next_diff
            predictions.append(float(next_value))
            
            last_value = next_value
            last_diff = next_diff
        
        return predictions
    
    @staticmethod
    def technical_indicators(data: List[float], horizon: int = 10) -> Dict[str, Any]:
        """技术指标计算"""
        if len(data) < 14:
            return {
                "rsi": 50.0,
                "macd": 0.0,
                "signal": 0.0,
                "trend": "neutral"
            }
        
        data_array = np.array(data)
        
        # RSI计算
        delta = np.diff(data_array)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # MACD计算（简化）
        ema12 = np.mean(data_array[-12:])
        ema26 = np.mean(data_array[-26:]) if len(data_array) >= 26 else np.mean(data_array)
        macd = ema12 - ema26
        signal = macd * 0.9  # 简化信号线
        
        # 趋势判断
        if rsi > 70:
            trend = "overbought"
        elif rsi < 30:
            trend = "oversold"
        elif macd > signal:
            trend = "bullish"
        else:
            trend = "bearish"
        
        return {
            "rsi": float(rsi),
            "macd": float(macd),
            "signal": float(signal),
            "trend": trend
        }


def select_algorithm(data: List[float], algorithm: str) -> str:
    """选择最适合的算法"""
    if algorithm != "auto":
        return algorithm
    
    # 根据数据特征自动选择算法
    data_length = len(data)
    
    if data_length < 10:
        return "ma"  # 移动平均
    elif data_length < 20:
        return "ema"  # 指数移动平均
    else:
        # 检查数据的波动性
        volatility = np.std(data) / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0
        
        if volatility > 0.1:
            return "arima"  # 高波动使用ARIMA
        else:
            return "ema"  # 低波动使用EMA


@app.post("/predict", response_model=CPUTimeseriesResponse)
async def predict_timeseries(request: CPUTimeseriesRequest) -> CPUTimeseriesResponse:
    """
    CPU时序预测
    
    Args:
        request: CPU时序预测请求
        
    Returns:
        时序预测结果
    """
    start_time = datetime.now()
    
    try:
        # 选择算法
        algorithm_used = select_algorithm(request.data, request.algorithm)
        
        # 在线程池中执行计算密集型任务
        loop = asyncio.get_event_loop()
        
        if algorithm_used == "ma":
            predictions = await loop.run_in_executor(
                executor,
                TraditionalTimeseriesAlgorithms.moving_average,
                request.data, 5, request.horizon
            )
        elif algorithm_used == "ema":
            predictions = await loop.run_in_executor(
                executor,
                TraditionalTimeseriesAlgorithms.exponential_moving_average,
                request.data, 0.3, request.horizon
            )
        elif algorithm_used == "arima":
            predictions = await loop.run_in_executor(
                executor,
                TraditionalTimeseriesAlgorithms.simple_arima,
                request.data, request.horizon
            )
        else:
            # 默认使用移动平均
            predictions = await loop.run_in_executor(
                executor,
                TraditionalTimeseriesAlgorithms.moving_average,
                request.data, 5, request.horizon
            )
        
        # 计算置信区间
        confidence_lower = None
        confidence_upper = None
        
        if request.confidence_intervals:
            std_dev = np.std(request.data[-min(20, len(request.data)):])
            confidence_lower = [p - 1.96 * std_dev for p in predictions]
            confidence_upper = [p + 1.96 * std_dev for p in predictions]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CPUTimeseriesResponse(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            algorithm_used=algorithm_used,
            processing_time=processing_time,
            model_info={
                "hardware": "CPU",
                "cores": int(os.getenv("OMP_NUM_THREADS", "12")),
                "algorithm": algorithm_used,
                "vectorized": os.getenv("ENABLE_VECTORIZATION", "true") == "true"
            }
        )
        
    except Exception as e:
        logger.error(f"CPU timeseries prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"CPU时序预测失败: {str(e)}")


@app.post("/technical-analysis")
async def technical_analysis(
    data: List[float] = Field(..., description="价格数据", min_items=14)
) -> Dict[str, Any]:
    """
    技术分析
    
    Args:
        data: 价格数据
        
    Returns:
        技术指标结果
    """
    try:
        # 在线程池中执行技术分析
        loop = asyncio.get_event_loop()
        indicators = await loop.run_in_executor(
            executor,
            TraditionalTimeseriesAlgorithms.technical_indicators,
            data, 10
        )
        
        return {
            "status": "success",
            "indicators": indicators,
            "data_points": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Technical analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"技术分析失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查CPU资源
        cpu_cores = int(os.getenv("OMP_NUM_THREADS", "12"))
        
        return {
            "status": "healthy",
            "cpu_cores": cpu_cores,
            "algorithms_available": ["ma", "ema", "arima", "technical"],
            "vectorization_enabled": os.getenv("ENABLE_VECTORIZATION", "true") == "true",
            "scipy_available": SCIPY_AVAILABLE,
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
        import psutil
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_cores": psutil.cpu_count(),
            "threads_configured": int(os.getenv("OMP_NUM_THREADS", "12")),
            "algorithms": ["ma", "ema", "arima", "technical"],
            "hardware_target": "CPU"
        }
        
    except Exception as e:
        logger.error(f"Get metrics failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # 配置CPU优化
    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "12")
    os.environ["MKL_NUM_THREADS"] = os.getenv("MKL_NUM_THREADS", "12")
    os.environ["NUMBA_NUM_THREADS"] = os.getenv("NUMBA_NUM_THREADS", "12")
    
    # 配置服务器
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting CPU timeseries service on {host}:{port}")
    logger.info(f"CPU Cores: {os.getenv('OMP_NUM_THREADS', '12')}")
    logger.info(f"Algorithms: {os.getenv('ALGORITHM_TYPES', 'arima,garch,technical_indicators')}")
    
    # 启动服务
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        workers=1  # CPU服务使用单进程，内部多线程
    )
