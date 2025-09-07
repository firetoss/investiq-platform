"""
InvestIQ时序预测服务 - CPU专用传统算法
支持ARIMA, GARCH, 技术指标等传统金融算法
"""

import asyncio
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
import logging

# 统计模型
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

# 技术指标
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from investiq_common.models import (
    TimeSeriesRequest, TimeSeriesResponse, TimeSeriesPrediction,
    HealthCheck, ServiceStatus
)
from investiq_common.utils import setup_logging, get_logger, performance_timer


logger = get_logger(__name__)

# 忽略统计模型的警告
warnings.filterwarnings('ignore')


class ModelType(str, Enum):
    """支持的模型类型"""
    ARIMA = "arima"
    SARIMA = "sarima"
    GARCH = "garch"
    MOVING_AVERAGE = "ma"
    EXPONENTIAL_SMOOTHING = "es"
    LINEAR_REGRESSION = "linear"


class TimeSeriesAnalyzer:
    """时序分析器"""
    
    def __init__(self):
        self.is_ready = True  # CPU算法无需加载模型
    
    @performance_timer(logger)
    async def predict(self, request: TimeSeriesRequest) -> TimeSeriesPrediction:
        """时序预测"""
        try:
            data = np.array(request.data)
            
            # 数据验证
            if len(data) < 10:
                raise ValueError("Insufficient data points (minimum 10)")
            
            # 根据模型类型进行预测
            model_type = ModelType(request.model_type.lower())
            
            if model_type == ModelType.ARIMA:
                predictions, confidence_intervals, model_info = await self._arima_predict(
                    data, request.prediction_steps
                )
            elif model_type == ModelType.GARCH:
                predictions, confidence_intervals, model_info = await self._garch_predict(
                    data, request.prediction_steps
                )
            elif model_type == ModelType.MOVING_AVERAGE:
                predictions, confidence_intervals, model_info = await self._ma_predict(
                    data, request.prediction_steps
                )
            elif model_type == ModelType.LINEAR_REGRESSION:
                predictions, confidence_intervals, model_info = await self._linear_predict(
                    data, request.prediction_steps
                )
            else:
                # 默认使用ARIMA
                predictions, confidence_intervals, model_info = await self._arima_predict(
                    data, request.prediction_steps
                )
            
            return TimeSeriesPrediction(
                symbol=request.symbol,
                prediction_type="price",
                predictions=predictions.tolist(),
                confidence_intervals=confidence_intervals,
                model_info=model_info
            )
        
        except Exception as e:
            logger.error(f"Time series prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def _arima_predict(
        self, 
        data: np.ndarray, 
        steps: int
    ) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, Any]]:
        """ARIMA预测"""
        try:
            # 自动选择ARIMA参数
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            # 简化的参数搜索
            for p in range(1, 4):
                for d in range(0, 2):
                    for q in range(1, 4):
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            fitted_model = model.fit(method_kwargs={"maxiter": 50, "disp": False})
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            # 使用最佳参数拟合模型
            model = ARIMA(data, order=best_params)
            fitted_model = model.fit(method_kwargs={"maxiter": 100, "disp": False})
            
            # 预测
            forecast = fitted_model.forecast(steps=steps)
            forecast_ci = fitted_model.get_forecast(steps=steps).conf_int()
            
            # 置信区间
            confidence_intervals = []
            for i in range(len(forecast)):
                confidence_intervals.append({
                    "lower": float(forecast_ci.iloc[i, 0]),
                    "upper": float(forecast_ci.iloc[i, 1])
                })
            
            model_info = {
                "model_type": "ARIMA",
                "parameters": {"p": best_params[0], "d": best_params[1], "q": best_params[2]},
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "mse": float(np.mean(fitted_model.resid**2))
            }
            
            return forecast.values, confidence_intervals, model_info
        
        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            # 回退到移动平均
            return await self._ma_predict(data, steps)
    
    async def _garch_predict(
        self,
        data: np.ndarray,
        steps: int
    ) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, Any]]:
        """GARCH波动率预测"""
        try:
            # 计算收益率
            returns = np.diff(np.log(data)) * 100  # 对数收益率 * 100
            
            if len(returns) < 20:
                raise ValueError("Insufficient data for GARCH model")
            
            # 拟合GARCH模型
            model = arch_model(returns, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # 预测波动率
            volatility_forecast = fitted_model.forecast(horizon=steps)
            
            # 基于波动率预测价格区间
            last_price = data[-1]
            predicted_returns = np.zeros(steps)
            predicted_volatilities = np.sqrt(volatility_forecast.variance.values[-1, :])
            
            # 简单的价格预测（使用随机游走 + 波动率）
            predictions = []
            confidence_intervals = []
            
            current_price = last_price
            for i in range(steps):
                vol = predicted_volatilities[i] / 100  # 转换回小数
                
                # 价格预测（假设收益率为0）
                predictions.append(current_price)
                
                # 置信区间（基于波动率）
                lower = current_price * (1 - 1.96 * vol)
                upper = current_price * (1 + 1.96 * vol)
                
                confidence_intervals.append({
                    "lower": float(lower),
                    "upper": float(upper)
                })
            
            model_info = {
                "model_type": "GARCH",
                "parameters": {"p": 1, "q": 1},
                "log_likelihood": float(fitted_model.loglikelihood),
                "avg_volatility": float(np.mean(predicted_volatilities))
            }
            
            return np.array(predictions), confidence_intervals, model_info
        
        except Exception as e:
            logger.error(f"GARCH prediction failed: {e}")
            return await self._ma_predict(data, steps)
    
    async def _ma_predict(
        self,
        data: np.ndarray,
        steps: int
    ) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, Any]]:
        """移动平均预测"""
        # 计算不同周期的移动平均
        ma_5 = np.mean(data[-5:]) if len(data) >= 5 else np.mean(data)
        ma_20 = np.mean(data[-20:]) if len(data) >= 20 else np.mean(data)
        
        # 趋势计算
        if len(data) >= 2:
            trend = (data[-1] - data[-min(10, len(data))]) / min(10, len(data))
        else:
            trend = 0
        
        # 简单预测：移动平均 + 趋势
        base_prediction = (ma_5 + ma_20) / 2
        
        predictions = []
        confidence_intervals = []
        
        # 计算历史波动率
        if len(data) >= 2:
            returns = np.diff(data) / data[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0.02  # 默认2%波动率
        
        for i in range(steps):
            pred = base_prediction + trend * i
            predictions.append(pred)
            
            # 基于波动率的置信区间
            margin = 1.96 * volatility * pred
            confidence_intervals.append({
                "lower": float(pred - margin),
                "upper": float(pred + margin)
            })
        
        model_info = {
            "model_type": "Moving Average",
            "ma_5": float(ma_5),
            "ma_20": float(ma_20),
            "trend": float(trend),
            "volatility": float(volatility)
        }
        
        return np.array(predictions), confidence_intervals, model_info
    
    async def _linear_predict(
        self,
        data: np.ndarray,
        steps: int
    ) -> Tuple[np.ndarray, List[Dict[str, float]], Dict[str, Any]]:
        """线性回归预测"""
        from sklearn.linear_model import LinearRegression
        
        # 准备数据
        X = np.arange(len(data)).reshape(-1, 1)
        y = data
        
        # 拟合线性回归
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测
        future_X = np.arange(len(data), len(data) + steps).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        # 计算残差的标准差作为置信区间
        y_pred_train = model.predict(X)
        residuals = y - y_pred_train
        std_residuals = np.std(residuals)
        
        confidence_intervals = []
        for pred in predictions:
            margin = 1.96 * std_residuals
            confidence_intervals.append({
                "lower": float(pred - margin),
                "upper": float(pred + margin)
            })
        
        model_info = {
            "model_type": "Linear Regression",
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "r2_score": float(model.score(X, y)),
            "residual_std": float(std_residuals)
        }
        
        return predictions, confidence_intervals, model_info
    
    async def calculate_technical_indicators(self, data: np.ndarray) -> Dict[str, float]:
        """计算技术指标"""
        if len(data) < 2:
            return {}
        
        indicators = {}
        
        # 移动平均
        if len(data) >= 5:
            indicators["ma_5"] = float(np.mean(data[-5:]))
        if len(data) >= 20:
            indicators["ma_20"] = float(np.mean(data[-20:]))
        
        # RSI
        if len(data) >= 14:
            indicators["rsi"] = float(self._calculate_rsi(data, 14))
        
        # 布林带
        if len(data) >= 20:
            bb_upper, bb_lower = self._calculate_bollinger_bands(data, 20, 2)
            indicators["bollinger_upper"] = float(bb_upper)
            indicators["bollinger_lower"] = float(bb_lower)
        
        # 波动率
        if len(data) >= 2:
            returns = np.diff(data) / data[:-1]
            indicators["volatility"] = float(np.std(returns))
        
        return indicators
    
    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> float:
        """计算RSI指标"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, data: np.ndarray, period: int, std_dev: float) -> Tuple[float, float]:
        """计算布林带"""
        ma = np.mean(data[-period:])
        std = np.std(data[-period:])
        
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        
        return upper, lower


# 创建应用和分析器
app = FastAPI(
    title="InvestIQ TimeSeries Service",
    description="基于传统统计模型的时序预测服务",
    version="1.0.0"
)

analyzer = TimeSeriesAnalyzer()

# 配置日志
setup_logging()


@app.get("/health")
async def health_check():
    """健康检查"""
    # 测试依赖库
    checks = {
        "numpy_available": True,
        "pandas_available": True,
        "statsmodels_available": True,
        "arch_available": True,
        "analyzer_ready": analyzer.is_ready
    }
    
    try:
        import numpy, pandas, statsmodels, arch
        from sklearn.linear_model import LinearRegression
        checks["sklearn_available"] = True
    except ImportError as e:
        checks["sklearn_available"] = False
        logger.error(f"Import error: {e}")
    
    overall_healthy = all(checks.values())
    status = ServiceStatus.HEALTHY if overall_healthy else ServiceStatus.UNHEALTHY
    
    return HealthCheck(
        status=status,
        version="1.0.0",
        checks=checks,
        metadata={
            "supported_models": ["arima", "garch", "ma", "linear"],
            "cpu_optimized": True
        }
    )


@app.post("/v1/predict", response_model=TimeSeriesResponse)
async def predict_timeseries(request: TimeSeriesRequest) -> TimeSeriesResponse:
    """时序预测"""
    start_time = time.time()
    
    try:
        prediction = await analyzer.predict(request)
        processing_time = time.time() - start_time
        
        # 计算技术指标
        technical_indicators = await analyzer.calculate_technical_indicators(np.array(request.data))
        
        logger.info(f"Predicted {request.prediction_steps} steps for {request.symbol} in {processing_time:.3f}s")
        
        return TimeSeriesResponse(
            prediction=prediction,
            model_metrics=technical_indicators,
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time series prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/v1/indicators/{symbol}")
async def get_technical_indicators(symbol: str, data: List[float]) -> Dict[str, float]:
    """获取技术指标"""
    try:
        if not data or len(data) < 2:
            raise HTTPException(status_code=400, detail="Insufficient data")
        
        indicators = await analyzer.calculate_technical_indicators(np.array(data))
        return indicators
    
    except Exception as e:
        logger.error(f"Technical indicators calculation failed: {e}")
        raise HTTPException(status_code=500, detail="Calculation failed")


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )