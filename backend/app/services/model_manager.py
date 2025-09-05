"""
模型管理器 - 管理llama.cpp和其他AI模型的加载、卸载和推理
支持动态模型切换和内存优化
"""

import asyncio
import os
import threading
from typing import Dict, Optional, Any, List, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging
import psutil
import time

from pydantic import BaseModel, Field
import torch

from .jetson_optimizer import jetson_optimizer, DLAEngine, TensorRTEngine

# 条件导入，避免在没有安装时报错
try:
    from llama_cpp import Llama, LlamaGrammar
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    LlamaGrammar = None

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """模型类型枚举"""
    LLM = "llm"  # 大语言模型
    SENTIMENT = "sentiment"  # 情感分析
    TIMESERIES = "timeseries"  # 时间序列
    EMBEDDING = "embedding"  # 嵌入模型


class ModelPrecision(str, Enum):
    """模型精度枚举"""
    FP32 = "fp32"  # 32位浮点
    FP16 = "fp16"  # 16位浮点
    INT8 = "int8"  # 8位整数
    INT4 = "int4"  # 4位整数


class HardwareTarget(str, Enum):
    """硬件目标枚举"""
    CPU = "cpu"  # CPU推理
    GPU = "gpu"  # GPU推理
    DLA = "dla"  # DLA推理
    TENSORRT = "tensorrt"  # TensorRT推理


@dataclass
class ModelConfig:
    """增强的模型配置"""
    name: str
    model_type: ModelType
    model_path: str
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 1  # 优先级，数字越小优先级越高
    precision: ModelPrecision = ModelPrecision.FP16  # 模型精度
    hardware_target: HardwareTarget = HardwareTarget.GPU  # 硬件目标
    memory_estimate: int = 1000  # 预估内存使用(MB)
    dla_core: Optional[int] = None  # DLA核心ID (0或1)
    tensorrt_engine_path: Optional[str] = None  # TensorRT引擎路径


class ModelStatus(BaseModel):
    """模型状态"""
    name: str
    model_type: ModelType
    loaded: bool = False
    memory_usage: Optional[int] = None  # MB
    last_used: Optional[str] = None
    error: Optional[str] = None


class ModelManager:
    """
    模型管理器
    
    功能：
    1. 动态加载/卸载模型
    2. 内存管理和优化
    3. 模型状态监控
    4. 推理接口统一
    """
    
    def __init__(self, models_dir: str = "/models", max_memory_mb: int = 32000):
        self.models_dir = Path(models_dir)
        self.max_memory_mb = max_memory_mb
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.status: Dict[str, ModelStatus] = {}
        self._lock = threading.Lock()
        
        # 硬件能力检测
        self.hardware_capabilities = self._detect_hardware_capabilities()
        
        # 默认模型配置
        self._setup_default_configs()
        
        logger.info(f"ModelManager initialized with capabilities: {self.hardware_capabilities}")
    
    def _detect_hardware_capabilities(self) -> Dict[str, Any]:
        """检测硬件能力"""
        capabilities = {
            "cpu_cores": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "dla_available": False,
            "tensorrt_available": TENSORRT_AVAILABLE,
            "jetson_platform": False
        }
        
        # 检测Jetson平台
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_info = f.read().strip()
                if 'jetson' in model_info.lower() or 'orin' in model_info.lower():
                    capabilities["jetson_platform"] = True
                    capabilities["dla_available"] = True  # Jetson Orin有DLA
        except:
            pass
        
        # 检测GPU信息
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                capabilities["gpu_name"] = gpu_name
                capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
        
        return capabilities
    
    def _check_memory_available(self, required_mb: int = 0) -> bool:
        """增强的内存检查"""
        # 获取当前内存使用
        current_usage = sum(
            self.configs[name].memory_estimate 
            for name in self.models.keys() 
            if name in self.configs
        )
        
        # 检查系统内存
        system_memory = psutil.virtual_memory()
        available_mb = system_memory.available / (1024**2)
        
        # 检查GPU内存
        gpu_available_mb = 0
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                gpu_used = torch.cuda.memory_allocated(0) / (1024**2)
                gpu_available_mb = gpu_memory - gpu_used
            except:
                pass
        
        # 综合判断
        total_required = current_usage + required_mb
        memory_ok = (
            total_required <= self.max_memory_mb and
            available_mb >= required_mb and
            (gpu_available_mb >= required_mb if required_mb > 1000 else True)
        )
        
        logger.debug(f"Memory check: required={required_mb}MB, current={current_usage}MB, "
                    f"available_system={available_mb:.0f}MB, available_gpu={gpu_available_mb:.0f}MB, "
                    f"result={memory_ok}")
        
        return memory_ok
    
    def _get_optimal_hardware_target(self, config: ModelConfig) -> HardwareTarget:
        """根据硬件能力选择最优硬件目标"""
        # 如果指定的硬件不可用，自动降级
        target = config.hardware_target
        
        if target == HardwareTarget.DLA and not self.hardware_capabilities.get("dla_available"):
            logger.warning(f"DLA not available for {config.name}, falling back to GPU")
            target = HardwareTarget.GPU
        
        if target == HardwareTarget.TENSORRT and not self.hardware_capabilities.get("tensorrt_available"):
            logger.warning(f"TensorRT not available for {config.name}, falling back to GPU")
            target = HardwareTarget.GPU
        
        if target == HardwareTarget.GPU and not self.hardware_capabilities.get("gpu_available"):
            logger.warning(f"GPU not available for {config.name}, falling back to CPU")
            target = HardwareTarget.CPU
        
        return target
        
    def _setup_default_configs(self):
        """设置默认模型配置 - 方案A优化"""
        default_configs = [
            # LLM: Qwen3-8B INT8量化 (~12GB)
            ModelConfig(
                name="qwen3-8b",
                model_type=ModelType.LLM,
                model_path=str(self.models_dir / "Qwen3-8B-INT8.gguf"),
                precision=ModelPrecision.INT8,
                hardware_target=HardwareTarget.GPU,
                memory_estimate=12000,  # 12GB
                config={
                    "n_ctx": 8192,
                    "n_gpu_layers": 35,  # Jetson Orin AGX优化
                    "n_threads": 8,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                    "verbose": False,
                    "use_mlock": True,  # 锁定内存，提高性能
                    "use_mmap": True,   # 内存映射，减少内存占用
                }
            ),
            # 情感分析: RoBERTa英文金融版 FP16 + DLA0 (~1.2GB) - 精度优先
            ModelConfig(
                name="roberta-financial-en",
                model_type=ModelType.SENTIMENT,
                model_path="Jean-Baptiste/roberta-large-financial-news-sentiment-en",
                precision=ModelPrecision.FP16,
                hardware_target=HardwareTarget.DLA,
                memory_estimate=1200,  # 1.2GB，精度优先
                dla_core=0,  # 使用DLA核心0
                config={
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "return_all_scores": True,
                    "batch_size": 32,  # DLA优化批处理
                    "max_length": 512,
                    "truncation": True,
                    "padding": True
                }
            ),
            # 时序预测: PatchTST金融版 FP16 + DLA1 (~500MB) - 金融专用
            ModelConfig(
                name="patchtst-financial",
                model_type=ModelType.TIMESERIES,
                model_path="ibm-granite/granite-timeseries-patchtst",
                precision=ModelPrecision.FP16,
                hardware_target=HardwareTarget.DLA,
                memory_estimate=500,  # 500MB，轻量高效
                dla_core=1,  # 使用DLA核心1
                config={
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "batch_size": 32,
                    "context_length": 512,
                    "prediction_length": 96,
                    "patch_size": 16
                }
            ),
        ]
        
        for config in default_configs:
            self.configs[config.name] = config
            self.status[config.name] = ModelStatus(
                name=config.name,
                model_type=config.model_type
            )
    
    async def load_model(self, model_name: str) -> bool:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否加载成功
        """
        if model_name not in self.configs:
            logger.error(f"Model config not found: {model_name}")
            return False
            
        config = self.configs[model_name]
        
        if not config.enabled:
            logger.warning(f"Model disabled: {model_name}")
            return False
            
        with self._lock:
            try:
                # 检查内存使用
                # 使用配置中的预估内存进行更精确的判定
                required_mb = self.configs[model_name].memory_estimate if model_name in self.configs else 0
                if not self._check_memory_available(required_mb=required_mb):
                    logger.warning("Memory limit reached, unloading least used models")
                    await self._unload_least_used_models()
                
                # 根据模型类型加载
                if config.model_type == ModelType.LLM:
                    model = await self._load_llm_model(config)
                elif config.model_type == ModelType.SENTIMENT:
                    model = await self._load_sentiment_model(config)
                elif config.model_type == ModelType.TIMESERIES:
                    model = await self._load_timeseries_model(config)
                else:
                    raise ValueError(f"Unsupported model type: {config.model_type}")
                
                self.models[model_name] = model
                self.status[model_name].loaded = True
                self.status[model_name].error = None
                
                logger.info(f"Model loaded successfully: {model_name}")
                return True
                
            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {str(e)}"
                logger.error(error_msg)
                self.status[model_name].error = error_msg
                return False
    
    async def _load_llm_model(self, config: ModelConfig) -> Any:
        """加载LLM模型"""
        if not LLAMA_CPP_AVAILABLE:
            # Mock模式，用于开发测试
            logger.warning(f"llama-cpp-python not available, using mock for {config.name}")
            return MockLlamaModel(config.name)
        
        # 检查模型文件是否存在
        model_path = Path(config.model_path)
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}, using mock")
            return MockLlamaModel(config.name)
        
        # 加载真实模型
        model = Llama(
            model_path=str(model_path),
            **config.config
        )
        return model
    
    async def _load_sentiment_model(self, config: ModelConfig) -> Any:
        """加载情感分析模型 - 支持DLA加速"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(f"transformers not available, using mock for {config.name}")
            return MockSentimentModel(config.name)
        
        # 获取最优硬件目标
        optimal_target = self._get_optimal_hardware_target(config)
        
        try:
            if optimal_target == HardwareTarget.DLA and config.dla_core is not None:
                # 使用DLA加速
                logger.info(f"Loading {config.name} with DLA acceleration")
                dla_engine = await jetson_optimizer.create_dla_engine(config.dla_core)
                success = await dla_engine.load_model(config.model_path, config.config)
                
                if success:
                    return DLAModelWrapper(dla_engine, config.name)
                else:
                    logger.warning(f"DLA loading failed for {config.name}, falling back to standard")
            
            # 标准加载方式
            model = pipeline(
                "text-classification",
                model=config.model_path,
                **config.config
            )
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load real model, using mock: {e}")
            return MockSentimentModel(config.name)
    
    async def _load_timeseries_model(self, config: ModelConfig) -> Any:
        """加载时间序列模型 - 支持TensorRT加速"""
        # 获取最优硬件目标
        optimal_target = self._get_optimal_hardware_target(config)
        
        try:
            if optimal_target == HardwareTarget.TENSORRT and config.tensorrt_engine_path:
                # 使用TensorRT加速
                logger.info(f"Loading {config.name} with TensorRT acceleration")
                trt_engine = await jetson_optimizer.create_tensorrt_engine(config.tensorrt_engine_path)
                success = await trt_engine.load_engine()
                
                if success:
                    return TensorRTModelWrapper(trt_engine, config.name)
                else:
                    logger.warning(f"TensorRT loading failed for {config.name}, falling back to mock")
            
            # 暂时使用Mock模式
            logger.warning(f"Using mock timeseries model for {config.name}")
            return MockTimeseriesModel(config.name)
            
        except Exception as e:
            logger.warning(f"Failed to load timeseries model: {e}")
            return MockTimeseriesModel(config.name)
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        with self._lock:
            if model_name in self.models:
                try:
                    # 清理模型资源
                    model = self.models[model_name]
                    if hasattr(model, 'close'):
                        model.close()
                    
                    del self.models[model_name]
                    self.status[model_name].loaded = False
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Model unloaded: {model_name}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to unload model {model_name}: {e}")
                    return False
            
            return True
    
    async def get_model(self, model_name: str) -> Optional[Any]:
        """获取模型实例"""
        if model_name not in self.models:
            # 自动加载模型
            success = await self.load_model(model_name)
            if not success:
                return None
        
        return self.models.get(model_name)
    
    def get_status(self, model_name: Optional[str] = None) -> Union[ModelStatus, Dict[str, ModelStatus]]:
        """获取模型状态"""
        if model_name:
            return self.status.get(model_name)
        return self.status.copy()

    
    async def _unload_least_used_models(self):
        """卸载最少使用的模型"""
        # 简化实现：卸载第一个加载的模型
        if self.models:
            model_name = next(iter(self.models))
            await self.unload_model(model_name)
    
    async def shutdown(self):
        """关闭管理器，清理所有资源"""
        logger.info("Shutting down model manager...")
        
        for model_name in list(self.models.keys()):
            await self.unload_model(model_name)
        
        logger.info("Model manager shutdown complete")


# Mock模型类，用于开发和测试
class MockLlamaModel:
    """Mock LLM模型"""
    
    def __init__(self, name: str):
        self.name = name
        
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """模拟推理"""
        return {
            "choices": [{
                "text": f"[Mock Response from {self.name}] 这是一个模拟的回答，用于开发测试。输入: {prompt[:50]}..."
            }]
        }
    
    def create_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """兼容llama.cpp接口"""
        return self(prompt, **kwargs)


class MockSentimentModel:
    """Mock情感分析模型"""
    
    def __init__(self, name: str):
        self.name = name
        
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """模拟情感分析"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        for text in texts:
            # 简单的规则判断
            if any(word in text for word in ["好", "涨", "利好", "增长", "盈利"]):
                sentiment = "POSITIVE"
                score = 0.8
            elif any(word in text for word in ["坏", "跌", "利空", "下降", "亏损"]):
                sentiment = "NEGATIVE"
                score = 0.8
            else:
                sentiment = "NEUTRAL"
                score = 0.6
                
            results.append({
                "label": sentiment,
                "score": score
            })
        
        return results


class MockTimeseriesModel:
    """Mock时间序列模型"""
    
    def __init__(self, name: str):
        self.name = name
        
    def predict(self, data: List[float], horizon: int = 10) -> List[float]:
        """模拟时间序列预测"""
        import random
        
        if not data:
            return [random.uniform(0.9, 1.1) for _ in range(horizon)]
        
        last_value = data[-1]
        predictions = []
        
        for i in range(horizon):
            # 简单的随机游走
            change = random.uniform(-0.05, 0.05)
            next_value = last_value * (1 + change)
            predictions.append(next_value)
            last_value = next_value
        
        return predictions


class DLAModelWrapper:
    """DLA模型包装器，提供统一接口"""
    
    def __init__(self, dla_engine: DLAEngine, name: str):
        self.dla_engine = dla_engine
        self.name = name
    
    def __call__(self, texts: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """统一的调用接口"""
        if isinstance(texts, str):
            texts = [texts]
        
        # 使用asyncio运行异步方法
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.dla_engine.predict(texts))
        except RuntimeError:
            # 如果没有事件循环，创建新的
            return asyncio.run(self.dla_engine.predict(texts))
    
    def close(self):
        """清理资源"""
        # DLA引擎清理
        if hasattr(self.dla_engine, 'model') and self.dla_engine.model:
            del self.dla_engine.model
        if hasattr(self.dla_engine, 'tokenizer') and self.dla_engine.tokenizer:
            del self.dla_engine.tokenizer


class TensorRTModelWrapper:
    """TensorRT模型包装器，提供统一接口"""
    
    def __init__(self, trt_engine: TensorRTEngine, name: str):
        self.trt_engine = trt_engine
        self.name = name
    
    def predict(self, data: List[float], horizon: int = 10) -> List[float]:
        """时间序列预测接口"""
        try:
            # 准备输入数据
            import numpy as np
            input_data = np.array(data, dtype=np.float32).reshape(1, -1)
            
            # 使用asyncio运行异步方法
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(self.trt_engine.predict(input_data))
            except RuntimeError:
                output = asyncio.run(self.trt_engine.predict(input_data))
            
            # 转换输出格式
            if len(output.shape) > 1:
                predictions = output[0][:horizon].tolist()
            else:
                predictions = output[:horizon].tolist()
            
            return predictions
            
        except Exception as e:
            logger.error(f"TensorRT prediction failed: {e}")
            # 降级到简单预测
            import random
            last_value = data[-1] if data else 1.0
            predictions = []
            for i in range(horizon):
                change = random.uniform(-0.02, 0.02)
                next_value = last_value * (1 + change)
                predictions.append(next_value)
                last_value = next_value
            return predictions
    
    def close(self):
        """清理资源"""
        # TensorRT引擎清理
        if hasattr(self.trt_engine, 'context') and self.trt_engine.context:
            del self.trt_engine.context
        if hasattr(self.trt_engine, 'engine') and self.trt_engine.engine:
            del self.trt_engine.engine


# 全局模型管理器实例
model_manager = ModelManager()
