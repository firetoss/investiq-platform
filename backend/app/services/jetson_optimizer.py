"""
Jetson硬件优化模块 - DLA和TensorRT加速支持
专门针对NVIDIA Jetson Orin AGX的硬件特性优化
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# 条件导入Jetson专用库
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DLAEngine:
    """
    DLA (Deep Learning Accelerator) 推理引擎
    专门用于BERT类模型的DLA加速
    """
    
    def __init__(self, dla_core: int = 0):
        self.dla_core = dla_core
        self.model = None
        self.tokenizer = None
        self.device = f"cuda:{dla_core}"  # DLA通过CUDA接口访问
        
    async def load_model(self, model_path: str, config: Dict[str, Any]) -> bool:
        """
        加载模型到DLA
        
        Args:
            model_path: 模型路径
            config: 模型配置
            
        Returns:
            是否加载成功
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                logger.error("Transformers not available for DLA")
                return False
            
            logger.info(f"Loading model to DLA core {self.dla_core}: {model_path}")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # 加载模型并设置为FP16
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": self.device}
            )
            
            # 设置为评估模式
            self.model.eval()
            
            # DLA优化设置
            if torch.cuda.is_available():
                # 启用DLA优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                
                # 预热DLA
                await self._warmup_dla()
            
            logger.info(f"Model loaded successfully to DLA core {self.dla_core}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model to DLA: {e}")
            return False
    
    async def _warmup_dla(self):
        """DLA预热，优化首次推理性能"""
        try:
            dummy_text = "这是一个测试文本用于DLA预热"
            dummy_inputs = self.tokenizer(
                dummy_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                _ = self.model(**dummy_inputs)
            
            logger.info("DLA warmup completed")
            
        except Exception as e:
            logger.warning(f"DLA warmup failed: {e}")
    
    async def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        DLA批量推理
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            预测结果列表
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded")
        
        results = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # DLA推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # 简化的情感分类（需要根据具体模型调整）
                logits = outputs.last_hidden_state.mean(dim=1)
                probabilities = torch.softmax(logits, dim=-1)
                
                for j, text in enumerate(batch_texts):
                    prob = probabilities[j].cpu().numpy()
                    
                    # 简化的三分类映射
                    if len(prob) >= 3:
                        pos_score = float(prob[2]) if len(prob) > 2 else 0.33
                        neg_score = float(prob[0]) if len(prob) > 0 else 0.33
                        neu_score = float(prob[1]) if len(prob) > 1 else 0.34
                    else:
                        pos_score = neg_score = neu_score = 0.33
                    
                    if pos_score > neg_score and pos_score > neu_score:
                        label = "POSITIVE"
                        score = pos_score
                    elif neg_score > pos_score and neg_score > neu_score:
                        label = "NEGATIVE"
                        score = neg_score
                    else:
                        label = "NEUTRAL"
                        score = neu_score
                    
                    results.append({
                        "label": label,
                        "score": score
                    })
        
        return results


class TensorRTEngine:
    """
    TensorRT推理引擎
    专门用于时间序列模型的TensorRT加速
    """
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        
    async def load_engine(self) -> bool:
        """
        加载TensorRT引擎
        
        Returns:
            是否加载成功
        """
        try:
            if not TENSORRT_AVAILABLE:
                logger.error("TensorRT not available")
                return False
            
            engine_path = Path(self.engine_path)
            if not engine_path.exists():
                logger.warning(f"TensorRT engine not found: {engine_path}")
                return False
            
            logger.info(f"Loading TensorRT engine: {engine_path}")
            
            # 加载引擎
            with open(engine_path, 'rb') as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            
            # 创建CUDA流
            self.stream = cuda.Stream()
            
            logger.info("TensorRT engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return False
    
    async def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        TensorRT推理
        
        Args:
            input_data: 输入数据
            
        Returns:
            预测结果
        """
        if not self.engine or not self.context:
            raise RuntimeError("TensorRT engine not loaded")
        
        try:
            # 分配GPU内存
            input_shape = input_data.shape
            output_shape = self._get_output_shape(input_shape)
            
            # 分配设备内存
            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
            
            # 复制输入数据到GPU
            cuda.memcpy_htod_async(d_input, input_data.astype(np.float32), self.stream)
            
            # 设置绑定
            bindings = [int(d_input), int(d_output)]
            
            # 执行推理
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            
            # 复制结果回CPU
            output_data = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
            
            # 同步流
            self.stream.synchronize()
            
            # 释放GPU内存
            d_input.free()
            d_output.free()
            
            return output_data
            
        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            raise
    
    def _get_output_shape(self, input_shape: tuple) -> tuple:
        """获取输出形状（简化实现）"""
        # 这里需要根据具体模型调整
        batch_size = input_shape[0]
        return (batch_size, 10)  # 假设预测10个时间步


class JetsonOptimizer:
    """
    Jetson硬件优化器
    统一管理DLA和TensorRT优化
    """
    
    def __init__(self):
        self.dla_engines: Dict[int, DLAEngine] = {}
        self.tensorrt_engines: Dict[str, TensorRTEngine] = {}
        
        # 检测Jetson平台
        self.is_jetson = self._detect_jetson_platform()
        
        if self.is_jetson:
            logger.info("Jetson platform detected, enabling hardware optimizations")
            self._setup_jetson_optimizations()
        else:
            logger.info("Non-Jetson platform, using standard optimizations")
    
    def _detect_jetson_platform(self) -> bool:
        """检测是否为Jetson平台"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_info = f.read().strip()
                return 'jetson' in model_info.lower() or 'orin' in model_info.lower()
        except:
            return False
    
    def _setup_jetson_optimizations(self):
        """设置Jetson优化"""
        try:
            # 设置CUDA优化
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'  # 2GB
            
            # 设置DLA优化
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
            
            logger.info("Jetson optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply Jetson optimizations: {e}")
    
    async def create_dla_engine(self, dla_core: int = 0) -> DLAEngine:
        """创建DLA引擎"""
        if dla_core not in self.dla_engines:
            self.dla_engines[dla_core] = DLAEngine(dla_core)
        
        return self.dla_engines[dla_core]
    
    async def create_tensorrt_engine(self, engine_path: str) -> TensorRTEngine:
        """创建TensorRT引擎"""
        if engine_path not in self.tensorrt_engines:
            self.tensorrt_engines[engine_path] = TensorRTEngine(engine_path)
        
        return self.tensorrt_engines[engine_path]
    
    def get_optimization_recommendations(self, model_type: str, model_size_mb: int) -> Dict[str, Any]:
        """获取优化建议"""
        recommendations = {
            "hardware_target": "gpu",
            "precision": "fp16",
            "batch_size": 1,
            "optimizations": []
        }
        
        if self.is_jetson:
            if model_type == "sentiment" and model_size_mb < 2000:
                recommendations.update({
                    "hardware_target": "dla",
                    "precision": "fp16",
                    "batch_size": 32,
                    "optimizations": ["dla_acceleration", "batch_optimization"]
                })
            
            elif model_type == "timeseries" and model_size_mb < 5000:
                recommendations.update({
                    "hardware_target": "tensorrt",
                    "precision": "fp16",
                    "batch_size": 16,
                    "optimizations": ["tensorrt_optimization", "workspace_tuning"]
                })
            
            elif model_type == "llm":
                recommendations.update({
                    "hardware_target": "gpu",
                    "precision": "int8",
                    "batch_size": 1,
                    "optimizations": ["gpu_layers_35", "memory_mapping", "context_optimization"]
                })
        
        return recommendations


# 全局Jetson优化器实例
jetson_optimizer = JetsonOptimizer()
