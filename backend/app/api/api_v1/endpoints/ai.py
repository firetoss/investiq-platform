"""
AI相关API端点
提供LLM推理、情感分析、时间序列预测等功能
"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import logging

from backend.app.services.ai_service import (
    ai_service,
    LLMRequest,
    LLMResponse,
    SentimentRequest,
    SentimentResponse,
    TimeseriesRequest,
    TimeseriesResponse
)
from backend.app.core.deps import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/llm/inference", response_model=LLMResponse)
async def llm_inference(
    request: LLMRequest,
    model_name: str = "qwen3-8b",
    current_user: User = Depends(get_current_user)
) -> LLMResponse:
    """
    LLM推理接口
    
    Args:
        request: LLM请求参数
        model_name: 模型名称
        current_user: 当前用户
        
    Returns:
        LLM推理结果
    """
    try:
        logger.info(f"User {current_user.id} requesting LLM inference with model {model_name}")
        
        response = await ai_service.llm_inference(request, model_name)
        
        logger.info(f"LLM inference completed in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"LLM inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM推理失败: {str(e)}")


@router.post("/sentiment/analyze", response_model=SentimentResponse)
async def sentiment_analysis(
    request: SentimentRequest,
    model_name: str = "chinese-financial-bert",
    current_user: User = Depends(get_current_user)
) -> SentimentResponse:
    """
    情感分析接口
    
    Args:
        request: 情感分析请求参数
        model_name: 模型名称
        current_user: 当前用户
        
    Returns:
        情感分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting sentiment analysis with model {model_name}")
        
        response = await ai_service.sentiment_analysis(request, model_name)
        
        logger.info(f"Sentiment analysis completed in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"情感分析失败: {str(e)}")


@router.post("/timeseries/forecast", response_model=TimeseriesResponse)
async def timeseries_forecast(
    request: TimeseriesRequest,
    model_name: str = "chronos-base",
    current_user: User = Depends(get_current_user)
) -> TimeseriesResponse:
    """
    时间序列预测接口
    
    Args:
        request: 时间序列预测请求参数
        model_name: 模型名称
        current_user: 当前用户
        
    Returns:
        时间序列预测结果
    """
    try:
        logger.info(f"User {current_user.id} requesting timeseries forecast with model {model_name}")
        
        response = await ai_service.timeseries_forecast(request, model_name)
        
        logger.info(f"Timeseries forecast completed in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Timeseries forecast failed: {e}")
        raise HTTPException(status_code=500, detail=f"时间序列预测失败: {str(e)}")


@router.post("/analysis/policy")
async def analyze_policy(
    policy_text: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    政策分析接口
    
    Args:
        policy_text: 政策文本
        current_user: 当前用户
        
    Returns:
        政策分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting policy analysis")
        
        if not policy_text.strip():
            raise HTTPException(status_code=400, detail="政策文本不能为空")
        
        result = await ai_service.analyze_policy(policy_text)
        
        logger.info(f"Policy analysis completed in {result['processing_time']:.2f}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"政策分析失败: {str(e)}")


@router.post("/analysis/company")
async def analyze_company(
    company_name: str,
    industry: str,
    financial_data: str,
    news: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    公司分析接口
    
    Args:
        company_name: 公司名称
        industry: 行业
        financial_data: 财务数据
        news: 最新新闻
        current_user: 当前用户
        
    Returns:
        公司分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting company analysis for {company_name}")
        
        if not all([company_name.strip(), industry.strip()]):
            raise HTTPException(status_code=400, detail="公司名称和行业不能为空")
        
        result = await ai_service.analyze_company(
            company_name, industry, financial_data, news
        )
        
        logger.info(f"Company analysis completed in {result['processing_time']:.2f}s")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Company analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"公司分析失败: {str(e)}")


@router.post("/analysis/market")
async def summarize_market(
    market_data: str,
    news_summary: str,
    policy_updates: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    市场总结接口
    
    Args:
        market_data: 市场数据
        news_summary: 新闻摘要
        policy_updates: 政策动态
        current_user: 当前用户
        
    Returns:
        市场总结结果
    """
    try:
        logger.info(f"User {current_user.id} requesting market summary")
        
        result = await ai_service.summarize_market(
            market_data, news_summary, policy_updates
        )
        
        logger.info(f"Market summary completed in {result['processing_time']:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Market summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"市场总结失败: {str(e)}")


@router.get("/models/status")
async def get_model_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取模型状态
    
    Args:
        current_user: 当前用户
        
    Returns:
        所有模型的状态信息
    """
    try:
        logger.info(f"User {current_user.id} requesting model status")
        
        status = await ai_service.get_model_status()
        
        return {
            "status": "success",
            "models": status,
            "total_models": len(status),
            "loaded_models": sum(1 for s in status.values() if s.loaded)
        }
        
    except Exception as e:
        logger.error(f"Get model status failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型状态失败: {str(e)}")


@router.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    加载模型
    
    Args:
        model_name: 模型名称
        background_tasks: 后台任务
        current_user: 当前用户
        
    Returns:
        加载结果
    """
    try:
        logger.info(f"User {current_user.id} requesting to load model {model_name}")
        
        # 在后台加载模型
        background_tasks.add_task(ai_service.load_model, model_name)
        
        return {
            "status": "success",
            "message": f"模型 {model_name} 正在后台加载",
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Load model failed: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@router.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    卸载模型
    
    Args:
        model_name: 模型名称
        current_user: 当前用户
        
    Returns:
        卸载结果
    """
    try:
        logger.info(f"User {current_user.id} requesting to unload model {model_name}")
        
        success = await ai_service.unload_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"模型 {model_name} 已成功卸载",
                "model_name": model_name
            }
        else:
            raise HTTPException(status_code=400, detail=f"模型 {model_name} 卸载失败")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unload model failed: {e}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    AI服务健康检查
    
    Returns:
        健康状态信息
    """
    try:
        # 获取模型状态
        status = await ai_service.get_model_status()
        
        # 计算健康指标
        total_models = len(status)
        loaded_models = sum(1 for s in status.values() if s.loaded)
        error_models = sum(1 for s in status.values() if s.error)
        
        health_score = (loaded_models / total_models * 100) if total_models > 0 else 0
        
        return {
            "status": "healthy" if health_score >= 50 else "degraded",
            "health_score": health_score,
            "total_models": total_models,
            "loaded_models": loaded_models,
            "error_models": error_models,
            "timestamp": "2025-01-04T15:42:00+08:00"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-04T15:42:00+08:00"
        }
