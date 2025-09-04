"""
智能分析API端点
提供政策分析、公司分析、市场情感分析、趋势预测等功能
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends
import logging

from pydantic import BaseModel, Field

from backend.app.services.intelligent_analysis import (
    intelligent_analysis,
    PolicyAnalysisResult,
    CompanyAnalysisResult,
    MarketSentimentResult,
    TrendPredictionResult
)
from backend.app.core.deps import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class PolicyAnalysisRequest(BaseModel):
    """政策分析请求"""
    policy_text: str = Field(..., description="政策文本内容", min_length=10)


class CompanyAnalysisRequest(BaseModel):
    """公司分析请求"""
    company_name: str = Field(..., description="公司名称", min_length=1)
    industry: str = Field(..., description="所属行业", min_length=1)
    financial_data: str = Field(default="", description="财务数据")
    recent_news: str = Field(default="", description="最新新闻")


class MarketSentimentRequest(BaseModel):
    """市场情感分析请求"""
    news_texts: List[str] = Field(..., description="新闻文本列表", min_items=1)
    market_data: Optional[str] = Field(default=None, description="市场数据（可选）")


class TrendPredictionRequest(BaseModel):
    """趋势预测请求"""
    symbol: str = Field(..., description="股票代码", min_length=1)
    price_data: List[float] = Field(..., description="历史价格数据", min_items=5)
    horizon: int = Field(default=10, description="预测期数", ge=1, le=30)


@router.post("/policy", response_model=PolicyAnalysisResult)
async def analyze_policy(
    request: PolicyAnalysisRequest,
    current_user: User = Depends(get_current_user)
) -> PolicyAnalysisResult:
    """
    政策影响分析
    
    分析政策文件对投资的影响，包括：
    - 受益行业识别
    - 政策强度评分
    - 实施确定性评估
    - 投资建议生成
    - 风险因素识别
    
    Args:
        request: 政策分析请求
        current_user: 当前用户
        
    Returns:
        政策分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting policy analysis")
        
        result = await intelligent_analysis.analyze_policy_impact(request.policy_text)
        
        logger.info(f"Policy analysis completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Policy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"政策分析失败: {str(e)}")


@router.post("/company", response_model=CompanyAnalysisResult)
async def analyze_company(
    request: CompanyAnalysisRequest,
    current_user: User = Depends(get_current_user)
) -> CompanyAnalysisResult:
    """
    公司基本面分析
    
    分析公司的基本面情况，包括：
    - 基本面评分
    - 市场地位评估
    - 竞争优势识别
    - 风险因素分析
    - 投资评级建议
    
    Args:
        request: 公司分析请求
        current_user: 当前用户
        
    Returns:
        公司分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting company analysis for {request.company_name}")
        
        result = await intelligent_analysis.analyze_company_fundamentals(
            request.company_name,
            request.industry,
            request.financial_data,
            request.recent_news
        )
        
        logger.info(f"Company analysis completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Company analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"公司分析失败: {str(e)}")


@router.post("/market-sentiment", response_model=MarketSentimentResult)
async def analyze_market_sentiment(
    request: MarketSentimentRequest,
    current_user: User = Depends(get_current_user)
) -> MarketSentimentResult:
    """
    市场情感分析
    
    分析市场整体情感状况，包括：
    - 整体情感倾向
    - 情感评分
    - 热点板块识别
    - 风险提示
    - 市场展望
    
    Args:
        request: 市场情感分析请求
        current_user: 当前用户
        
    Returns:
        市场情感分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting market sentiment analysis")
        
        result = await intelligent_analysis.analyze_market_sentiment(
            request.news_texts,
            request.market_data
        )
        
        logger.info(f"Market sentiment analysis completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Market sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"市场情感分析失败: {str(e)}")


@router.post("/trend-prediction", response_model=TrendPredictionResult)
async def predict_trend(
    request: TrendPredictionRequest,
    current_user: User = Depends(get_current_user)
) -> TrendPredictionResult:
    """
    价格趋势预测
    
    基于历史价格数据预测未来趋势，包括：
    - 价格预测序列
    - 趋势方向判断
    - 置信区间
    - 关键价位计算
    
    Args:
        request: 趋势预测请求
        current_user: 当前用户
        
    Returns:
        趋势预测结果
    """
    try:
        logger.info(f"User {current_user.id} requesting trend prediction for {request.symbol}")
        
        result = await intelligent_analysis.predict_price_trend(
            request.symbol,
            request.price_data,
            request.horizon
        )
        
        logger.info(f"Trend prediction completed in {result.processing_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Trend prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"趋势预测失败: {str(e)}")


@router.post("/comprehensive")
async def comprehensive_analysis(
    symbol: str,
    company_name: str,
    industry: str,
    financial_data: str = "",
    recent_news: str = "",
    price_data: Optional[List[float]] = None,
    policy_text: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    综合分析
    
    对单个标的进行全方位分析，整合多个分析维度
    
    Args:
        symbol: 股票代码
        company_name: 公司名称
        industry: 所属行业
        financial_data: 财务数据
        recent_news: 最新新闻
        price_data: 历史价格数据（可选）
        policy_text: 相关政策文本（可选）
        current_user: 当前用户
        
    Returns:
        综合分析结果
    """
    try:
        logger.info(f"User {current_user.id} requesting comprehensive analysis for {symbol}")
        
        results = {}
        
        # 公司基本面分析
        company_result = await intelligent_analysis.analyze_company_fundamentals(
            company_name, industry, financial_data, recent_news
        )
        results["company_analysis"] = company_result
        
        # 趋势预测（如果有价格数据）
        if price_data and len(price_data) >= 5:
            trend_result = await intelligent_analysis.predict_price_trend(
                symbol, price_data, 10
            )
            results["trend_prediction"] = trend_result
        
        # 政策分析（如果有政策文本）
        if policy_text:
            policy_result = await intelligent_analysis.analyze_policy_impact(policy_text)
            results["policy_analysis"] = policy_result
        
        # 市场情感分析（基于新闻）
        if recent_news:
            sentiment_result = await intelligent_analysis.analyze_market_sentiment([recent_news])
            results["market_sentiment"] = sentiment_result
        
        # 生成综合评分和建议
        comprehensive_score = _calculate_comprehensive_score(results)
        investment_recommendation = _generate_investment_recommendation(results, comprehensive_score)
        
        results["comprehensive"] = {
            "symbol": symbol,
            "company_name": company_name,
            "industry": industry,
            "overall_score": comprehensive_score,
            "investment_recommendation": investment_recommendation,
            "analysis_timestamp": "2025-01-04T15:53:00+08:00"
        }
        
        logger.info(f"Comprehensive analysis completed for {symbol}")
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"综合分析失败: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    智能分析服务健康检查
    
    Returns:
        健康状态信息
    """
    try:
        # 检查AI服务状态
        ai_status = await intelligent_analysis.ai_service.get_model_status()
        
        # 计算健康指标
        total_models = len(ai_status)
        loaded_models = sum(1 for status in ai_status.values() if status.loaded)
        
        health_score = (loaded_models / total_models * 100) if total_models > 0 else 0
        
        return {
            "status": "healthy" if health_score >= 50 else "degraded",
            "health_score": health_score,
            "ai_models": {
                "total": total_models,
                "loaded": loaded_models,
                "available_functions": [
                    "policy_analysis",
                    "company_analysis", 
                    "market_sentiment",
                    "trend_prediction"
                ]
            },
            "timestamp": "2025-01-04T15:53:00+08:00"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-04T15:53:00+08:00"
        }


def _calculate_comprehensive_score(results: Dict[str, Any]) -> int:
    """计算综合评分"""
    scores = []
    
    # 公司基本面评分
    if "company_analysis" in results:
        scores.append(results["company_analysis"].fundamental_score)
    
    # 政策影响评分
    if "policy_analysis" in results:
        scores.append(results["policy_analysis"].policy_strength_score)
    
    # 市场情感评分（转换为0-100）
    if "market_sentiment" in results:
        sentiment_score = results["market_sentiment"].sentiment_score * 100
        scores.append(sentiment_score)
    
    # 趋势评分（基于趋势方向）
    if "trend_prediction" in results:
        trend_direction = results["trend_prediction"].trend_direction
        if trend_direction == "上涨":
            scores.append(75)
        elif trend_direction == "下跌":
            scores.append(25)
        else:
            scores.append(50)
    
    return int(sum(scores) / len(scores)) if scores else 50


def _generate_investment_recommendation(results: Dict[str, Any], score: int) -> str:
    """生成投资建议"""
    if score >= 80:
        recommendation = "强烈推荐"
    elif score >= 70:
        recommendation = "推荐"
    elif score >= 60:
        recommendation = "谨慎推荐"
    elif score >= 40:
        recommendation = "观望"
    else:
        recommendation = "不推荐"
    
    # 基于具体分析结果调整
    if "company_analysis" in results:
        rating = results["company_analysis"].investment_rating
        if rating == "买入" and recommendation in ["观望", "不推荐"]:
            recommendation = "谨慎推荐"
        elif rating == "卖出":
            recommendation = "不推荐"
    
    return recommendation
