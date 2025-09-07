"""
API路由 - 主要业务端点的精简版本
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List

from investiq_common.clients import registry
from investiq_common.models import (
    LLMRequest, SentimentRequest, TimeSeriesRequest,
    APIResponse, EquityScore
)
from investiq_common.utils import get_logger, performance_timer

from ..core.redis import redis_client


logger = get_logger(__name__)
router = APIRouter()


@router.get("/analysis/{symbol}")
@performance_timer(logger)
async def get_stock_analysis(symbol: str) -> APIResponse:
    """
    综合股票分析 - 协调所有AI服务
    这是微服务架构的核心：业务协调而非直接计算
    """
    try:
        # 检查缓存
        cache_key = f"analysis:{symbol}"
        cached_result = await redis_client.get(cache_key)
        
        if cached_result:
            import json
            return APIResponse(
                success=True,
                message="Analysis retrieved from cache",
                data=json.loads(cached_result)
            )
        
        # 并行调用多个AI服务
        tasks = []
        
        # 1. 情感分析 - 获取相关新闻情感
        sentiment_request = SentimentRequest(
            texts=[f"Recent news about {symbol}"]  # 实际应用中从新闻API获取
        )
        sentiment_task = registry.clients["sentiment"].analyze(sentiment_request)
        tasks.append(("sentiment", sentiment_task))
        
        # 2. 时序预测 - 价格趋势预测
        timeseries_request = TimeSeriesRequest(
            symbol=symbol,
            data=[100, 102, 101, 105, 103],  # 实际应用中从数据源获取
            prediction_steps=5
        )
        timeseries_task = registry.clients["timeseries"].predict(timeseries_request)
        tasks.append(("timeseries", timeseries_task))
        
        # 3. LLM分析 - 综合分析报告
        llm_request = LLMRequest(
            prompt=f"Provide investment analysis for stock {symbol}",
            max_tokens=1024
        )
        llm_task = registry.clients["llm"].inference(llm_request)
        tasks.append(("llm", llm_task))
        
        # 等待所有任务完成
        results = {}
        import asyncio
        
        for task_name, task in tasks:
            try:
                result = await task
                results[task_name] = result
            except Exception as e:
                logger.error(f"Service {task_name} failed: {e}")
                results[task_name] = {"error": str(e)}
        
        # 聚合结果
        analysis_result = {
            "symbol": symbol,
            "sentiment_analysis": results.get("sentiment"),
            "price_prediction": results.get("timeseries"), 
            "llm_analysis": results.get("llm"),
            "overall_score": _calculate_overall_score(results),
        }
        
        # 缓存结果
        import json
        await redis_client.set(cache_key, json.dumps(analysis_result), expire=300)
        
        return APIResponse(
            success=True,
            message=f"Comprehensive analysis for {symbol}",
            data=analysis_result
        )
        
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Analysis service error")


@router.post("/portfolio/optimize")
@performance_timer(logger)
async def optimize_portfolio(
    symbols: List[str],
    risk_level: str = "moderate"
) -> APIResponse:
    """投资组合优化 - 演示服务协调"""
    try:
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Too many symbols")
        
        # 为每个股票获取分析
        portfolio_analysis = []
        
        for symbol in symbols:
            # 调用分析服务
            analysis = await get_stock_analysis(symbol)
            if analysis.success:
                portfolio_analysis.append(analysis.data)
        
        # 简化的投资组合优化逻辑
        optimized_weights = _optimize_portfolio_weights(portfolio_analysis, risk_level)
        
        result = {
            "symbols": symbols,
            "risk_level": risk_level,
            "optimized_weights": optimized_weights,
            "expected_return": _calculate_expected_return(portfolio_analysis, optimized_weights),
            "risk_score": _calculate_risk_score(portfolio_analysis, optimized_weights)
        }
        
        return APIResponse(
            success=True,
            message="Portfolio optimization completed",
            data=result
        )
        
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Optimization service error")


@router.get("/services/status")
async def get_services_status() -> APIResponse:
    """获取所有微服务状态"""
    try:
        health_status = await registry.health_check_all()
        
        return APIResponse(
            success=True,
            message="Services status retrieved",
            data={
                "services": health_status,
                "total_services": len(health_status),
                "healthy_services": sum(1 for h in health_status.values() if h.status == "healthy")
            }
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Status check error")


# =============== 辅助函数 ===============

def _calculate_overall_score(results: Dict[str, Any]) -> float:
    """计算综合评分 - 简化版本"""
    score = 50.0  # 基础分数
    
    # 情感分析贡献
    if "sentiment" in results and "error" not in results["sentiment"]:
        sentiment_data = results["sentiment"]
        if hasattr(sentiment_data, 'results') and sentiment_data.results:
            avg_sentiment = sum(r.score for r in sentiment_data.results) / len(sentiment_data.results)
            score += avg_sentiment * 20
    
    # 时序预测贡献
    if "timeseries" in results and "error" not in results["timeseries"]:
        # 根据预测趋势调整分数
        score += 10  # 简化处理
    
    return min(max(score, 0), 100)


def _optimize_portfolio_weights(analysis_data: List[Dict], risk_level: str) -> Dict[str, float]:
    """简化的投资组合权重优化"""
    n_stocks = len(analysis_data)
    if n_stocks == 0:
        return {}
    
    # 基于风险等级的简单权重分配
    if risk_level == "conservative":
        # 均匀分配，保守策略
        weight = 1.0 / n_stocks
        return {data["symbol"]: weight for data in analysis_data}
    
    elif risk_level == "aggressive":
        # 基于分数的权重分配
        scores = [data.get("overall_score", 50) for data in analysis_data]
        total_score = sum(scores)
        
        if total_score > 0:
            return {
                analysis_data[i]["symbol"]: scores[i] / total_score 
                for i in range(n_stocks)
            }
    
    # 默认：适中策略
    weight = 1.0 / n_stocks
    return {data["symbol"]: weight for data in analysis_data}


def _calculate_expected_return(analysis_data: List[Dict], weights: Dict[str, float]) -> float:
    """计算期望收益 - 简化版本"""
    total_return = 0.0
    
    for data in analysis_data:
        symbol = data["symbol"]
        weight = weights.get(symbol, 0)
        score = data.get("overall_score", 50)
        
        # 简单映射：分数越高，期望收益越高
        expected_return = (score - 50) / 100  # -0.5 到 0.5
        total_return += weight * expected_return
    
    return total_return


def _calculate_risk_score(analysis_data: List[Dict], weights: Dict[str, float]) -> float:
    """计算风险分数 - 简化版本"""
    # 分散投资降低风险
    n_stocks = len([w for w in weights.values() if w > 0])
    diversification_factor = min(n_stocks / 10, 1.0)
    
    base_risk = 0.5  # 基础风险
    return base_risk * (1 - diversification_factor * 0.3)