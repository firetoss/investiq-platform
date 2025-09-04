"""
智能分析模块 - 基于AI的投资分析服务
整合政策分析、情感分析、时间序列预测等功能
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import re
import json

from pydantic import BaseModel, Field

from .ai_service import ai_service, LLMRequest, SentimentRequest, TimeseriesRequest
from .scoring_engine import scoring_engine

logger = logging.getLogger(__name__)


class PolicyAnalysisResult(BaseModel):
    """政策分析结果"""
    policy_text: str
    beneficiary_industries: List[str] = []
    policy_strength_score: int = 0  # 1-100
    implementation_certainty: str = "中"  # 高/中/低
    investment_suggestions: List[str] = []
    risk_factors: List[str] = []
    processing_time: float = 0.0
    confidence: float = 0.0


class CompanyAnalysisResult(BaseModel):
    """公司分析结果"""
    company_name: str
    industry: str
    fundamental_score: int = 0  # 1-100
    market_position: str = "中等"
    competitive_advantages: List[str] = []
    risk_factors: List[str] = []
    investment_rating: str = "持有"  # 买入/持有/卖出
    target_price: Optional[float] = None
    processing_time: float = 0.0
    confidence: float = 0.0


class MarketSentimentResult(BaseModel):
    """市场情感分析结果"""
    overall_sentiment: str = "中性"  # 积极/中性/消极
    sentiment_score: float = 0.5  # 0-1
    hot_sectors: List[str] = []
    risk_alerts: List[str] = []
    market_outlook: str = ""
    processing_time: float = 0.0


class TrendPredictionResult(BaseModel):
    """趋势预测结果"""
    symbol: str
    predictions: List[float] = []
    trend_direction: str = "横盘"  # 上涨/下跌/横盘
    confidence_intervals: Optional[Tuple[List[float], List[float]]] = None
    key_levels: Dict[str, float] = {}  # 支撑位、阻力位等
    processing_time: float = 0.0


class IntelligentAnalysis:
    """
    智能分析服务
    
    提供基于AI的投资分析功能：
    1. 政策影响分析
    2. 公司基本面分析
    3. 市场情感分析
    4. 价格趋势预测
    5. 综合投资建议
    """
    
    def __init__(self):
        self.ai_service = ai_service
        self.scoring_engine = scoring_engine
        
        # 行业关键词映射
        self.industry_keywords = {
            "半导体": ["芯片", "集成电路", "晶圆", "封装", "测试", "设备", "材料"],
            "新能源": ["光伏", "风电", "储能", "电池", "新能源汽车", "充电桩"],
            "人工智能": ["AI", "机器学习", "深度学习", "算法", "数据", "云计算"],
            "生物医药": ["医药", "生物", "疫苗", "创新药", "医疗器械", "诊断"],
            "军工": ["军工", "航空", "航天", "雷达", "导弹", "无人机"],
            "消费": ["消费", "零售", "品牌", "食品", "饮料", "服装"],
            "金融": ["银行", "保险", "证券", "基金", "支付", "金融科技"],
            "地产": ["房地产", "建筑", "装修", "物业", "园区", "城市更新"]
        }
        
        # 情感关键词
        self.sentiment_keywords = {
            "positive": ["利好", "上涨", "增长", "盈利", "突破", "创新", "领先", "优势"],
            "negative": ["利空", "下跌", "亏损", "风险", "压力", "困难", "挑战", "下滑"],
            "neutral": ["稳定", "持平", "维持", "观望", "等待", "关注"]
        }
    
    async def analyze_policy_impact(self, policy_text: str) -> PolicyAnalysisResult:
        """
        分析政策对投资的影响
        
        Args:
            policy_text: 政策文本内容
            
        Returns:
            政策分析结果
        """
        start_time = datetime.now()
        
        try:
            # 使用AI进行政策分析
            ai_result = await self.ai_service.analyze_policy(policy_text)
            analysis_text = ai_result.get("analysis", "")
            
            # 提取受益行业
            beneficiary_industries = self._extract_industries_from_text(analysis_text)
            
            # 计算政策强度评分
            policy_strength = self._calculate_policy_strength(policy_text, analysis_text)
            
            # 评估实施确定性
            implementation_certainty = self._assess_implementation_certainty(policy_text)
            
            # 提取投资建议
            investment_suggestions = self._extract_investment_suggestions(analysis_text)
            
            # 识别风险因素
            risk_factors = self._extract_risk_factors(analysis_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PolicyAnalysisResult(
                policy_text=policy_text[:500] + "..." if len(policy_text) > 500 else policy_text,
                beneficiary_industries=beneficiary_industries,
                policy_strength_score=policy_strength,
                implementation_certainty=implementation_certainty,
                investment_suggestions=investment_suggestions,
                risk_factors=risk_factors,
                processing_time=processing_time,
                confidence=0.8  # 基于AI分析的置信度
            )
            
        except Exception as e:
            logger.error(f"Policy analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return PolicyAnalysisResult(
                policy_text=policy_text[:500] + "..." if len(policy_text) > 500 else policy_text,
                processing_time=processing_time,
                confidence=0.0
            )
    
    async def analyze_company_fundamentals(
        self, 
        company_name: str, 
        industry: str,
        financial_data: str,
        recent_news: str
    ) -> CompanyAnalysisResult:
        """
        分析公司基本面
        
        Args:
            company_name: 公司名称
            industry: 所属行业
            financial_data: 财务数据
            recent_news: 最新新闻
            
        Returns:
            公司分析结果
        """
        start_time = datetime.now()
        
        try:
            # 使用AI进行公司分析
            ai_result = await self.ai_service.analyze_company(
                company_name, industry, financial_data, recent_news
            )
            analysis_text = ai_result.get("analysis", "")
            
            # 计算基本面评分
            fundamental_score = self._calculate_fundamental_score(financial_data, analysis_text)
            
            # 评估市场地位
            market_position = self._assess_market_position(analysis_text)
            
            # 提取竞争优势
            competitive_advantages = self._extract_competitive_advantages(analysis_text)
            
            # 识别风险因素
            risk_factors = self._extract_risk_factors(analysis_text)
            
            # 确定投资评级
            investment_rating = self._determine_investment_rating(analysis_text, fundamental_score)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CompanyAnalysisResult(
                company_name=company_name,
                industry=industry,
                fundamental_score=fundamental_score,
                market_position=market_position,
                competitive_advantages=competitive_advantages,
                risk_factors=risk_factors,
                investment_rating=investment_rating,
                processing_time=processing_time,
                confidence=0.75
            )
            
        except Exception as e:
            logger.error(f"Company analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CompanyAnalysisResult(
                company_name=company_name,
                industry=industry,
                processing_time=processing_time,
                confidence=0.0
            )
    
    async def analyze_market_sentiment(
        self, 
        news_texts: List[str],
        market_data: Optional[str] = None
    ) -> MarketSentimentResult:
        """
        分析市场情感
        
        Args:
            news_texts: 新闻文本列表
            market_data: 市场数据（可选）
            
        Returns:
            市场情感分析结果
        """
        start_time = datetime.now()
        
        try:
            # 批量情感分析
            sentiment_request = SentimentRequest(texts=news_texts)
            sentiment_response = await self.ai_service.sentiment_analysis(sentiment_request)
            
            # 计算整体情感
            sentiment_scores = [result.score for result in sentiment_response.results]
            positive_count = sum(1 for result in sentiment_response.results if result.label == "POSITIVE")
            negative_count = sum(1 for result in sentiment_response.results if result.label == "NEGATIVE")
            
            overall_sentiment_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
            if overall_sentiment_score > 0.6:
                overall_sentiment = "积极"
            elif overall_sentiment_score < 0.4:
                overall_sentiment = "消极"
            else:
                overall_sentiment = "中性"
            
            # 识别热点板块
            hot_sectors = self._identify_hot_sectors(news_texts)
            
            # 生成风险提示
            risk_alerts = self._generate_risk_alerts(sentiment_response.results, news_texts)
            
            # 生成市场展望
            market_outlook = await self._generate_market_outlook(
                overall_sentiment, hot_sectors, market_data
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MarketSentimentResult(
                overall_sentiment=overall_sentiment,
                sentiment_score=overall_sentiment_score,
                hot_sectors=hot_sectors,
                risk_alerts=risk_alerts,
                market_outlook=market_outlook,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MarketSentimentResult(
                processing_time=processing_time
            )
    
    async def predict_price_trend(
        self, 
        symbol: str, 
        price_data: List[float],
        horizon: int = 10
    ) -> TrendPredictionResult:
        """
        预测价格趋势
        
        Args:
            symbol: 股票代码
            price_data: 历史价格数据
            horizon: 预测期数
            
        Returns:
            趋势预测结果
        """
        start_time = datetime.now()
        
        try:
            # 使用AI进行时间序列预测
            ts_request = TimeseriesRequest(
                data=price_data,
                horizon=horizon,
                confidence_intervals=True
            )
            ts_response = await self.ai_service.timeseries_forecast(ts_request)
            
            # 判断趋势方向
            if len(price_data) > 0 and len(ts_response.predictions) > 0:
                current_price = price_data[-1]
                future_price = ts_response.predictions[-1]
                
                if future_price > current_price * 1.05:
                    trend_direction = "上涨"
                elif future_price < current_price * 0.95:
                    trend_direction = "下跌"
                else:
                    trend_direction = "横盘"
            else:
                trend_direction = "横盘"
            
            # 计算关键价位
            key_levels = self._calculate_key_levels(price_data, ts_response.predictions)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TrendPredictionResult(
                symbol=symbol,
                predictions=ts_response.predictions,
                trend_direction=trend_direction,
                confidence_intervals=(
                    ts_response.confidence_lower,
                    ts_response.confidence_upper
                ) if ts_response.confidence_lower else None,
                key_levels=key_levels,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Price trend prediction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TrendPredictionResult(
                symbol=symbol,
                processing_time=processing_time
            )
    
    def _extract_industries_from_text(self, text: str) -> List[str]:
        """从文本中提取受益行业"""
        industries = []
        for industry, keywords in self.industry_keywords.items():
            if any(keyword in text for keyword in keywords):
                industries.append(industry)
        return industries[:5]  # 最多返回5个行业
    
    def _calculate_policy_strength(self, policy_text: str, analysis_text: str) -> int:
        """计算政策强度评分"""
        score = 50  # 基础分
        
        # 基于政策文本长度
        if len(policy_text) > 1000:
            score += 10
        
        # 基于关键词
        strong_keywords = ["重点", "重大", "优先", "加快", "大力", "全面", "深入"]
        score += sum(5 for keyword in strong_keywords if keyword in policy_text)
        
        # 基于分析结果
        if "重要" in analysis_text or "重大" in analysis_text:
            score += 15
        
        return min(100, max(0, score))
    
    def _assess_implementation_certainty(self, policy_text: str) -> str:
        """评估政策实施确定性"""
        high_certainty_keywords = ["已", "将", "必须", "确保", "落实"]
        medium_certainty_keywords = ["计划", "拟", "预计", "力争"]
        low_certainty_keywords = ["研究", "探索", "考虑", "可能"]
        
        high_count = sum(1 for keyword in high_certainty_keywords if keyword in policy_text)
        medium_count = sum(1 for keyword in medium_certainty_keywords if keyword in policy_text)
        low_count = sum(1 for keyword in low_certainty_keywords if keyword in policy_text)
        
        if high_count > medium_count and high_count > low_count:
            return "高"
        elif low_count > high_count and low_count > medium_count:
            return "低"
        else:
            return "中"
    
    def _extract_investment_suggestions(self, text: str) -> List[str]:
        """提取投资建议"""
        suggestions = []
        
        # 简单的规则提取
        if "建议" in text:
            sentences = text.split("。")
            for sentence in sentences:
                if "建议" in sentence and len(sentence) < 100:
                    suggestions.append(sentence.strip())
        
        return suggestions[:3]  # 最多返回3条建议
    
    def _extract_risk_factors(self, text: str) -> List[str]:
        """提取风险因素"""
        risk_factors = []
        risk_keywords = ["风险", "不确定", "挑战", "压力", "困难"]
        
        sentences = text.split("。")
        for sentence in sentences:
            if any(keyword in sentence for keyword in risk_keywords) and len(sentence) < 100:
                risk_factors.append(sentence.strip())
        
        return risk_factors[:3]  # 最多返回3个风险因素
    
    def _calculate_fundamental_score(self, financial_data: str, analysis_text: str) -> int:
        """计算基本面评分"""
        score = 50  # 基础分
        
        # 基于财务数据关键词
        positive_keywords = ["增长", "盈利", "改善", "提升", "强劲"]
        negative_keywords = ["下降", "亏损", "恶化", "压力", "困难"]
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in financial_data)
        negative_count = sum(1 for keyword in negative_keywords if keyword in financial_data)
        
        score += positive_count * 8 - negative_count * 8
        
        # 基于分析文本
        if "优秀" in analysis_text or "良好" in analysis_text:
            score += 15
        elif "一般" in analysis_text:
            score += 5
        elif "较差" in analysis_text:
            score -= 15
        
        return min(100, max(0, score))
    
    def _assess_market_position(self, analysis_text: str) -> str:
        """评估市场地位"""
        if any(keyword in analysis_text for keyword in ["领先", "龙头", "第一"]):
            return "领先"
        elif any(keyword in analysis_text for keyword in ["前列", "优势", "竞争力"]):
            return "优势"
        elif any(keyword in analysis_text for keyword in ["落后", "劣势", "困难"]):
            return "劣势"
        else:
            return "中等"
    
    def _extract_competitive_advantages(self, text: str) -> List[str]:
        """提取竞争优势"""
        advantages = []
        advantage_keywords = ["优势", "领先", "专利", "技术", "品牌", "渠道"]
        
        sentences = text.split("。")
        for sentence in sentences:
            if any(keyword in sentence for keyword in advantage_keywords) and len(sentence) < 100:
                advantages.append(sentence.strip())
        
        return advantages[:3]
    
    def _determine_investment_rating(self, analysis_text: str, fundamental_score: int) -> str:
        """确定投资评级"""
        if fundamental_score >= 80 and ("买入" in analysis_text or "推荐" in analysis_text):
            return "买入"
        elif fundamental_score <= 40 or "卖出" in analysis_text:
            return "卖出"
        else:
            return "持有"
    
    def _identify_hot_sectors(self, news_texts: List[str]) -> List[str]:
        """识别热点板块"""
        sector_mentions = {}
        
        for text in news_texts:
            for sector, keywords in self.industry_keywords.items():
                mentions = sum(1 for keyword in keywords if keyword in text)
                sector_mentions[sector] = sector_mentions.get(sector, 0) + mentions
        
        # 按提及次数排序
        sorted_sectors = sorted(sector_mentions.items(), key=lambda x: x[1], reverse=True)
        return [sector for sector, count in sorted_sectors[:5] if count > 0]
    
    def _generate_risk_alerts(self, sentiment_results, news_texts: List[str]) -> List[str]:
        """生成风险提示"""
        alerts = []
        
        # 基于负面情感比例
        negative_count = sum(1 for result in sentiment_results if result.label == "NEGATIVE")
        if negative_count / len(sentiment_results) > 0.6:
            alerts.append("市场负面情绪较重，需谨慎操作")
        
        # 基于关键词
        risk_keywords = ["暴跌", "崩盘", "危机", "恐慌", "抛售"]
        for text in news_texts:
            if any(keyword in text for keyword in risk_keywords):
                alerts.append("市场出现恐慌情绪，注意风险控制")
                break
        
        return alerts
    
    async def _generate_market_outlook(
        self, 
        sentiment: str, 
        hot_sectors: List[str], 
        market_data: Optional[str]
    ) -> str:
        """生成市场展望"""
        try:
            prompt = f"""
基于以下信息生成市场展望：
- 整体情感：{sentiment}
- 热点板块：{', '.join(hot_sectors)}
- 市场数据：{market_data or '无'}

请提供简洁的市场展望（100字以内）。
"""
            
            request = LLMRequest(prompt=prompt, max_tokens=200)
            response = await self.ai_service.llm_inference(request)
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Generate market outlook failed: {e}")
            return f"基于当前{sentiment}的市场情感，建议关注{', '.join(hot_sectors[:2])}等热点板块。"
    
    def _calculate_key_levels(self, historical_data: List[float], predictions: List[float]) -> Dict[str, float]:
        """计算关键价位"""
        if not historical_data:
            return {}
        
        all_data = historical_data + predictions
        
        return {
            "support": min(all_data[-20:]) if len(all_data) >= 20 else min(all_data),
            "resistance": max(all_data[-20:]) if len(all_data) >= 20 else max(all_data),
            "current": historical_data[-1] if historical_data else 0,
            "target": predictions[-1] if predictions else 0
        }


# 全局智能分析服务实例
intelligent_analysis = IntelligentAnalysis()
