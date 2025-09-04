"""
InvestIQ Platform - 置信度计算服务
实现PRD规定的置信度公式和停牌处理
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from backend.app.core.config import settings
from backend.app.core.exceptions import ConfidenceException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.equity import Equity
from backend.app.models.snapshots import TechnicalIndicatorSnapshot


logger = get_logger(__name__)


class ConfidenceCalculator:
    """置信度计算器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    @log_performance("confidence_calculation")
    async def calculate_confidence(
        self,
        entity_type: str,
        entity_id: str,
        scores: Dict[str, float],
        evidence_data: List[Dict],
        technical_data: Optional[Dict] = None,
        as_of: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        计算置信度 (基于PRD公式)
        
        PRD规定公式:
        - w_cov = min(1, N_obs/1250) (5年×250个交易日)  
        - w_stale = 1(≤5日) / 0.8(6–20日) / 0.6(>20日)
        - w_src ∈ {1.0, 0.9, 0.8} (官方/授权/公开抓取)
        - Conf = 100 × w_cov × w_stale × w_src
        
        Args:
            entity_type: 实体类型 (industry, equity)
            entity_id: 实体ID
            scores: 评分数据
            evidence_data: 证据数据
            technical_data: 技术数据 (用于停牌检查)
            as_of: 计算时点
        
        Returns:
            置信度计算结果
        """
        try:
            as_of = as_of or date.today()
            
            # 计算覆盖权重 w_cov
            w_cov = await self._calculate_coverage_weight(entity_type, entity_id, as_of)
            
            # 计算新鲜度权重 w_stale
            w_stale = await self._calculate_staleness_weight(entity_type, entity_id, as_of, technical_data)
            
            # 计算数据源权重 w_src
            w_src = self._calculate_source_weight(evidence_data)
            
            # 计算评分完整性权重 w_completeness
            w_completeness = self._calculate_completeness_weight(scores)
            
            # 综合置信度
            confidence = 100.0 * w_cov * w_stale * w_src * w_completeness
            confidence = min(100.0, max(0.0, confidence))
            
            # 构建详细结果
            result = {
                "confidence": round(confidence, 2),
                "components": {
                    "coverage_weight": round(w_cov, 3),
                    "staleness_weight": round(w_stale, 3),
                    "source_weight": round(w_src, 3),
                    "completeness_weight": round(w_completeness, 3)
                },
                "metadata": {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "as_of": as_of.isoformat(),
                    "calculated_at": datetime.utcnow().isoformat(),
                    "evidence_count": len(evidence_data),
                    "scores_provided": len([v for v in scores.values() if v is not None and v > 0])
                },
                "flags": []
            }
            
            # 添加警告标记
            if confidence < 60:
                result["flags"].append("low_confidence")
            if w_stale < 0.8:
                result["flags"].append("stale_data")
            if w_cov < 0.8:
                result["flags"].append("limited_coverage")
            if w_src < 0.9:
                result["flags"].append("weak_sources")
            
            return result
            
        except Exception as e:
            logger.error(f"Confidence calculation failed for {entity_type}:{entity_id}: {e}")
            raise ConfidenceException(f"置信度计算失败: {e}")
    
    async def _calculate_coverage_weight(self, entity_type: str, entity_id: str, as_of: date) -> float:
        """
        计算覆盖权重 w_cov
        
        基于观测天数: w_cov = min(1, N_obs/1250)
        1250 = 5年 × 250个交易日
        """
        try:
            # 获取历史数据天数
            if entity_type == "equity":
                # 查询技术指标快照的天数
                stmt = select(TechnicalIndicatorSnapshot).where(
                    and_(
                        TechnicalIndicatorSnapshot.ticker == entity_id,
                        TechnicalIndicatorSnapshot.as_of <= as_of,
                        TechnicalIndicatorSnapshot.as_of >= as_of - timedelta(days=1825)  # 5年
                    )
                )
                result = await self.db_session.execute(stmt)
                observations = result.scalars().all()
                n_obs = len(observations)
            else:
                # 对于行业，使用固定的观测天数估算
                # 假设行业数据可获得度较高
                n_obs = min(1000, (datetime.now().date() - as_of).days + 1000)
            
            # 计算覆盖权重
            w_cov = min(1.0, n_obs / 1250.0)
            
            logger.debug(f"Coverage weight calculated: {w_cov} (n_obs={n_obs})")
            return w_cov
            
        except Exception as e:
            logger.warning(f"Failed to calculate coverage weight, using default: {e}")
            return 0.8  # 默认覆盖权重
    
    async def _calculate_staleness_weight(
        self, 
        entity_type: str, 
        entity_id: str, 
        as_of: date,
        technical_data: Optional[Dict] = None
    ) -> float:
        """
        计算新鲜度权重 w_stale
        
        PRD规定:
        - w_stale = 1 (≤5日)
        - w_stale = 0.8 (6–20日)  
        - w_stale = 0.6 (>20日)
        
        对于停牌股票特殊处理:
        - 停牌期间使用"前值携带参与均线样本"，标注 isStale=true
        - 停牌 > 20个交易日，置信度权重下调
        """
        try:
            if entity_type == "equity":
                # 检查停牌状态
                suspension_info = await self._check_suspension_status(entity_id, as_of)
                
                if suspension_info["is_suspended"]:
                    suspension_days = suspension_info["suspension_days"]
                    
                    # 停牌天数对应权重调整
                    if suspension_days <= 5:
                        w_stale = 0.9  # 短期停牌轻微折扣
                    elif suspension_days <= 20:
                        w_stale = 0.6  # 中期停牌较大折扣
                    else:
                        w_stale = 0.4  # 长期停牌大幅折扣
                    
                    logger.info(f"Equity {entity_id} suspended for {suspension_days} days, staleness_weight={w_stale}")
                    return w_stale
                
                # 非停牌情况：检查最新数据的新鲜度
                if technical_data and "last_update_days" in technical_data:
                    days_since_update = technical_data["last_update_days"]
                else:
                    # 查询最新的技术指标快照
                    stmt = select(TechnicalIndicatorSnapshot).where(
                        TechnicalIndicatorSnapshot.ticker == entity_id
                    ).order_by(TechnicalIndicatorSnapshot.as_of.desc()).limit(1)
                    
                    result = await self.db_session.execute(stmt)
                    latest_snapshot = result.scalar_one_or_none()
                    
                    if latest_snapshot:
                        days_since_update = (as_of - latest_snapshot.as_of).days
                    else:
                        days_since_update = 999  # 无数据视为严重过时
            else:
                # 行业数据通常更新较慢，标准放宽
                days_since_update = 3  # 假设行业数据3天内
            
            # 根据PRD规定计算权重
            if days_since_update <= 5:
                w_stale = 1.0
            elif days_since_update <= 20:
                w_stale = 0.8
            else:
                w_stale = 0.6
            
            return w_stale
            
        except Exception as e:
            logger.warning(f"Failed to calculate staleness weight, using default: {e}")
            return 0.8
    
    async def _check_suspension_status(self, ticker: str, as_of: date) -> Dict[str, Any]:
        """检查股票停牌状态"""
        try:
            # 查询最新的技术指标快照
            stmt = select(TechnicalIndicatorSnapshot).where(
                and_(
                    TechnicalIndicatorSnapshot.ticker == ticker,
                    TechnicalIndicatorSnapshot.as_of <= as_of
                )
            ).order_by(TechnicalIndicatorSnapshot.as_of.desc()).limit(1)
            
            result = await self.db_session.execute(stmt)
            latest_snapshot = result.scalar_one_or_none()
            
            if not latest_snapshot:
                return {
                    "is_suspended": False,
                    "suspension_days": 0,
                    "last_trading_day": None
                }
            
            is_suspended = latest_snapshot.is_suspended
            suspension_days = latest_snapshot.suspension_days
            last_trading_day = latest_snapshot.last_trading_day
            
            return {
                "is_suspended": is_suspended,
                "suspension_days": suspension_days,
                "last_trading_day": last_trading_day,
                "data_as_of": latest_snapshot.as_of
            }
            
        except Exception as e:
            logger.warning(f"Failed to check suspension status for {ticker}: {e}")
            return {
                "is_suspended": False,
                "suspension_days": 0,
                "last_trading_day": None
            }
    
    def _calculate_source_weight(self, evidence_data: List[Dict]) -> float:
        """
        计算数据源权重 w_src
        
        PRD规定:
        - w_src = 1.0 (官方)
        - w_src = 0.9 (授权) 
        - w_src = 0.8 (公开抓取)
        """
        if not evidence_data:
            return 0.6  # 无证据时的默认权重
        
        source_weights = []
        
        for evidence in evidence_data:
            source_type = evidence.get("source_type", "public")
            
            if source_type in ["official", "government", "exchange"]:
                weight = 1.0
            elif source_type in ["authorized", "licensed", "api"]:
                weight = 0.9
            elif source_type in ["public", "scraping", "web"]:
                weight = 0.8
            else:
                weight = 0.7  # 未知来源
            
            # 考虑证据的重要性权重
            importance = evidence.get("importance", 1.0)
            source_weights.append(weight * importance)
        
        # 加权平均
        if source_weights:
            total_importance = sum(evidence.get("importance", 1.0) for evidence in evidence_data)
            w_src = sum(source_weights) / total_importance
        else:
            w_src = 0.8
        
        return min(1.0, w_src)
    
    def _calculate_completeness_weight(self, scores: Dict[str, float]) -> float:
        """
        计算评分完整性权重
        
        基于有效评分的占比
        """
        if not scores:
            return 0.0
        
        total_scores = len(scores)
        valid_scores = len([v for v in scores.values() if v is not None and v >= 0])
        
        completeness = valid_scores / total_scores if total_scores > 0 else 0.0
        
        # 应用权重曲线 (完整性越高权重越大)
        if completeness >= 0.9:
            return 1.0
        elif completeness >= 0.7:
            return 0.9
        elif completeness >= 0.5:
            return 0.8
        else:
            return 0.6
    
    @log_performance("batch_confidence_calculation")
    async def batch_calculate_confidence(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量计算置信度
        
        Args:
            entities: 实体列表，每个包含 entity_type, entity_id, scores, evidence_data
        
        Returns:
            置信度结果列表
        """
        results = []
        
        for entity in entities:
            try:
                confidence_result = await self.calculate_confidence(
                    entity_type=entity["entity_type"],
                    entity_id=entity["entity_id"],
                    scores=entity.get("scores", {}),
                    evidence_data=entity.get("evidence_data", []),
                    technical_data=entity.get("technical_data"),
                    as_of=entity.get("as_of")
                )
                
                confidence_result["entity_type"] = entity["entity_type"]
                confidence_result["entity_id"] = entity["entity_id"]
                results.append(confidence_result)
                
            except Exception as e:
                logger.error(f"Batch confidence calculation failed for {entity}: {e}")
                # 添加错误结果
                results.append({
                    "entity_type": entity["entity_type"],
                    "entity_id": entity["entity_id"],
                    "confidence": 0.0,
                    "error": str(e),
                    "components": {
                        "coverage_weight": 0.0,
                        "staleness_weight": 0.0,
                        "source_weight": 0.0,
                        "completeness_weight": 0.0
                    }
                })
        
        return results


class DMACalculator:
    """
    DMA (动量移动平均) 计算器
    
    实现PRD要求的200DMA计算和停牌处理
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
    
    @log_performance("dma_calculation")
    async def calculate_200dma_status(
        self,
        ticker: str,
        as_of: date,
        include_suspended: bool = True
    ) -> Dict[str, Any]:
        """
        计算200日移动平均线状态
        
        PRD规定: 停牌样本参与声明 - "停牌期间采用前值携带参与均线样本，标注 isStale=true"
        
        Args:
            ticker: 股票代码
            as_of: 计算时点  
            include_suspended: 是否包含停牌期间的数据
        
        Returns:
            200DMA状态和相关指标
        """
        try:
            # 获取200个交易日的价格数据
            end_date = as_of
            start_date = as_of - timedelta(days=400)  # 预留足够天数
            
            stmt = select(TechnicalIndicatorSnapshot).where(
                and_(
                    TechnicalIndicatorSnapshot.ticker == ticker,
                    TechnicalIndicatorSnapshot.as_of >= start_date,
                    TechnicalIndicatorSnapshot.as_of <= end_date
                )
            ).order_by(TechnicalIndicatorSnapshot.as_of.desc())
            
            result = await self.db_session.execute(stmt)
            snapshots = result.scalars().all()
            
            if len(snapshots) < 100:  # 至少需要100天数据
                return {
                    "ticker": ticker,
                    "as_of": as_of.isoformat(),
                    "above_200dma": None,
                    "current_price": None,
                    "dma_200": None,
                    "error": "Insufficient data",
                    "data_points": len(snapshots),
                    "is_stale": True
                }
            
            # 处理停牌数据
            processed_prices = []
            last_valid_price = None
            suspended_days_count = 0
            
            for snapshot in reversed(snapshots):  # 按时间正序处理
                if snapshot.is_suspended and include_suspended:
                    # 停牌期间使用前值携带
                    if last_valid_price is not None:
                        processed_prices.append({
                            "date": snapshot.as_of,
                            "price": last_valid_price,
                            "is_carried_forward": True,
                            "is_suspended": True
                        })
                        suspended_days_count += 1
                elif not snapshot.is_suspended:
                    # 正常交易日
                    processed_prices.append({
                        "date": snapshot.as_of,
                        "price": snapshot.current_price,
                        "is_carried_forward": False,
                        "is_suspended": False
                    })
                    last_valid_price = snapshot.current_price
            
            if len(processed_prices) < 200:
                # 数据不足200天，使用现有数据
                dma_period = len(processed_prices)
                logger.warning(f"Using {dma_period}-day MA instead of 200-day for {ticker}")
            else:
                dma_period = 200
                processed_prices = processed_prices[-200:]  # 取最近200天
            
            # 计算移动平均
            prices = [p["price"] for p in processed_prices]
            dma_200 = sum(prices) / len(prices)
            
            # 当前价格
            current_price = processed_prices[-1]["price"]
            
            # 判断是否在均线上方
            above_200dma = current_price > dma_200
            
            # 计算额外指标
            price_vs_dma_pct = ((current_price - dma_200) / dma_200) * 100
            
            # 计算连续天数
            consecutive_days_above = 0
            for p in reversed(processed_prices):
                if p["price"] > dma_200:
                    consecutive_days_above += 1
                else:
                    break
            
            # 数据质量评估
            carried_forward_count = sum(1 for p in processed_prices if p["is_carried_forward"])
            data_quality_score = (len(processed_prices) - carried_forward_count) / len(processed_prices) * 100
            
            is_stale = (
                suspended_days_count > 20 or  # 停牌超过20天
                carried_forward_count > dma_period * 0.3 or  # 超过30%为前值携带
                (as_of - snapshots[0].as_of).days > 5  # 最新数据超过5天
            )
            
            result = {
                "ticker": ticker,
                "as_of": as_of.isoformat(),
                "above_200dma": above_200dma,
                "current_price": round(current_price, 2),
                "dma_200": round(dma_200, 2),
                "price_vs_dma_pct": round(price_vs_dma_pct, 2),
                "consecutive_days_above": consecutive_days_above,
                "dma_period_used": dma_period,
                "data_quality": {
                    "total_data_points": len(processed_prices),
                    "carried_forward_count": carried_forward_count,
                    "suspended_days": suspended_days_count,
                    "data_quality_score": round(data_quality_score, 1),
                    "is_stale": is_stale
                },
                "metadata": {
                    "calculation_method": "carried_forward_during_suspension",
                    "include_suspended": include_suspended,
                    "calculated_at": datetime.utcnow().isoformat()
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"200DMA calculation failed for {ticker}: {e}")
            raise ConfidenceException(f"200DMA计算失败: {e}")


# 全局服务实例工厂
async def create_confidence_services(db_session: AsyncSession) -> Tuple[ConfidenceCalculator, DMACalculator]:
    """创建置信度计算服务实例"""
    confidence_calc = ConfidenceCalculator(db_session)
    dma_calc = DMACalculator(db_session)
    return confidence_calc, dma_calc