"""
InvestIQ Platform - 评分引擎服务
实现行业和个股的评分算法
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.exceptions import ScoringException, GPUException
from backend.app.core.logging import get_logger, log_performance
from backend.app.utils.gpu import check_gpu_availability, gpu_fallback, GPUMemoryMonitor
from backend.app.models.industry import IndustryScoreSnapshot
from backend.app.models.equity import EquityScoreSnapshot


logger = get_logger(__name__)


class ScoringEngine:
    """评分引擎核心类"""
    
    def __init__(self):
        self.gpu_available = check_gpu_availability()
        self.industry_weights = settings.industry_score_weights
        self.equity_weights = settings.equity_score_weights
        
        if self.gpu_available:
            logger.info("GPU acceleration enabled for scoring engine")
        else:
            logger.warning("GPU not available, using CPU fallback")
    
    @log_performance("industry_scoring")
    async def calculate_industry_score(
        self,
        industry_id: str,
        score_p: float,
        score_e: float,
        score_m: float,
        score_r_neg: float,
        weights: Optional[Dict[str, float]] = None,
        evidence_data: Optional[List[Dict]] = None,
        as_of: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        计算行业评分
        
        基于PRD公式: Score^{Ind} = 0.35P + 0.25E + 0.25M + 0.15×(100-R^{-})
        """
        try:
            # 验证输入参数
            self._validate_score_inputs([score_p, score_e, score_m, score_r_neg])
            
            # 使用配置的权重或传入的权重
            weights = weights or self.industry_weights
            
            # 计算总评分
            total_score = self._calculate_industry_score_formula(
                score_p, score_e, score_m, score_r_neg, weights
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(
                evidence_data or [],
                [score_p, score_e, score_m, score_r_neg]
            )
            
            # 评分分解
            breakdown = {
                "P_weighted": score_p * weights["P"],
                "E_weighted": score_e * weights["E"],
                "M_weighted": score_m * weights["M"],
                "R_weighted": (100 - score_r_neg) * weights["R"],
                "total": total_score
            }
            
            # 生成评分详情
            score_details = {
                "raw_scores": {
                    "P": score_p,
                    "E": score_e,
                    "M": score_m,
                    "R_neg": score_r_neg
                },
                "weights_used": weights,
                "breakdown": breakdown,
                "calculation_method": "standard_formula",
                "calculated_at": datetime.utcnow().isoformat()
            }
            
            # 检查阈值
            meets_threshold = total_score >= settings.GATE_INDUSTRY_THRESHOLD
            is_core_candidate = total_score >= 75.0  # 核心候选阈值
            
            result = {
                "industry_id": industry_id,
                "total_score": total_score,
                "confidence": confidence,
                "breakdown": breakdown,
                "score_details": score_details,
                "meets_threshold": meets_threshold,
                "is_core_candidate": is_core_candidate,
                "evidence_count": len(evidence_data) if evidence_data else 0,
                "as_of": as_of or date.today(),
                "calculated_at": datetime.utcnow()
            }
            
            logger.log_scoring_event(
                event_type="industry_score_calculated",
                entity_type="industry",
                entity_id=industry_id,
                score=total_score,
                confidence=confidence,
                meets_threshold=meets_threshold
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Industry scoring failed for {industry_id}: {e}")
            raise ScoringException(f"行业评分计算失败: {e}")
    
    @log_performance("equity_scoring")
    async def calculate_equity_score(
        self,
        equity_id: str,
        ticker: str,
        score_q: float,
        score_v: float,
        score_m: float,
        score_c: float,
        score_s: float,
        score_r_neg: float = 0,
        weights: Optional[Dict[str, float]] = None,
        evidence_data: Optional[List[Dict]] = None,
        as_of: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        计算个股评分
        
        基于PRD权重: Q(30%) + V(20%) + M(25%) + C(15%) + S(10%) - R^{-}
        """
        try:
            # 验证输入参数
            self._validate_score_inputs([score_q, score_v, score_m, score_c, score_s])
            
            # 使用配置的权重或传入的权重
            weights = weights or self.equity_weights
            
            # 计算总评分
            total_score = self._calculate_equity_score_formula(
                score_q, score_v, score_m, score_c, score_s, score_r_neg, weights
            )
            
            # 计算置信度
            confidence = self._calculate_confidence(
                evidence_data or [],
                [score_q, score_v, score_m, score_c, score_s]
            )
            
            # 评分分解
            breakdown = {
                "Q_weighted": score_q * weights["Q"],
                "V_weighted": score_v * weights["V"],
                "M_weighted": score_m * weights["M"],
                "C_weighted": score_c * weights["C"],
                "S_weighted": score_s * weights["S"],
                "R_neg_penalty": score_r_neg,
                "total": total_score
            }
            
            # 检查红旗
            has_red_flags = score_r_neg > 0
            red_flags = self._analyze_red_flags(score_r_neg, evidence_data or [])
            
            # 生成评分详情
            score_details = {
                "raw_scores": {
                    "Q": score_q,
                    "V": score_v,
                    "M": score_m,
                    "C": score_c,
                    "S": score_s,
                    "R_neg": score_r_neg
                },
                "weights_used": weights,
                "breakdown": breakdown,
                "red_flags": red_flags,
                "calculation_method": "standard_formula",
                "calculated_at": datetime.utcnow().isoformat()
            }
            
            # 检查阈值
            meets_observe_threshold = total_score >= 65.0 and not has_red_flags
            meets_build_threshold = total_score >= settings.GATE_EQUITY_THRESHOLD and not has_red_flags
            
            result = {
                "equity_id": equity_id,
                "ticker": ticker,
                "total_score": total_score,
                "confidence": confidence,
                "breakdown": breakdown,
                "score_details": score_details,
                "has_red_flags": has_red_flags,
                "red_flags": red_flags,
                "meets_observe_threshold": meets_observe_threshold,
                "meets_build_threshold": meets_build_threshold,
                "evidence_count": len(evidence_data) if evidence_data else 0,
                "as_of": as_of or date.today(),
                "calculated_at": datetime.utcnow()
            }
            
            logger.log_scoring_event(
                event_type="equity_score_calculated",
                entity_type="equity",
                entity_id=equity_id,
                score=total_score,
                confidence=confidence,
                has_red_flags=has_red_flags
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Equity scoring failed for {ticker}: {e}")
            raise ScoringException(f"个股评分计算失败: {e}")
    
    def _calculate_industry_score_formula(
        self, 
        score_p: float, 
        score_e: float, 
        score_m: float, 
        score_r_neg: float,
        weights: Dict[str, float]
    ) -> float:
        """计算行业评分公式"""
        # PRD公式: Score^{Ind} = 0.35P + 0.25E + 0.25M + 0.15×(100-R^{-})
        total_score = (
            score_p * weights["P"] +
            score_e * weights["E"] +
            score_m * weights["M"] +
            (100 - score_r_neg) * weights["R"]
        )
        
        return round(max(0, min(100, total_score)), 2)
    
    def _calculate_equity_score_formula(
        self,
        score_q: float,
        score_v: float,
        score_m: float,
        score_c: float,
        score_s: float,
        score_r_neg: float,
        weights: Dict[str, float]
    ) -> float:
        """计算个股评分公式"""
        # 基础评分
        base_score = (
            score_q * weights["Q"] +
            score_v * weights["V"] +
            score_m * weights["M"] +
            score_c * weights["C"] +
            score_s * weights["S"]
        )
        
        # 扣除红旗分数
        final_score = base_score - score_r_neg
        
        return round(max(0, min(100, final_score)), 2)
    
    def _validate_score_inputs(self, scores: List[float]) -> None:
        """验证评分输入"""
        for i, score in enumerate(scores):
            if score is None:
                raise ScoringException(f"评分输入不能为空: 位置 {i}")
            if not isinstance(score, (int, float)):
                raise ScoringException(f"评分必须为数字: 位置 {i}, 值 {score}")
            if score < 0 or score > 100:
                raise ScoringException(f"评分必须在0-100之间: 位置 {i}, 值 {score}")
    
    def _calculate_confidence(self, evidence_data: List[Dict], scores: List[float]) -> float:
        """计算置信度"""
        # 基于证据数量和质量计算置信度
        evidence_count = len(evidence_data)
        
        # 证据数量权重
        evidence_weight = min(1.0, evidence_count / 5.0)  # 5个证据为满分
        
        # 评分完整性权重
        completeness_weight = len([s for s in scores if s is not None and s > 0]) / len(scores)
        
        # 证据质量权重
        quality_weight = 1.0
        if evidence_data:
            quality_scores = [e.get("quality_score", 80) for e in evidence_data]
            quality_weight = sum(quality_scores) / (len(quality_scores) * 100)
        
        # 综合置信度
        confidence = (evidence_weight * 0.4 + completeness_weight * 0.4 + quality_weight * 0.2) * 100
        
        return round(min(100, confidence), 2)
    
    def _analyze_red_flags(self, score_r_neg: float, evidence_data: List[Dict]) -> List[Dict]:
        """分析红旗指标"""
        red_flags = []
        
        if score_r_neg > 0:
            # 从证据中提取红旗信息
            for evidence in evidence_data:
                if evidence.get("evidence_type") == "red_flag":
                    red_flags.append({
                        "type": evidence.get("flag_type", "unknown"),
                        "severity": evidence.get("severity", "medium"),
                        "description": evidence.get("description", ""),
                        "score_impact": evidence.get("score_impact", 0)
                    })
        
        return red_flags


class GPUAcceleratedScoringEngine(ScoringEngine):
    """GPU加速的评分引擎"""
    
    def __init__(self):
        super().__init__()
        self.gpu_enabled = self.gpu_available and settings.ENABLE_GPU_ACCELERATION
        
        if self.gpu_enabled:
            try:
                import cupy as cp
                self.cp = cp
                logger.info("GPU scoring engine initialized")
            except ImportError:
                self.gpu_enabled = False
                logger.warning("CuPy not available, falling back to CPU")
    
    @gpu_fallback(ScoringEngine.batch_calculate_industry_scores)
    @log_performance("gpu_batch_industry_scoring")
    async def batch_calculate_industry_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPU加速的批量行业评分计算"""
        if not self.gpu_enabled:
            return await self._cpu_batch_calculate_industry_scores(score_data)
        
        try:
            with GPUMemoryMonitor():
                return await self._gpu_batch_calculate_industry_scores(score_data)
        except Exception as e:
            logger.warning(f"GPU batch scoring failed, falling back to CPU: {e}")
            return await self._cpu_batch_calculate_industry_scores(score_data)
    
    async def _gpu_batch_calculate_industry_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPU批量计算行业评分"""
        if not score_data:
            return []
        
        try:
            # 提取评分数据到GPU数组
            p_scores = self.cp.array([item["score_p"] for item in score_data], dtype=self.cp.float32)
            e_scores = self.cp.array([item["score_e"] for item in score_data], dtype=self.cp.float32)
            m_scores = self.cp.array([item["score_m"] for item in score_data], dtype=self.cp.float32)
            r_neg_scores = self.cp.array([item["score_r_neg"] for item in score_data], dtype=self.cp.float32)
            
            # 权重向量
            weights = self.industry_weights
            
            # GPU并行计算
            total_scores = (
                p_scores * weights["P"] +
                e_scores * weights["E"] +
                m_scores * weights["M"] +
                (100 - r_neg_scores) * weights["R"]
            )
            
            # 限制在0-100范围内
            total_scores = self.cp.clip(total_scores, 0, 100)
            
            # 转回CPU
            cpu_scores = self.cp.asnumpy(total_scores)
            
            # 构建结果
            results = []
            for i, item in enumerate(score_data):
                result = {
                    "industry_id": item["industry_id"],
                    "total_score": round(float(cpu_scores[i]), 2),
                    "breakdown": {
                        "P_weighted": item["score_p"] * weights["P"],
                        "E_weighted": item["score_e"] * weights["E"],
                        "M_weighted": item["score_m"] * weights["M"],
                        "R_weighted": (100 - item["score_r_neg"]) * weights["R"],
                    },
                    "meets_threshold": cpu_scores[i] >= settings.GATE_INDUSTRY_THRESHOLD,
                    "is_core_candidate": cpu_scores[i] >= 75.0,
                    "calculated_at": datetime.utcnow()
                }
                results.append(result)
            
            logger.log_gpu_operation(
                operation="batch_industry_scoring",
                batch_size=len(score_data)
            )
            
            return results
            
        except Exception as e:
            raise GPUException(f"GPU批量行业评分失败: {e}")
    
    async def _cpu_batch_calculate_industry_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """CPU批量计算行业评分"""
        results = []
        weights = self.industry_weights
        
        for item in score_data:
            total_score = self._calculate_industry_score_formula(
                item["score_p"], item["score_e"], item["score_m"], item["score_r_neg"], weights
            )
            
            result = {
                "industry_id": item["industry_id"],
                "total_score": total_score,
                "breakdown": {
                    "P_weighted": item["score_p"] * weights["P"],
                    "E_weighted": item["score_e"] * weights["E"],
                    "M_weighted": item["score_m"] * weights["M"],
                    "R_weighted": (100 - item["score_r_neg"]) * weights["R"],
                },
                "meets_threshold": total_score >= settings.GATE_INDUSTRY_THRESHOLD,
                "is_core_candidate": total_score >= 75.0,
                "calculated_at": datetime.utcnow()
            }
            results.append(result)
        
        return results
    
    @gpu_fallback(ScoringEngine.batch_calculate_equity_scores)
    @log_performance("gpu_batch_equity_scoring")
    async def batch_calculate_equity_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPU加速的批量个股评分计算"""
        if not self.gpu_enabled:
            return await self._cpu_batch_calculate_equity_scores(score_data)
        
        try:
            with GPUMemoryMonitor():
                return await self._gpu_batch_calculate_equity_scores(score_data)
        except Exception as e:
            logger.warning(f"GPU batch equity scoring failed, falling back to CPU: {e}")
            return await self._cpu_batch_calculate_equity_scores(score_data)
    
    async def _gpu_batch_calculate_equity_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """GPU批量计算个股评分"""
        if not score_data:
            return []
        
        try:
            # 提取评分数据到GPU数组
            q_scores = self.cp.array([item["score_q"] for item in score_data], dtype=self.cp.float32)
            v_scores = self.cp.array([item["score_v"] for item in score_data], dtype=self.cp.float32)
            m_scores = self.cp.array([item["score_m"] for item in score_data], dtype=self.cp.float32)
            c_scores = self.cp.array([item["score_c"] for item in score_data], dtype=self.cp.float32)
            s_scores = self.cp.array([item["score_s"] for item in score_data], dtype=self.cp.float32)
            r_neg_scores = self.cp.array([item.get("score_r_neg", 0) for item in score_data], dtype=self.cp.float32)
            
            # 权重
            weights = self.equity_weights
            
            # GPU并行计算基础评分
            base_scores = (
                q_scores * weights["Q"] +
                v_scores * weights["V"] +
                m_scores * weights["M"] +
                c_scores * weights["C"] +
                s_scores * weights["S"]
            )
            
            # 扣除红旗分数
            total_scores = base_scores - r_neg_scores
            
            # 限制在0-100范围内
            total_scores = self.cp.clip(total_scores, 0, 100)
            
            # 转回CPU
            cpu_scores = self.cp.asnumpy(total_scores)
            
            # 构建结果
            results = []
            for i, item in enumerate(score_data):
                has_red_flags = item.get("score_r_neg", 0) > 0
                
                result = {
                    "equity_id": item["equity_id"],
                    "ticker": item["ticker"],
                    "total_score": round(float(cpu_scores[i]), 2),
                    "breakdown": {
                        "Q_weighted": item["score_q"] * weights["Q"],
                        "V_weighted": item["score_v"] * weights["V"],
                        "M_weighted": item["score_m"] * weights["M"],
                        "C_weighted": item["score_c"] * weights["C"],
                        "S_weighted": item["score_s"] * weights["S"],
                        "R_neg_penalty": item.get("score_r_neg", 0),
                    },
                    "has_red_flags": has_red_flags,
                    "meets_observe_threshold": cpu_scores[i] >= 65.0 and not has_red_flags,
                    "meets_build_threshold": cpu_scores[i] >= settings.GATE_EQUITY_THRESHOLD and not has_red_flags,
                    "calculated_at": datetime.utcnow()
                }
                results.append(result)
            
            logger.log_gpu_operation(
                operation="batch_equity_scoring",
                batch_size=len(score_data)
            )
            
            return results
            
        except Exception as e:
            raise GPUException(f"GPU批量个股评分失败: {e}")
    
    async def _cpu_batch_calculate_equity_scores(
        self,
        score_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """CPU批量计算个股评分"""
        results = []
        weights = self.equity_weights
        
        for item in score_data:
            total_score = self._calculate_equity_score_formula(
                item["score_q"], item["score_v"], item["score_m"],
                item["score_c"], item["score_s"], item.get("score_r_neg", 0), weights
            )
            
            has_red_flags = item.get("score_r_neg", 0) > 0
            
            result = {
                "equity_id": item["equity_id"],
                "ticker": item["ticker"],
                "total_score": total_score,
                "breakdown": {
                    "Q_weighted": item["score_q"] * weights["Q"],
                    "V_weighted": item["score_v"] * weights["V"],
                    "M_weighted": item["score_m"] * weights["M"],
                    "C_weighted": item["score_c"] * weights["C"],
                    "S_weighted": item["score_s"] * weights["S"],
                    "R_neg_penalty": item.get("score_r_neg", 0),
                },
                "has_red_flags": has_red_flags,
                "meets_observe_threshold": total_score >= 65.0 and not has_red_flags,
                "meets_build_threshold": total_score >= settings.GATE_EQUITY_THRESHOLD and not has_red_flags,
                "calculated_at": datetime.utcnow()
            }
            results.append(result)
        
        return results


# 创建全局评分引擎实例
scoring_engine = GPUAcceleratedScoringEngine()
