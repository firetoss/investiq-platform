"""
InvestIQ Platform - 四闸门校验服务
实现投资决策的四重校验逻辑
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.exceptions import GatekeeperException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.industry import IndustryScoreSnapshot
from backend.app.models.equity import EquityScoreSnapshot, TechnicalIndicator, ValuationPercentileSnapshot


logger = get_logger(__name__)


class FourGateKeeper:
    """四闸门校验器"""
    
    def __init__(self):
        self.gate_thresholds = settings.gate_thresholds
        logger.info("Four Gate Keeper initialized with thresholds", thresholds=self.gate_thresholds)
    
    @log_performance("four_gate_check")
    async def check_all_gates(
        self,
        industry_score: float,
        equity_score: float,
        valuation_percentile: Optional[float],
        above_200dma: bool,
        peg: Optional[float] = None,
        is_growth_stock: bool = False,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行四闸门完整校验
        
        Args:
            industry_score: 行业评分
            equity_score: 个股评分
            valuation_percentile: 估值分位 (0-1)
            above_200dma: 是否在200日均线上方
            peg: PEG比率 (可选)
            is_growth_stock: 是否为成长股
            additional_context: 额外上下文信息
        
        Returns:
            校验结果字典
        """
        try:
            gates_result = {}
            overall_pass = True
            failed_gates = []
            
            # Gate 1: 行业闸门
            gate1_result = await self._check_industry_gate(industry_score)
            gates_result["industry"] = gate1_result
            if not gate1_result["pass"]:
                overall_pass = False
                failed_gates.append("industry")
            
            # Gate 2: 公司闸门
            gate2_result = await self._check_company_gate(equity_score)
            gates_result["company"] = gate2_result
            if not gate2_result["pass"]:
                overall_pass = False
                failed_gates.append("company")
            
            # Gate 3: 估值闸门
            gate3_result = await self._check_valuation_gate(
                valuation_percentile, peg, is_growth_stock
            )
            gates_result["valuation"] = gate3_result
            if not gate3_result["pass"]:
                overall_pass = False
                failed_gates.append("valuation")
            
            # Gate 4: 执行闸门
            gate4_result = await self._check_execution_gate(above_200dma)
            gates_result["execution"] = gate4_result
            if not gate4_result["pass"]:
                overall_pass = False
                failed_gates.append("execution")
            
            # 生成综合结果
            result = {
                "overall_pass": overall_pass,
                "failed_gates": failed_gates,
                "gates": gates_result,
                "summary": {
                    "total_gates": 4,
                    "passed_gates": 4 - len(failed_gates),
                    "pass_rate": (4 - len(failed_gates)) / 4
                },
                "context": additional_context or {},
                "checked_at": datetime.utcnow(),
                "thresholds_used": self.gate_thresholds
            }
            
            logger.log_scoring_event(
                event_type="four_gate_check",
                entity_type="gatekeeper",
                entity_id="system",
                overall_pass=overall_pass,
                failed_gates=failed_gates,
                industry_score=industry_score,
                equity_score=equity_score
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Four gate check failed: {e}")
            raise GatekeeperException(f"四闸门校验失败: {e}")
    
    async def _check_industry_gate(self, industry_score: float) -> Dict[str, Any]:
        """
        Gate 1: 行业闸门校验
        
        要求: 行业评分 >= 70分，且处于"落地/兑现期"
        """
        threshold = self.gate_thresholds["industry"]
        pass_check = industry_score >= threshold
        
        return {
            "gate_name": "industry",
            "pass": pass_check,
            "score": industry_score,
            "threshold": threshold,
            "margin": industry_score - threshold,
            "message": "行业评分达标" if pass_check else f"行业评分不足，需要 >= {threshold}",
            "details": {
                "requirement": "行业分达标，处落地/兑现期",
                "current_score": industry_score,
                "min_required": threshold
            }
        }
    
    async def _check_company_gate(self, equity_score: float) -> Dict[str, Any]:
        """
        Gate 2: 公司闸门校验
        
        要求: 个股评分 >= 70分，且无红旗
        """
        threshold = self.gate_thresholds["equity"]
        pass_check = equity_score >= threshold
        
        return {
            "gate_name": "company",
            "pass": pass_check,
            "score": equity_score,
            "threshold": threshold,
            "margin": equity_score - threshold,
            "message": "公司评分达标" if pass_check else f"公司评分不足，需要 >= {threshold}",
            "details": {
                "requirement": "个股分达标，无红旗",
                "current_score": equity_score,
                "min_required": threshold
            }
        }
    
    async def _check_valuation_gate(
        self, 
        valuation_percentile: Optional[float],
        peg: Optional[float] = None,
        is_growth_stock: bool = False
    ) -> Dict[str, Any]:
        """
        Gate 3: 估值闸门校验
        
        要求: 
        - 普通股票: 估值分位 <= 70%
        - 成长股票: 估值分位 <= 80% 且 PEG <= 1.5
        """
        if valuation_percentile is None:
            return {
                "gate_name": "valuation",
                "pass": False,
                "message": "估值分位数据缺失",
                "details": {
                    "requirement": "估值分位数据必须可用",
                    "current_percentile": None,
                    "error": "数据缺失"
                }
            }
        
        # 确定阈值
        if is_growth_stock:
            percentile_threshold = self.gate_thresholds["growth_percentile_max"]
            peg_threshold = self.gate_thresholds["peg_max"]
            
            # 成长股需要同时满足估值分位和PEG要求
            percentile_pass = valuation_percentile <= percentile_threshold
            peg_pass = peg is not None and peg <= peg_threshold
            pass_check = percentile_pass and peg_pass
            
            message_parts = []
            if not percentile_pass:
                message_parts.append(f"估值分位过高 ({valuation_percentile:.1%} > {percentile_threshold:.1%})")
            if not peg_pass:
                if peg is None:
                    message_parts.append("PEG数据缺失")
                else:
                    message_parts.append(f"PEG过高 ({peg:.2f} > {peg_threshold})")
            
            message = "成长股估值检查通过" if pass_check else f"成长股估值检查失败: {'; '.join(message_parts)}"
            
            return {
                "gate_name": "valuation",
                "pass": pass_check,
                "percentile": valuation_percentile,
                "peg": peg,
                "is_growth_stock": True,
                "percentile_threshold": percentile_threshold,
                "peg_threshold": peg_threshold,
                "message": message,
                "details": {
                    "requirement": f"成长股: 估值分位 <= {percentile_threshold:.1%} 且 PEG <= {peg_threshold}",
                    "current_percentile": valuation_percentile,
                    "current_peg": peg,
                    "percentile_pass": percentile_pass,
                    "peg_pass": peg_pass
                }
            }
        else:
            # 普通股票只检查估值分位
            percentile_threshold = self.gate_thresholds["valuation_percentile_max"]
            pass_check = valuation_percentile <= percentile_threshold
            
            message = "估值检查通过" if pass_check else f"估值分位过高 ({valuation_percentile:.1%} > {percentile_threshold:.1%})"
            
            return {
                "gate_name": "valuation",
                "pass": pass_check,
                "percentile": valuation_percentile,
                "is_growth_stock": False,
                "percentile_threshold": percentile_threshold,
                "message": message,
                "details": {
                    "requirement": f"估值分位 <= {percentile_threshold:.1%}",
                    "current_percentile": valuation_percentile,
                    "threshold": percentile_threshold
                }
            }
    
    async def _check_execution_gate(self, above_200dma: bool) -> Dict[str, Any]:
        """
        Gate 4: 执行闸门校验
        
        要求: 价格在200日均线上方，支持三段式建仓
        """
        pass_check = above_200dma
        
        return {
            "gate_name": "execution",
            "pass": pass_check,
            "above_200dma": above_200dma,
            "message": "执行条件满足" if pass_check else "价格未在200日均线上方",
            "details": {
                "requirement": "价格在200日均线上方",
                "current_status": "在200DMA上方" if above_200dma else "在200DMA下方",
                "entry_strategy": "三段式建仓 (40%/30%/30%)" if pass_check else "暂不建仓"
            }
        }
    
    async def check_single_gate(
        self,
        gate_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """检查单个闸门"""
        gate_methods = {
            "industry": self._check_industry_gate,
            "company": self._check_company_gate,
            "valuation": self._check_valuation_gate,
            "execution": self._check_execution_gate
        }
        
        if gate_name not in gate_methods:
            raise GatekeeperException(f"未知的闸门类型: {gate_name}")
        
        try:
            return await gate_methods[gate_name](**kwargs)
        except Exception as e:
            logger.error(f"Single gate check failed for {gate_name}: {e}")
            raise GatekeeperException(f"闸门 {gate_name} 校验失败: {e}")
    
    async def batch_check_gates(
        self,
        check_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量四闸门校验"""
        results = []
        
        for request in check_requests:
            try:
                result = await self.check_all_gates(**request)
                result["request_id"] = request.get("request_id")
                results.append(result)
            except Exception as e:
                error_result = {
                    "request_id": request.get("request_id"),
                    "overall_pass": False,
                    "error": str(e),
                    "checked_at": datetime.utcnow()
                }
                results.append(error_result)
        
        return results
    
    def get_gate_requirements(self) -> Dict[str, Dict]:
        """获取所有闸门要求"""
        return {
            "industry": {
                "name": "行业闸门",
                "requirement": f"行业评分 >= {self.gate_thresholds['industry']}",
                "description": "行业分达标，处于落地/兑现期"
            },
            "company": {
                "name": "公司闸门", 
                "requirement": f"个股评分 >= {self.gate_thresholds['equity']}",
                "description": "个股分达标，无红旗"
            },
            "valuation": {
                "name": "估值闸门",
                "requirement": f"估值分位 <= {self.gate_thresholds['valuation_percentile_max']:.1%}",
                "description": f"普通股估值分位 <= {self.gate_thresholds['valuation_percentile_max']:.1%}；成长股 <= {self.gate_thresholds['growth_percentile_max']:.1%} 且 PEG <= {self.gate_thresholds['peg_max']}"
            },
            "execution": {
                "name": "执行闸门",
                "requirement": "价格在200日均线上方",
                "description": "价格在长期趋势线上，支持三段式建仓"
            }
        }


class EntryPlanGenerator:
    """建仓计划生成器"""
    
    def __init__(self):
        self.default_entry_plan = [0.4, 0.3, 0.3]  # 三段式建仓比例
    
    @log_performance("entry_plan_generation")
    async def generate_entry_plan(
        self,
        ticker: str,
        target_position: float,
        current_price: float,
        technical_data: Optional[Dict] = None,
        risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """
        生成建仓计划
        
        Args:
            ticker: 股票代码
            target_position: 目标仓位金额
            current_price: 当前价格
            technical_data: 技术分析数据
            risk_tolerance: 风险承受度 (low/medium/high)
        
        Returns:
            建仓计划
        """
        try:
            # 根据风险承受度调整建仓比例
            entry_ratios = self._get_entry_ratios(risk_tolerance)
            
            # 计算各段建仓金额
            entry_legs = []
            for i, ratio in enumerate(entry_ratios):
                leg_amount = target_position * ratio
                leg_shares = int(leg_amount / current_price)
                
                # 计算建仓条件
                condition = self._generate_entry_condition(i + 1, technical_data)
                
                entry_legs.append({
                    "leg": i + 1,
                    "ratio": ratio,
                    "amount": leg_amount,
                    "shares": leg_shares,
                    "condition": condition,
                    "status": "pending"
                })
            
            # 生成止损和止盈建议
            risk_levels = self._calculate_risk_levels(current_price, technical_data)
            
            plan = {
                "ticker": ticker,
                "target_position": target_position,
                "current_price": current_price,
                "entry_legs": entry_legs,
                "risk_management": risk_levels,
                "total_shares": sum(leg["shares"] for leg in entry_legs),
                "estimated_cost": sum(leg["amount"] for leg in entry_legs),
                "risk_tolerance": risk_tolerance,
                "generated_at": datetime.utcnow()
            }
            
            logger.log_portfolio_event(
                event_type="entry_plan_generated",
                ticker=ticker,
                action="generate_plan",
                target_amount=target_position,
                legs_count=len(entry_legs)
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Entry plan generation failed for {ticker}: {e}")
            raise GatekeeperException(f"建仓计划生成失败: {e}")
    
    def _get_entry_ratios(self, risk_tolerance: str) -> List[float]:
        """根据风险承受度获取建仓比例"""
        ratios_map = {
            "low": [0.5, 0.3, 0.2],      # 保守型：更多首段建仓
            "medium": [0.4, 0.3, 0.3],   # 平衡型：标准三段式
            "high": [0.3, 0.3, 0.4]      # 激进型：更多后段建仓
        }
        return ratios_map.get(risk_tolerance, self.default_entry_plan)
    
    def _generate_entry_condition(self, leg_number: int, technical_data: Optional[Dict]) -> str:
        """生成建仓条件"""
        if not technical_data:
            return f"第{leg_number}段: 立即执行"
        
        # 基于技术指标生成条件
        conditions = {
            1: "立即执行",  # 第一段立即执行
            2: "价格回调至20日均线附近或突破前高",
            3: "价格进一步确认趋势或出现超跌反弹机会"
        }
        
        # 可以根据技术数据进一步细化条件
        if technical_data:
            ma_20 = technical_data.get("ma_20")
            current_price = technical_data.get("current_price")
            
            if leg_number == 2 and ma_20 and current_price:
                if current_price > ma_20 * 1.05:  # 价格高于20日均线5%以上
                    conditions[2] = f"价格回调至{ma_20:.2f}附近 (20日均线)"
        
        return conditions.get(leg_number, f"第{leg_number}段: 待定")
    
    def _calculate_risk_levels(self, current_price: float, technical_data: Optional[Dict]) -> Dict[str, float]:
        """计算风险管理水平"""
        risk_levels = {
            "stop_loss": current_price * 0.9,    # 默认10%止损
            "take_profit": current_price * 1.2,  # 默认20%止盈
        }
        
        if technical_data:
            # 基于技术指标调整止损止盈
            ma_200 = technical_data.get("ma_200")
            atr = technical_data.get("atr")  # 平均真实波幅
            
            if ma_200:
                # 止损设在200日均线附近
                risk_levels["stop_loss"] = max(risk_levels["stop_loss"], ma_200 * 0.98)
            
            if atr:
                # 基于ATR设置止盈
                risk_levels["take_profit"] = current_price + (atr * 2)
        
        return risk_levels


# 创建全局四闸门校验器实例
gatekeeper = FourGateKeeper()
entry_plan_generator = EntryPlanGenerator()
