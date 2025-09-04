"""
InvestIQ Platform - 流动性校验服务
实现流动性和容量检查逻辑
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.exceptions import LiquidityException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.equity import Equity
from backend.app.models.portfolio import LiquidityCheck


logger = get_logger(__name__)


class LiquidityChecker:
    """流动性校验器"""
    
    def __init__(self):
        self.liquidity_config = settings.liquidity_config
        logger.info("Liquidity checker initialized", config=self.liquidity_config)
    
    @log_performance("liquidity_check")
    async def check_liquidity(
        self,
        ticker: str,
        target_position: float,
        currency: str,
        adv_20: float,
        turnover: float,
        free_float_market_cap: float,
        participation_rate: Optional[float] = None,
        exit_days: Optional[int] = None,
        market_type: str = "A"
    ) -> Dict[str, Any]:
        """
        执行流动性校验
        
        Args:
            ticker: 股票代码
            target_position: 目标仓位金额
            currency: 货币类型
            adv_20: 20日平均成交额
            turnover: 换手率
            free_float_market_cap: 自由流通市值
            participation_rate: 参与率 (可选，使用默认值)
            exit_days: 退出天数 (可选，使用默认值)
            market_type: 市场类型 (A/H)
        
        Returns:
            流动性校验结果
        """
        try:
            # 确定参与率和退出天数
            used_participation_rate = participation_rate or self._get_default_participation_rate(market_type)
            used_exit_days = exit_days or self._get_default_exit_days("core")  # 默认使用核心持仓退出天数
            
            # 计算最小ADV要求
            adv_min_required = target_position / (used_participation_rate * used_exit_days)
            
            # 执行各项检查
            adv_check = self._check_adv_requirement(adv_20, adv_min_required)
            absolute_floor_check = self._check_absolute_floor(adv_20, turnover, market_type)
            free_float_check = self._check_free_float_capacity(
                target_position, free_float_market_cap, currency
            )
            
            # 计算自由流通占用百分比
            free_float_utilization_pct = (target_position / free_float_market_cap) * 100 if free_float_market_cap > 0 else 0
            
            # 综合判断
            overall_pass = adv_check["pass"] and absolute_floor_check["pass"] and free_float_check["pass"]
            
            # 生成建议和备注
            notes = []
            recommendations = []
            
            if not adv_check["pass"]:
                notes.append(f"ADV不足: 需要{adv_min_required:,.0f}，实际{adv_20:,.0f}")
                recommendations.append("考虑降低目标仓位或延长建仓周期")
            
            if not absolute_floor_check["pass"]:
                notes.append("未达到绝对底线要求")
                recommendations.append("选择流动性更好的标的")
            
            if not free_float_check["pass"]:
                notes.append("自由流通占用过高")
                recommendations.append("降低目标仓位以控制市场影响")
            
            if overall_pass:
                notes.append("流动性检查通过")
                recommendations.append("可以按计划建仓")
            
            result = {
                "ticker": ticker,
                "overall_pass": overall_pass,
                "adv_min_required": adv_min_required,
                "absolute_floor_pass": absolute_floor_check["pass"],
                "free_float_cap_pass": free_float_check["pass"],
                "used_participation_rate": used_participation_rate,
                "used_exit_days": used_exit_days,
                "free_float_utilization_pct": round(free_float_utilization_pct, 4),
                "notes": notes,
                "recommendations": recommendations,
                "details": {
                    "adv_check": adv_check,
                    "absolute_floor_check": absolute_floor_check,
                    "free_float_check": free_float_check
                },
                "market_data": {
                    "adv_20": adv_20,
                    "turnover": turnover,
                    "free_float_market_cap": free_float_market_cap,
                    "currency": currency
                },
                "checked_at": datetime.utcnow()
            }
            
            logger.log_portfolio_event(
                event_type="liquidity_check",
                ticker=ticker,
                action="check_liquidity",
                overall_pass=overall_pass,
                target_position=target_position,
                adv_20=adv_20
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Liquidity check failed for {ticker}: {e}")
            raise LiquidityException(f"流动性检查失败: {e}")
    
    def _get_default_participation_rate(self, market_type: str) -> float:
        """获取默认参与率"""
        return self.liquidity_config["participation_rate"].get(market_type, 0.10)
    
    def _get_default_exit_days(self, position_type: str) -> int:
        """获取默认退出天数"""
        return self.liquidity_config["exit_days"].get(position_type, 5)
    
    def _check_adv_requirement(self, adv_20: float, adv_min_required: float) -> Dict[str, Any]:
        """检查ADV要求"""
        pass_check = adv_20 >= adv_min_required
        
        return {
            "pass": pass_check,
            "adv_20": adv_20,
            "adv_min_required": adv_min_required,
            "margin": adv_20 - adv_min_required,
            "margin_pct": ((adv_20 - adv_min_required) / adv_min_required) if adv_min_required > 0 else 0
        }
    
    def _check_absolute_floor(self, adv_20: float, turnover: float, market_type: str) -> Dict[str, Any]:
        """检查绝对底线要求"""
        # 获取市场特定的底线要求
        adv_min = self.liquidity_config["adv_min"].get(market_type, 30000000)
        turnover_min = self.liquidity_config["turnover_min"].get(market_type, 0.005)
        
        adv_pass = adv_20 >= adv_min
        turnover_pass = turnover >= turnover_min
        overall_pass = adv_pass and turnover_pass
        
        return {
            "pass": overall_pass,
            "adv_pass": adv_pass,
            "turnover_pass": turnover_pass,
            "adv_20": adv_20,
            "adv_min_required": adv_min,
            "turnover": turnover,
            "turnover_min_required": turnover_min,
            "market_type": market_type
        }
    
    def _check_free_float_capacity(
        self, 
        target_position: float, 
        free_float_market_cap: float,
        currency: str
    ) -> Dict[str, Any]:
        """检查自由流通容量"""
        if free_float_market_cap <= 0:
            return {
                "pass": False,
                "error": "自由流通市值数据无效",
                "free_float_market_cap": free_float_market_cap
            }
        
        # 计算占用比例
        utilization_pct = target_position / free_float_market_cap
        max_utilization = 0.02  # 2%上限
        
        pass_check = utilization_pct <= max_utilization
        
        return {
            "pass": pass_check,
            "utilization_pct": utilization_pct,
            "max_utilization": max_utilization,
            "target_position": target_position,
            "free_float_market_cap": free_float_market_cap,
            "currency": currency,
            "margin_pct": max_utilization - utilization_pct
        }
    
    async def batch_check_liquidity(
        self,
        check_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量流动性检查"""
        results = []
        
        for request in check_requests:
            try:
                result = await self.check_liquidity(**request)
                result["request_id"] = request.get("request_id")
                results.append(result)
            except Exception as e:
                error_result = {
                    "request_id": request.get("request_id"),
                    "ticker": request.get("ticker"),
                    "overall_pass": False,
                    "error": str(e),
                    "checked_at": datetime.utcnow()
                }
                results.append(error_result)
        
        return results
    
    async def get_liquidity_requirements(self, market_type: str = "A") -> Dict[str, Any]:
        """获取流动性要求说明"""
        config = self.liquidity_config
        
        return {
            "market_type": market_type,
            "participation_rate": config["participation_rate"].get(market_type, 0.10),
            "exit_days": {
                "core": config["exit_days"]["core"],
                "tactical": config["exit_days"]["tactical"]
            },
            "absolute_floors": {
                "adv_min": config["adv_min"].get(market_type, 30000000),
                "turnover_min": config["turnover_min"].get(market_type, 0.005)
            },
            "free_float_cap_max": 0.02,
            "formula": "ADV_min >= target_position / (participation_rate × exit_days)",
            "description": {
                "participation_rate": "单日最大参与成交量比例",
                "exit_days": "完全退出所需天数",
                "absolute_floor": "不可突破的最低流动性要求",
                "free_float_cap": "自由流通市值占用上限"
            }
        }


class CapacityCalculator:
    """容量计算器"""
    
    def __init__(self):
        self.liquidity_checker = LiquidityChecker()
    
    @log_performance("capacity_calculation")
    async def calculate_max_position(
        self,
        ticker: str,
        adv_20: float,
        free_float_market_cap: float,
        market_type: str = "A",
        position_type: str = "core"
    ) -> Dict[str, Any]:
        """
        计算最大可建仓位
        
        Args:
            ticker: 股票代码
            adv_20: 20日平均成交额
            free_float_market_cap: 自由流通市值
            market_type: 市场类型
            position_type: 持仓类型 (core/tactical)
        
        Returns:
            最大仓位计算结果
        """
        try:
            # 获取参数
            participation_rate = self.liquidity_checker._get_default_participation_rate(market_type)
            exit_days = self.liquidity_checker._get_default_exit_days(position_type)
            
            # 基于ADV计算的最大仓位
            max_position_adv = adv_20 * participation_rate * exit_days
            
            # 基于自由流通市值计算的最大仓位
            max_position_float = free_float_market_cap * 0.02  # 2%上限
            
            # 取较小值作为最终限制
            max_position = min(max_position_adv, max_position_float)
            
            # 检查绝对底线
            absolute_floor = self.liquidity_checker.liquidity_config["adv_min"].get(market_type, 30000000)
            meets_absolute_floor = adv_20 >= absolute_floor
            
            result = {
                "ticker": ticker,
                "max_position": max_position,
                "max_position_adv": max_position_adv,
                "max_position_float": max_position_float,
                "limiting_factor": "adv" if max_position_adv < max_position_float else "free_float",
                "meets_absolute_floor": meets_absolute_floor,
                "parameters": {
                    "adv_20": adv_20,
                    "free_float_market_cap": free_float_market_cap,
                    "participation_rate": participation_rate,
                    "exit_days": exit_days,
                    "market_type": market_type,
                    "position_type": position_type
                },
                "calculated_at": datetime.utcnow()
            }
            
            if not meets_absolute_floor:
                result["warning"] = f"ADV低于绝对底线 {absolute_floor:,.0f}"
            
            return result
            
        except Exception as e:
            logger.error(f"Capacity calculation failed for {ticker}: {e}")
            raise LiquidityException(f"容量计算失败: {e}")
    
    async def batch_calculate_capacity(
        self,
        calculation_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """批量容量计算"""
        results = []
        
        for request in calculation_requests:
            try:
                result = await self.calculate_max_position(**request)
                result["request_id"] = request.get("request_id")
                results.append(result)
            except Exception as e:
                error_result = {
                    "request_id": request.get("request_id"),
                    "ticker": request.get("ticker"),
                    "max_position": 0,
                    "error": str(e),
                    "calculated_at": datetime.utcnow()
                }
                results.append(error_result)
        
        return results


class BoardLotValidator:
    """整手校验器 - 特别针对H股"""
    
    @staticmethod
    def validate_board_lot(
        ticker: str,
        shares: int,
        board_lot: int = 100,
        market_type: str = "A"
    ) -> Dict[str, Any]:
        """
        校验整手要求
        
        Args:
            ticker: 股票代码
            shares: 股数
            board_lot: 每手股数
            market_type: 市场类型
        
        Returns:
            整手校验结果
        """
        if market_type == "H" and shares % board_lot != 0:
            # H股必须整手交易
            suggested_shares_down = (shares // board_lot) * board_lot
            suggested_shares_up = suggested_shares_down + board_lot
            
            return {
                "ticker": ticker,
                "valid": False,
                "shares": shares,
                "board_lot": board_lot,
                "remainder": shares % board_lot,
                "suggestions": {
                    "round_down": suggested_shares_down,
                    "round_up": suggested_shares_up
                },
                "message": f"H股必须整手交易，建议调整为 {suggested_shares_down} 或 {suggested_shares_up} 股",
                "market_type": market_type
            }
        else:
            return {
                "ticker": ticker,
                "valid": True,
                "shares": shares,
                "board_lot": board_lot,
                "message": "股数符合交易规则",
                "market_type": market_type
            }


class LiquidityOptimizer:
    """流动性优化器"""
    
    def __init__(self):
        self.liquidity_checker = LiquidityChecker()
        self.capacity_calculator = CapacityCalculator()
    
    @log_performance("liquidity_optimization")
    async def optimize_position_size(
        self,
        ticker: str,
        desired_position: float,
        market_data: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        优化仓位大小
        
        Args:
            ticker: 股票代码
            desired_position: 期望仓位
            market_data: 市场数据
            constraints: 约束条件
        
        Returns:
            优化后的仓位建议
        """
        try:
            # 计算最大可建仓位
            max_capacity = await self.capacity_calculator.calculate_max_position(
                ticker=ticker,
                adv_20=market_data["adv_20"],
                free_float_market_cap=market_data["free_float_market_cap"],
                market_type=market_data.get("market_type", "A")
            )
            
            # 确定优化后的仓位
            max_position = max_capacity["max_position"]
            optimized_position = min(desired_position, max_position)
            
            # 执行流动性检查
            liquidity_result = await self.liquidity_checker.check_liquidity(
                ticker=ticker,
                target_position=optimized_position,
                currency=market_data.get("currency", "CNY"),
                adv_20=market_data["adv_20"],
                turnover=market_data["turnover"],
                free_float_market_cap=market_data["free_float_market_cap"],
                market_type=market_data.get("market_type", "A")
            )
            
            # 生成优化建议
            optimization_notes = []
            if optimized_position < desired_position:
                reduction_pct = (desired_position - optimized_position) / desired_position
                optimization_notes.append(f"仓位已优化，减少 {reduction_pct:.1%}")
                optimization_notes.append(f"限制因素: {max_capacity['limiting_factor']}")
            
            result = {
                "ticker": ticker,
                "desired_position": desired_position,
                "optimized_position": optimized_position,
                "max_capacity": max_position,
                "optimization_ratio": optimized_position / desired_position if desired_position > 0 else 0,
                "liquidity_check": liquidity_result,
                "capacity_analysis": max_capacity,
                "optimization_notes": optimization_notes,
                "optimized_at": datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Position optimization failed for {ticker}: {e}")
            raise LiquidityException(f"仓位优化失败: {e}")


# 创建全局实例
liquidity_checker = LiquidityChecker()
capacity_calculator = CapacityCalculator()
board_lot_validator = BoardLotValidator()
liquidity_optimizer = LiquidityOptimizer()
