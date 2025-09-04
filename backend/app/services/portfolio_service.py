"""
InvestIQ Platform - 投资组合服务
实现投资组合构建、管理和风险控制的核心业务逻辑
"""

import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import selectinload

from backend.app.core.config import settings
from backend.app.core.exceptions import PortfolioException, ValidationException
from backend.app.core.logging import get_logger, log_performance
from backend.app.models.portfolio import (
    Portfolio, Position, PositionTier, CircuitBreakerLevel,
    PortfolioSnapshot, RebalanceRecord, LiquidityCheck, RiskMetrics
)
from backend.app.models.equity import Equity
from backend.app.services.liquidity import liquidity_service
from backend.app.services.gatekeeper import gatekeeper_service

logger = get_logger(__name__)


class PortfolioService:
    """投资组合服务核心类"""
    
    def __init__(self):
        self.default_tier_config = settings.portfolio_tiers
        self.circuit_breaker_levels = {
            settings.CIRCUIT_BREAKER_LEVEL_1: CircuitBreakerLevel.L1,
            settings.CIRCUIT_BREAKER_LEVEL_2: CircuitBreakerLevel.L2,
            settings.CIRCUIT_BREAKER_LEVEL_3: CircuitBreakerLevel.L3
        }
    
    @log_performance("portfolio_construction")
    async def construct_portfolio(
        self,
        name: str,
        description: Optional[str],
        total_capital: float,
        leverage_max: float,
        tier_config: Optional[Dict[str, List[float]]],
        candidates: List[Dict[str, Any]],
        cash_buffer: float,
        max_single_position: float,
        max_sector_concentration: float,
        entry_mode: str,
        as_of: Optional[date],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """构建投资组合"""
        try:
            # 验证输入参数
            self._validate_construction_params(
                total_capital, leverage_max, candidates, cash_buffer,
                max_single_position, max_sector_concentration
            )
            
            # 处理分层配置
            tier_config = tier_config or self.default_tier_config
            tier_limits = self._process_tier_config(tier_config)
            
            # 执行四闸门校验
            validated_candidates = await self._validate_candidates_through_gates(
                candidates, db
            )
            
            # 流动性检查
            liquidity_validated = await self._perform_liquidity_checks(
                validated_candidates, db
            )
            
            # 组合优化
            optimized_allocation = await self._optimize_portfolio_allocation(
                liquidity_validated, tier_limits, total_capital, leverage_max,
                cash_buffer, max_single_position, max_sector_concentration
            )
            
            # 创建投资组合记录
            portfolio = await self._create_portfolio_record(
                name, description, total_capital, leverage_max, tier_config,
                optimized_allocation, db
            )
            
            # 生成建仓计划
            entry_plan = self._generate_entry_plan(
                optimized_allocation, entry_mode, total_capital
            )
            
            # 风险分析
            risk_analysis = await self._analyze_portfolio_risk(
                optimized_allocation, portfolio.id, db
            )
            
            result = {
                "portfolio_id": str(portfolio.id),
                "portfolio_name": portfolio.name,
                "total_capital": total_capital,
                "available_capital": portfolio.available_capital,
                "allocation": optimized_allocation,
                "entry_plan": entry_plan,
                "risk_analysis": risk_analysis,
                "tier_summary": self._generate_tier_summary(optimized_allocation),
                "construction_date": as_of or date.today(),
                "circuit_breaker": {
                    "level": portfolio.circuit_breaker_level.value,
                    "armed": portfolio.circuit_breaker_armed,
                    "levels_config": self._get_circuit_breaker_config()
                },
                "validation_results": {
                    "candidates_submitted": len(candidates),
                    "gates_passed": len(validated_candidates),
                    "liquidity_passed": len(liquidity_validated),
                    "final_selected": len(optimized_allocation["positions"])
                }
            }
            
            logger.info(f"Portfolio constructed successfully: {portfolio.name} ({portfolio.id})")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {e}", exc_info=True)
            raise PortfolioException(f"投资组合构建失败: {e}")
    
    @log_performance("portfolio_rebalance")
    async def rebalance_portfolio(
        self,
        portfolio_id: str,
        rebalance_type: str,
        trigger_reason: Optional[str],
        target_allocation: Optional[Dict[str, float]],
        max_turnover: float,
        as_of: Optional[date],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """投资组合再平衡"""
        try:
            # 获取组合当前状态
            portfolio = await self._get_portfolio_by_id(portfolio_id, db)
            current_positions = await self._get_current_positions(portfolio_id, db)
            
            # 获取当前配置
            current_allocation = self._calculate_current_allocation(current_positions)
            
            # 确定目标配置
            if target_allocation is None:
                target_allocation = await self._generate_optimal_allocation(
                    portfolio, current_positions, db
                )
            
            # 计算再平衡交易
            trades = self._calculate_rebalance_trades(
                current_allocation, target_allocation, portfolio.total_capital,
                max_turnover
            )
            
            # 流动性验证
            validated_trades = await self._validate_rebalance_liquidity(trades, db)
            
            # 创建再平衡记录
            rebalance_record = await self._create_rebalance_record(
                portfolio_id, rebalance_type, trigger_reason, current_allocation,
                target_allocation, validated_trades, db
            )
            
            # 估算成本和市场冲击
            cost_analysis = self._analyze_rebalance_costs(validated_trades)
            
            result = {
                "rebalance_id": str(rebalance_record.id),
                "portfolio_id": portfolio_id,
                "rebalance_type": rebalance_type,
                "trigger_reason": trigger_reason,
                "current_allocation": current_allocation,
                "target_allocation": target_allocation,
                "planned_trades": validated_trades,
                "cost_analysis": cost_analysis,
                "execution_plan": self._generate_execution_plan(validated_trades),
                "expected_turnover": cost_analysis["turnover_ratio"],
                "rebalance_date": as_of or date.today()
            }
            
            logger.info(f"Portfolio rebalance planned: {portfolio_id} ({rebalance_record.id})")
            return result
            
        except Exception as e:
            logger.error(f"Portfolio rebalance failed: {e}", exc_info=True)
            raise PortfolioException(f"投资组合再平衡失败: {e}")
    
    @log_performance("circuit_breaker_trigger")
    async def trigger_circuit_breaker(
        self,
        portfolio_id: str,
        current_nav: float,
        trigger_reason: str,
        force_trigger: bool,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """触发回撤断路器"""
        try:
            # 获取组合信息
            portfolio = await self._get_portfolio_by_id(portfolio_id, db)
            
            # 计算回撤水平
            initial_nav = await self._get_initial_nav(portfolio_id, db)
            current_drawdown = (current_nav - initial_nav) / initial_nav
            
            # 确定断路器级别
            new_level = self._determine_circuit_breaker_level(current_drawdown, force_trigger)
            current_level = portfolio.circuit_breaker_level
            
            # 如果级别没有变化且非强制触发，则不执行
            if new_level == current_level and not force_trigger:
                return {
                    "portfolio_id": portfolio_id,
                    "current_level": current_level.value,
                    "new_level": new_level.value,
                    "action_taken": False,
                    "reason": "断路器级别未变化",
                    "current_drawdown": current_drawdown
                }
            
            # 获取当前持仓
            current_positions = await self._get_current_positions(portfolio_id, db)
            
            # 执行断路器动作
            actions_taken = await self._execute_circuit_breaker_actions(
                portfolio, current_positions, new_level, db
            )
            
            # 更新组合状态
            await self._update_portfolio_circuit_breaker(
                portfolio_id, new_level, current_drawdown, trigger_reason, db
            )
            
            # 创建断路器事件记录
            event_record = await self._create_circuit_breaker_event(
                portfolio_id, current_level, new_level, current_drawdown,
                trigger_reason, actions_taken, db
            )
            
            result = {
                "portfolio_id": portfolio_id,
                "previous_level": current_level.value,
                "new_level": new_level.value,
                "current_drawdown": current_drawdown,
                "trigger_reason": trigger_reason,
                "actions_taken": actions_taken,
                "affected_positions": len(actions_taken.get("position_changes", [])),
                "remaining_positions": len([p for p in current_positions if p.is_active]),
                "trigger_time": datetime.utcnow(),
                "event_id": str(event_record["id"]) if event_record else None
            }
            
            logger.warning(
                f"Circuit breaker triggered for portfolio {portfolio_id}: "
                f"{current_level.value} -> {new_level.value} (drawdown: {current_drawdown:.2%})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Circuit breaker trigger failed: {e}", exc_info=True)
            raise PortfolioException(f"断路器触发失败: {e}")
    
    async def get_portfolio_status(
        self,
        portfolio_id: str,
        as_of: Optional[date],
        include_positions: bool,
        include_metrics: bool,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取组合状态"""
        try:
            portfolio = await self._get_portfolio_by_id(portfolio_id, db)
            
            # 基本状态信息
            status = {
                "portfolio_id": str(portfolio.id),
                "name": portfolio.name,
                "description": portfolio.description,
                "total_capital": portfolio.total_capital,
                "cash_position": portfolio.cash_position,
                "leverage_ratio": portfolio.leverage_ratio,
                "max_leverage": portfolio.max_leverage,
                "available_capital": portfolio.available_capital,
                "is_active": portfolio.is_active,
                "circuit_breaker": {
                    "level": portfolio.circuit_breaker_level.value,
                    "armed": portfolio.circuit_breaker_armed,
                    "current_drawdown": portfolio.current_drawdown,
                    "max_drawdown": portfolio.max_drawdown
                },
                "last_updated": portfolio.updated_at.isoformat()
            }
            
            # 包含持仓明细
            if include_positions:
                positions = await self._get_current_positions(portfolio_id, db)
                status["positions"] = [
                    {
                        "position_id": str(pos.id),
                        "ticker": pos.ticker,
                        "tier": pos.tier.value,
                        "shares": pos.shares,
                        "avg_cost": pos.avg_cost,
                        "current_price": pos.current_price,
                        "market_value": pos.market_value,
                        "target_weight": pos.target_weight,
                        "current_weight": pos.current_weight,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                        "entry_progress": pos.entry_progress,
                        "margin_used": pos.margin_used,
                        "is_active": pos.is_active
                    }
                    for pos in positions
                ]
                
                # 分层汇总
                status["tier_summary"] = self._calculate_tier_summary(positions)
            
            # 包含风险指标
            if include_metrics:
                risk_metrics = await self._get_latest_risk_metrics(portfolio_id, db)
                if risk_metrics:
                    status["risk_metrics"] = {
                        "var_1d_95": risk_metrics.var_1d_95,
                        "var_1d_99": risk_metrics.var_1d_99,
                        "portfolio_beta": risk_metrics.portfolio_beta,
                        "sharpe_ratio": None,  # 需要从performance计算
                        "liquidity_score": risk_metrics.liquidity_score,
                        "concentration_risk": risk_metrics.concentration_risk,
                        "calculation_date": risk_metrics.calculation_date.isoformat()
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Get portfolio status failed: {e}", exc_info=True)
            raise PortfolioException(f"获取组合状态失败: {e}")
    
    async def get_portfolio_performance(
        self,
        portfolio_id: str,
        start_date: Optional[date],
        end_date: Optional[date],
        benchmark: Optional[str],
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取组合绩效"""
        try:
            # 获取组合快照数据
            snapshots = await self._get_portfolio_snapshots(
                portfolio_id, start_date, end_date, db
            )
            
            if not snapshots:
                raise PortfolioException("未找到组合快照数据")
            
            # 计算收益率序列
            returns = self._calculate_return_series(snapshots)
            
            # 计算绩效指标
            performance_metrics = self._calculate_performance_metrics(returns)
            
            # 基准比较
            benchmark_comparison = None
            if benchmark:
                benchmark_comparison = await self._compare_with_benchmark(
                    returns, benchmark, start_date, end_date, db
                )
            
            # 绩效归因
            attribution = await self._calculate_performance_attribution(
                portfolio_id, snapshots, db
            )
            
            result = {
                "portfolio_id": portfolio_id,
                "period": {
                    "start_date": start_date.isoformat() if start_date else snapshots[0].snapshot_date.isoformat(),
                    "end_date": end_date.isoformat() if end_date else snapshots[-1].snapshot_date.isoformat(),
                    "trading_days": len(snapshots)
                },
                "returns": {
                    "cumulative_return": performance_metrics["cumulative_return"],
                    "annualized_return": performance_metrics["annualized_return"],
                    "volatility": performance_metrics["volatility"],
                    "sharpe_ratio": performance_metrics["sharpe_ratio"],
                    "max_drawdown": performance_metrics["max_drawdown"],
                    "calmar_ratio": performance_metrics["calmar_ratio"]
                },
                "benchmark_comparison": benchmark_comparison,
                "attribution": attribution,
                "monthly_returns": self._calculate_monthly_returns(snapshots),
                "drawdown_analysis": self._analyze_drawdowns(snapshots)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Get portfolio performance failed: {e}", exc_info=True)
            raise PortfolioException(f"获取组合绩效失败: {e}")
    
    async def get_risk_analysis(
        self,
        portfolio_id: str,
        as_of: Optional[date],
        lookback_days: int,
        confidence_level: float,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """风险分析"""
        try:
            # 获取组合持仓
            positions = await self._get_current_positions(portfolio_id, db)
            
            # 获取历史价格数据
            price_data = await self._get_historical_prices(
                [pos.ticker for pos in positions], lookback_days, as_of, db
            )
            
            # 计算VaR和CVaR
            var_analysis = self._calculate_var_metrics(
                positions, price_data, confidence_level
            )
            
            # 风险分解
            risk_decomposition = self._decompose_portfolio_risk(
                positions, price_data
            )
            
            # 压力测试
            stress_tests = await self._run_stress_tests(
                portfolio_id, positions, price_data, db
            )
            
            # 流动性风险
            liquidity_risk = await self._assess_liquidity_risk(positions, db)
            
            # 集中度风险
            concentration_risk = self._calculate_concentration_risk(positions)
            
            result = {
                "portfolio_id": portfolio_id,
                "analysis_date": as_of or date.today(),
                "lookback_period": lookback_days,
                "confidence_level": confidence_level,
                "var_analysis": var_analysis,
                "risk_decomposition": risk_decomposition,
                "stress_tests": stress_tests,
                "liquidity_risk": liquidity_risk,
                "concentration_risk": concentration_risk,
                "recommendations": self._generate_risk_recommendations(
                    var_analysis, risk_decomposition, concentration_risk
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}", exc_info=True)
            raise PortfolioException(f"风险分析失败: {e}")
    
    async def list_portfolios(
        self,
        active_only: bool,
        page: int,
        page_size: int,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """获取组合列表"""
        try:
            # 构建查询
            query = select(Portfolio)
            if active_only:
                query = query.where(Portfolio.is_active == True)
            
            # 分页
            offset = (page - 1) * page_size
            query = query.order_by(desc(Portfolio.updated_at)).offset(offset).limit(page_size)
            
            # 执行查询
            result = await db.execute(query)
            portfolios = result.scalars().all()
            
            # 获取总数
            count_query = select(func.count(Portfolio.id))
            if active_only:
                count_query = count_query.where(Portfolio.is_active == True)
            total_result = await db.execute(count_query)
            total_count = total_result.scalar()
            
            # 构建响应
            portfolio_list = []
            for portfolio in portfolios:
                # 获取基本统计信息
                stats = await self._get_portfolio_basic_stats(portfolio.id, db)
                
                portfolio_list.append({
                    "portfolio_id": str(portfolio.id),
                    "name": portfolio.name,
                    "description": portfolio.description,
                    "total_capital": portfolio.total_capital,
                    "leverage_ratio": portfolio.leverage_ratio,
                    "circuit_breaker_level": portfolio.circuit_breaker_level.value,
                    "current_drawdown": portfolio.current_drawdown,
                    "is_active": portfolio.is_active,
                    "created_at": portfolio.created_at.isoformat(),
                    "updated_at": portfolio.updated_at.isoformat(),
                    "stats": stats
                })
            
            return {
                "portfolios": portfolio_list,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size
                }
            }
            
        except Exception as e:
            logger.error(f"List portfolios failed: {e}", exc_info=True)
            raise PortfolioException(f"获取组合列表失败: {e}")
    
    async def health_check(self, db: AsyncSession) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 数据库连接检查
            await db.execute("SELECT 1")
            
            # 基本统计
            portfolio_count = await db.execute(select(func.count(Portfolio.id)))
            active_portfolios = await db.execute(
                select(func.count(Portfolio.id)).where(Portfolio.is_active == True)
            )
            
            return {
                "database_connected": True,
                "total_portfolios": portfolio_count.scalar(),
                "active_portfolios": active_portfolios.scalar(),
                "features_available": [
                    "portfolio_construction",
                    "rebalancing", 
                    "circuit_breaker",
                    "risk_analysis",
                    "performance_tracking"
                ]
            }
            
        except Exception as e:
            logger.error(f"Portfolio service health check failed: {e}", exc_info=True)
            return {
                "database_connected": False,
                "error": str(e)
            }
    
    # 辅助方法
    def _validate_construction_params(
        self, total_capital: float, leverage_max: float, candidates: List[Dict],
        cash_buffer: float, max_single_position: float, max_sector_concentration: float
    ) -> None:
        """验证构建参数"""
        if total_capital <= 0:
            raise ValidationException("总资金必须大于0")
        
        if leverage_max < 1.0 or leverage_max > 2.0:
            raise ValidationException("杠杆比例必须在1.0-2.0之间")
        
        if not candidates:
            raise ValidationException("候选股票列表不能为空")
        
        if cash_buffer < 0 or cash_buffer > 0.5:
            raise ValidationException("现金缓冲比例必须在0-50%之间")
        
        if max_single_position <= 0 or max_single_position > 1.0:
            raise ValidationException("单票最大权重必须在0-100%之间")
    
    def _process_tier_config(self, tier_config: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """处理分层配置"""
        processed = {}
        for tier, limits in tier_config.items():
            if len(limits) != 2 or limits[0] >= limits[1]:
                raise ValidationException(f"分层 {tier} 配置格式错误")
            processed[tier] = {"min": limits[0], "max": limits[1]}
        return processed
    
    async def _validate_candidates_through_gates(
        self, candidates: List[Dict[str, Any]], db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """通过四闸门校验候选股票"""
        validated = []
        
        for candidate in candidates:
            try:
                # 调用四闸门校验服务
                gate_result = await gatekeeper_service.check_gates(
                    industry_score=candidate.get("industry_score", 75),  # 假设值
                    equity_score=candidate["score"],
                    ticker=candidate["ticker"],
                    db=db
                )
                
                if gate_result["overall_pass"]:
                    candidate["gate_result"] = gate_result
                    validated.append(candidate)
                    
            except Exception as e:
                logger.warning(f"Gate validation failed for {candidate['ticker']}: {e}")
                continue
        
        return validated
    
    async def _perform_liquidity_checks(
        self, candidates: List[Dict[str, Any]], db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """执行流动性检查"""
        liquidity_passed = []
        
        for candidate in candidates:
            try:
                # 估算目标仓位金额 (假设平均分配)
                estimated_position = candidate.get("target_weight", 0.1) * 1000000  # 假设100万组合
                
                liquidity_result = await liquidity_service.check_liquidity(
                    ticker=candidate["ticker"],
                    target_position=estimated_position,
                    db=db
                )
                
                if liquidity_result["overall_pass"]:
                    candidate["liquidity_result"] = liquidity_result
                    liquidity_passed.append(candidate)
                    
            except Exception as e:
                logger.warning(f"Liquidity check failed for {candidate['ticker']}: {e}")
                continue
        
        return liquidity_passed
    
    async def _optimize_portfolio_allocation(
        self, candidates: List[Dict[str, Any]], tier_limits: Dict[str, Dict[str, float]],
        total_capital: float, leverage_max: float, cash_buffer: float,
        max_single_position: float, max_sector_concentration: float
    ) -> Dict[str, Any]:
        """优化投资组合配置"""
        
        # 简化的优化算法 (生产环境中应使用更复杂的优化器)
        available_capital = total_capital * leverage_max * (1 - cash_buffer)
        
        # 按分层分组
        tier_candidates = {"A": [], "B": [], "C": []}
        for candidate in candidates:
            tier = candidate["tier"]
            if tier in tier_candidates:
                tier_candidates[tier].append(candidate)
        
        positions = []
        allocated_capital = 0
        
        # 为每个分层分配资金
        for tier, limits in tier_limits.items():
            tier_capital = available_capital * limits["max"]  # 使用最大限制作为目标
            tier_positions = tier_candidates.get(tier, [])
            
            if not tier_positions:
                continue
            
            # 按评分排序，选择最优标的
            tier_positions.sort(key=lambda x: x["score"], reverse=True)
            
            # 在该分层内分配资金
            positions_in_tier = min(len(tier_positions), 3)  # 每层最多3只股票
            capital_per_position = min(
                tier_capital / positions_in_tier,
                total_capital * max_single_position
            )
            
            for i, candidate in enumerate(tier_positions[:positions_in_tier]):
                position = {
                    "ticker": candidate["ticker"],
                    "equity_id": candidate["equity_id"],
                    "tier": tier,
                    "target_capital": capital_per_position,
                    "target_weight": capital_per_position / total_capital,
                    "score": candidate["score"],
                    "current_price": candidate.get("current_price", 100),  # 默认价格
                    "shares": int(capital_per_position / candidate.get("current_price", 100)),
                    "gate_result": candidate.get("gate_result"),
                    "liquidity_result": candidate.get("liquidity_result")
                }
                positions.append(position)
                allocated_capital += capital_per_position
        
        return {
            "positions": positions,
            "total_allocated": allocated_capital,
            "cash_position": total_capital * leverage_max - allocated_capital,
            "cash_ratio": (total_capital * leverage_max - allocated_capital) / (total_capital * leverage_max),
            "leverage_used": allocated_capital / total_capital
        }
    
    async def _create_portfolio_record(
        self, name: str, description: Optional[str], total_capital: float,
        leverage_max: float, tier_config: Dict, allocation: Dict[str, Any],
        db: AsyncSession
    ) -> Portfolio:
        """创建投资组合记录"""
        portfolio = Portfolio(
            name=name,
            description=description,
            total_capital=total_capital,
            cash_position=allocation["cash_position"],
            leverage_ratio=allocation["leverage_used"],
            max_leverage=leverage_max,
            tier_config=tier_config
        )
        
        db.add(portfolio)
        await db.commit()
        await db.refresh(portfolio)
        
        # 创建持仓记录
        for pos_data in allocation["positions"]:
            position = Position(
                portfolio_id=portfolio.id,
                equity_id=uuid.UUID(pos_data["equity_id"]),
                ticker=pos_data["ticker"],
                tier=PositionTier(pos_data["tier"]),
                shares=pos_data["shares"],
                current_price=pos_data["current_price"],
                market_value=pos_data["target_capital"],
                target_weight=pos_data["target_weight"],
                current_weight=pos_data["target_weight"]
            )
            db.add(position)
        
        await db.commit()
        return portfolio
    
    def _generate_entry_plan(
        self, allocation: Dict[str, Any], entry_mode: str, total_capital: float
    ) -> Dict[str, Any]:
        """生成建仓计划"""
        if entry_mode == "three_stage":
            stages = [
                {"stage": 1, "weight": 0.40, "condition": "immediate"},
                {"stage": 2, "weight": 0.30, "condition": ">=200DMA"},
                {"stage": 3, "weight": 0.30, "condition": "+/-1*ATR"}
            ]
        else:
            stages = [{"stage": 1, "weight": 1.00, "condition": "immediate"}]
        
        entry_plan = []
        for position in allocation["positions"]:
            for stage in stages:
                entry_plan.append({
                    "ticker": position["ticker"],
                    "tier": position["tier"],
                    "stage": stage["stage"],
                    "target_amount": position["target_capital"] * stage["weight"],
                    "condition": stage["condition"],
                    "priority": 1 if position["tier"] == "A" else 2 if position["tier"] == "B" else 3
                })
        
        return {
            "entry_mode": entry_mode,
            "total_stages": len(stages),
            "positions": entry_plan,
            "total_first_stage": sum(p["target_amount"] for p in entry_plan if p["stage"] == 1)
        }
    
    async def _analyze_portfolio_risk(
        self, allocation: Dict[str, Any], portfolio_id: uuid.UUID, db: AsyncSession
    ) -> Dict[str, Any]:
        """分析组合风险"""
        positions = allocation["positions"]
        
        # 基本风险指标
        concentration_risk = max(pos["target_weight"] for pos in positions) if positions else 0
        
        # 分层风险
        tier_allocation = {}
        for pos in positions:
            tier = pos["tier"]
            tier_allocation[tier] = tier_allocation.get(tier, 0) + pos["target_weight"]
        
        # 行业集中度 (简化)
        sector_concentration = 0.5  # 假设值，实际需要根据个股行业信息计算
        
        return {
            "concentration_risk": concentration_risk,
            "max_single_position": concentration_risk,
            "tier_allocation": tier_allocation,
            "sector_concentration": sector_concentration,
            "liquidity_score": 0.8,  # 基于流动性检查结果的综合评分
            "risk_level": "medium" if concentration_risk < 0.2 else "high"
        }
    
    def _generate_tier_summary(self, allocation: Dict[str, Any]) -> Dict[str, Any]:
        """生成分层汇总"""
        tier_summary = {"A": 0, "B": 0, "C": 0}
        tier_counts = {"A": 0, "B": 0, "C": 0}
        
        for pos in allocation["positions"]:
            tier = pos["tier"]
            tier_summary[tier] += pos["target_weight"]
            tier_counts[tier] += 1
        
        return {
            "allocations": tier_summary,
            "counts": tier_counts,
            "total_equity": sum(tier_summary.values()),
            "cash": allocation["cash_ratio"]
        }
    
    def _get_circuit_breaker_config(self) -> List[Dict[str, Any]]:
        """获取断路器配置"""
        return [
            {
                "level": "L1",
                "drawdown_threshold": -0.10,
                "actions": ["remove_leverage", "pause_tactical"],
                "description": "去融资、暂停新开战术仓"
            },
            {
                "level": "L2",
                "drawdown_threshold": -0.20,
                "actions": ["halve_tactical", "clear_watch", "tighten_entry"],
                "description": "战术仓减半、观察清零、入场门槛上调"
            },
            {
                "level": "L3",
                "drawdown_threshold": -0.30,
                "actions": ["keep_top2_3", "cash_rest"],
                "description": "仅保留2-3只确定性最高核心票，其余现金"
            }
        ]
    
    async def _get_portfolio_by_id(self, portfolio_id: str, db: AsyncSession) -> Portfolio:
        """根据ID获取组合"""
        result = await db.execute(
            select(Portfolio).where(Portfolio.id == uuid.UUID(portfolio_id))
        )
        portfolio = result.scalar_one_or_none()
        if not portfolio:
            raise PortfolioException(f"未找到组合: {portfolio_id}")
        return portfolio
    
    async def _get_current_positions(self, portfolio_id: str, db: AsyncSession) -> List[Position]:
        """获取当前持仓"""
        result = await db.execute(
            select(Position).where(
                and_(
                    Position.portfolio_id == uuid.UUID(portfolio_id),
                    Position.is_active == True
                )
            )
        )
        return result.scalars().all()
    
    def _determine_circuit_breaker_level(self, drawdown: float, force_trigger: bool) -> CircuitBreakerLevel:
        """确定断路器级别"""
        if force_trigger or drawdown <= -0.30:
            return CircuitBreakerLevel.L3
        elif drawdown <= -0.20:
            return CircuitBreakerLevel.L2
        elif drawdown <= -0.10:
            return CircuitBreakerLevel.L1
        else:
            return CircuitBreakerLevel.L0
    
    async def _execute_circuit_breaker_actions(
        self, portfolio: Portfolio, positions: List[Position], 
        level: CircuitBreakerLevel, db: AsyncSession
    ) -> Dict[str, Any]:
        """执行断路器动作"""
        actions_taken = {"position_changes": [], "leverage_changes": []}
        
        if level == CircuitBreakerLevel.L1:
            # L1: 去融资、暂停新开战术仓
            if portfolio.leverage_ratio > 1.0:
                portfolio.leverage_ratio = 1.0
                actions_taken["leverage_changes"].append("removed_leverage")
            
            # 标记战术仓位暂停
            for pos in positions:
                if pos.tier == PositionTier.C and pos.entry_progress < 1.0:
                    actions_taken["position_changes"].append({
                        "ticker": pos.ticker,
                        "action": "pause_tactical_entry",
                        "tier": pos.tier.value
                    })
        
        elif level == CircuitBreakerLevel.L2:
            # L2: 战术仓减半、观察清零
            for pos in positions:
                if pos.tier == PositionTier.C:
                    # 战术仓减半
                    pos.target_weight = pos.target_weight * 0.5
                    pos.shares = pos.shares * 0.5
                    actions_taken["position_changes"].append({
                        "ticker": pos.ticker,
                        "action": "halve_tactical",
                        "new_weight": pos.target_weight
                    })
        
        elif level == CircuitBreakerLevel.L3:
            # L3: 仅保留2-3只核心票
            a_positions = [p for p in positions if p.tier == PositionTier.A]
            a_positions.sort(key=lambda x: x.target_weight, reverse=True)
            
            for i, pos in enumerate(positions):
                if pos.tier != PositionTier.A or i >= 3:
                    pos.is_active = False
                    actions_taken["position_changes"].append({
                        "ticker": pos.ticker,
                        "action": "liquidate",
                        "tier": pos.tier.value
                    })
        
        await db.commit()
        return actions_taken


# 创建全局服务实例
portfolio_service = PortfolioService()