"""
InvestIQ Platform - YAML配置管理服务
实现PRD要求的配置管理、红线参数控制和CI对比器
"""

import os
import yaml
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from deepdiff import DeepDiff
from jsonschema import validate, ValidationError

from backend.app.core.logging import get_logger

logger = get_logger(__name__)


class ConfigType(Enum):
    """配置类型枚举"""
    REDLINES = "redlines"          # 红线参数 (严格控制)
    APPLICATION = "application"     # 应用配置 (可调整)
    ENVIRONMENT = "environment"     # 环境配置 (部署相关)


class ChangeImpact(Enum):
    """变更影响级别"""
    LOW = "low"           # 低影响：日志级别、缓存TTL等
    MEDIUM = "medium"     # 中影响：业务参数调整
    HIGH = "high"         # 高影响：算法权重、阈值调整
    CRITICAL = "critical" # 关键影响：红线参数变更


@dataclass
class ConfigChange:
    """配置变更记录"""
    path: str
    old_value: Any
    new_value: Any
    change_type: str
    impact_level: ChangeImpact
    timestamp: datetime
    changed_by: str
    approval_status: Optional[str] = None
    rollback_data: Optional[Dict] = None


@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    business_rule_violations: List[str]
    impact_analysis: Dict[str, Any]


class ConfigManager:
    """配置管理器 - 核心配置管理服务"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 配置文件路径
        self.redlines_file = self.config_dir / "redlines.yaml"
        self.application_file = self.config_dir / "application.yaml" 
        self.environment_file = self.config_dir / "environment.yaml"
        
        # 变更历史
        self.change_history: List[ConfigChange] = []
        
        # 缓存
        self._config_cache: Dict[str, Dict] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # 加载配置文件
        self._load_all_configs()
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        try:
            for config_type in ConfigType:
                self._load_config(config_type)
            logger.info("All configurations loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise
    
    def _load_config(self, config_type: ConfigType, force_reload: bool = False) -> Dict:
        """加载指定类型的配置"""
        cache_key = config_type.value
        
        # 检查缓存
        if not force_reload and cache_key in self._config_cache:
            if datetime.now() - self._cache_timestamp[cache_key] < self._cache_ttl:
                return self._config_cache[cache_key]
        
        # 获取文件路径
        config_file = self._get_config_file_path(config_type)
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # 更新缓存
            self._config_cache[cache_key] = config
            self._cache_timestamp[cache_key] = datetime.now()
            
            logger.debug(f"Configuration loaded: {config_type.value}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config {config_type.value}: {e}")
            return {}
    
    def _get_config_file_path(self, config_type: ConfigType) -> Path:
        """获取配置文件路径"""
        file_mapping = {
            ConfigType.REDLINES: self.redlines_file,
            ConfigType.APPLICATION: self.application_file,
            ConfigType.ENVIRONMENT: self.environment_file
        }
        return file_mapping[config_type]
    
    def get_config(self, config_type: ConfigType, path: Optional[str] = None) -> Any:
        """获取配置值"""
        config = self._load_config(config_type)
        
        if path is None:
            return config
        
        # 支持点分路径，如 "gatekeeper.industry.threshold"
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        except Exception as e:
            logger.error(f"Failed to get config {path}: {e}")
            return None
    
    def set_config(
        self, 
        config_type: ConfigType, 
        path: str, 
        value: Any, 
        changed_by: str = "system",
        skip_validation: bool = False
    ) -> bool:
        """设置配置值"""
        try:
            # 获取当前配置
            config = self._load_config(config_type, force_reload=True)
            
            # 获取旧值
            old_value = self.get_config(config_type, path)
            
            # 验证变更
            if not skip_validation:
                validation_result = self._validate_change(
                    config_type, path, old_value, value
                )
                if not validation_result.is_valid:
                    logger.error(f"Config validation failed: {validation_result.errors}")
                    return False
            
            # 设置新值
            keys = path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
            
            # 保存配置
            if self._save_config(config_type, config):
                # 记录变更
                change = ConfigChange(
                    path=path,
                    old_value=old_value,
                    new_value=value,
                    change_type="update",
                    impact_level=self._assess_change_impact(config_type, path),
                    timestamp=datetime.now(),
                    changed_by=changed_by,
                    rollback_data={"old_value": old_value}
                )
                self.change_history.append(change)
                
                # 清除缓存
                self._clear_cache(config_type)
                
                logger.info(f"Configuration updated: {config_type.value}.{path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to set config {path}: {e}")
            return False
    
    def _save_config(self, config_type: ConfigType, config: Dict) -> bool:
        """保存配置到文件"""
        try:
            config_file = self._get_config_file_path(config_type)
            
            # 备份现有文件
            if config_file.exists():
                backup_file = config_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
                config_file.rename(backup_file)
            
            # 保存新配置
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    config, 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2
                )
            
            logger.info(f"Configuration saved: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def _validate_change(
        self, 
        config_type: ConfigType, 
        path: str, 
        old_value: Any, 
        new_value: Any
    ) -> ConfigValidationResult:
        """验证配置变更"""
        errors = []
        warnings = []
        business_violations = []
        
        try:
            # 类型验证
            if old_value is not None and type(old_value) != type(new_value):
                errors.append(f"Type mismatch: expected {type(old_value)}, got {type(new_value)}")
            
            # 红线参数特殊验证
            if config_type == ConfigType.REDLINES:
                violations = self._validate_redline_change(path, old_value, new_value)
                business_violations.extend(violations)
            
            # 业务规则验证
            business_errors = self._validate_business_rules(config_type, path, new_value)
            business_violations.extend(business_errors)
            
            # 影响分析
            impact_analysis = self._analyze_change_impact(config_type, path, old_value, new_value)
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            impact_analysis = {}
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0 and len(business_violations) == 0,
            errors=errors,
            warnings=warnings,
            business_rule_violations=business_violations,
            impact_analysis=impact_analysis
        )
    
    def _validate_redline_change(self, path: str, old_value: Any, new_value: Any) -> List[str]:
        """验证红线参数变更"""
        violations = []
        
        # 红线参数变更需要特殊审批
        if path.startswith('gatekeeper.'):
            if abs(float(new_value) - float(old_value)) > 0.1:  # 变化超过10%
                violations.append(f"红线参数 {path} 变更超过10%，需要风险委员会审批")
        
        # 权重和不能超过1.0
        if 'weight' in path and isinstance(new_value, (int, float)):
            if new_value < 0 or new_value > 1:
                violations.append(f"权重参数 {path} 必须在0-1之间")
        
        # 风险控制参数不能放宽
        if 'circuit_breaker' in path or 'risk_control' in path:
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                if new_value > old_value:  # 放宽风险控制
                    violations.append(f"风险控制参数 {path} 不能放宽，需要特殊审批")
        
        return violations
    
    def _validate_business_rules(self, config_type: ConfigType, path: str, value: Any) -> List[str]:
        """验证业务规则"""
        violations = []
        
        try:
            # 评分权重和必须为1.0
            if 'scoring_weights' in path:
                config = self._load_config(config_type)
                if 'industry' in path:
                    weights = config.get('scoring_weights', {}).get('industry', {})
                    total = sum([
                        weights.get('policy_weight', 0),
                        weights.get('execution_weight', 0),
                        weights.get('market_weight', 0),
                        weights.get('risk_weight', 0)
                    ])
                    if abs(total - 1.0) > 0.001:
                        violations.append("行业评分权重和必须等于1.0")
                
                elif 'equity' in path:
                    weights = config.get('scoring_weights', {}).get('equity', {})
                    total = sum([
                        weights.get('quality_weight', 0),
                        weights.get('valuation_weight', 0),
                        weights.get('momentum_weight', 0),
                        weights.get('catalyst_weight', 0),
                        weights.get('share_weight', 0)
                    ])
                    if abs(total - 1.0) > 0.001:
                        violations.append("个股评分权重和必须等于1.0")
            
            # 投资组合权重范围检查
            if 'portfolio.tier_weights' in path:
                if isinstance(value, dict) and 'min_weight' in value and 'max_weight' in value:
                    if value['min_weight'] >= value['max_weight']:
                        violations.append("投资组合权重最小值必须小于最大值")
            
        except Exception as e:
            violations.append(f"业务规则验证失败: {e}")
        
        return violations
    
    def _assess_change_impact(self, config_type: ConfigType, path: str) -> ChangeImpact:
        """评估变更影响级别"""
        if config_type == ConfigType.REDLINES:
            return ChangeImpact.CRITICAL
        
        # 评分权重和算法参数
        if any(keyword in path for keyword in ['weight', 'threshold', 'algorithm']):
            return ChangeImpact.HIGH
        
        # 业务逻辑参数
        if any(keyword in path for keyword in ['business', 'portfolio', 'risk']):
            return ChangeImpact.MEDIUM
        
        # 系统配置参数
        return ChangeImpact.LOW
    
    def _analyze_change_impact(
        self, 
        config_type: ConfigType, 
        path: str, 
        old_value: Any, 
        new_value: Any
    ) -> Dict[str, Any]:
        """分析变更影响"""
        analysis = {
            "config_type": config_type.value,
            "path": path,
            "change_percentage": None,
            "affected_services": [],
            "restart_required": False,
            "migration_needed": False
        }
        
        try:
            # 计算变化百分比
            if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)) and old_value != 0:
                analysis["change_percentage"] = abs((new_value - old_value) / old_value * 100)
            
            # 分析受影响的服务
            if 'scoring' in path:
                analysis["affected_services"].extend(["scoring_engine", "gatekeeper"])
            if 'portfolio' in path:
                analysis["affected_services"].extend(["portfolio_service", "risk_engine"])
            if 'alerts' in path:
                analysis["affected_services"].extend(["alert_service", "notification_service"])
            
            # 判断是否需要重启
            if config_type == ConfigType.APPLICATION and any(
                keyword in path for keyword in ['database', 'redis', 'external_services']
            ):
                analysis["restart_required"] = True
            
            # 判断是否需要数据迁移
            if 'schema' in path or 'model' in path:
                analysis["migration_needed"] = True
                
        except Exception as e:
            logger.error(f"Impact analysis failed: {e}")
        
        return analysis
    
    def _clear_cache(self, config_type: Optional[ConfigType] = None):
        """清除配置缓存"""
        if config_type:
            cache_key = config_type.value
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
                del self._cache_timestamp[cache_key]
        else:
            self._config_cache.clear()
            self._cache_timestamp.clear()


class ConfigComparator:
    """配置对比器 - 用于CI/CD环境配置对比"""
    
    def __init__(self):
        self.differ = DeepDiff
    
    def compare_configs(
        self, 
        config1: Dict, 
        config2: Dict,
        ignore_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """对比两个配置"""
        ignore_keys = ignore_keys or ['updated_at', 'version', 'metadata.updated_at']
        
        # 移除忽略的键
        clean_config1 = self._remove_ignored_keys(config1, ignore_keys)
        clean_config2 = self._remove_ignored_keys(config2, ignore_keys)
        
        # 执行深度对比
        diff = self.differ(
            clean_config1, 
            clean_config2,
            ignore_order=True,
            verbose_level=2
        )
        
        return {
            "has_changes": bool(diff),
            "summary": self._summarize_changes(diff),
            "detailed_diff": dict(diff),
            "risk_assessment": self._assess_diff_risk(diff)
        }
    
    def _remove_ignored_keys(self, config: Dict, ignore_keys: List[str]) -> Dict:
        """移除忽略的键"""
        import copy
        clean_config = copy.deepcopy(config)
        
        for key_path in ignore_keys:
            keys = key_path.split('.')
            current = clean_config
            
            try:
                for key in keys[:-1]:
                    if key in current and isinstance(current[key], dict):
                        current = current[key]
                    else:
                        break
                else:
                    if keys[-1] in current:
                        del current[keys[-1]]
            except (KeyError, TypeError):
                continue
        
        return clean_config
    
    def _summarize_changes(self, diff: Dict) -> Dict[str, Any]:
        """总结变更"""
        summary = {
            "added_items": 0,
            "removed_items": 0,
            "changed_items": 0,
            "type_changes": 0,
            "critical_changes": [],
            "business_impact_changes": []
        }
        
        # 统计各类变更
        if 'dictionary_item_added' in diff:
            summary["added_items"] = len(diff['dictionary_item_added'])
            
        if 'dictionary_item_removed' in diff:
            summary["removed_items"] = len(diff['dictionary_item_removed'])
            
        if 'values_changed' in diff:
            summary["changed_items"] = len(diff['values_changed'])
            
        if 'type_changes' in diff:
            summary["type_changes"] = len(diff['type_changes'])
        
        # 识别关键变更
        for change_type, changes in diff.items():
            if change_type in ['values_changed', 'type_changes']:
                for path, details in changes.items():
                    if self._is_critical_path(path):
                        summary["critical_changes"].append({
                            "path": path,
                            "change_type": change_type,
                            "details": details
                        })
                    elif self._is_business_impact_path(path):
                        summary["business_impact_changes"].append({
                            "path": path, 
                            "change_type": change_type,
                            "details": details
                        })
        
        return summary
    
    def _is_critical_path(self, path: str) -> bool:
        """判断是否为关键路径"""
        critical_patterns = [
            'gatekeeper',
            'risk_controls.circuit_breaker',
            'scoring_weights',
            'redlines'
        ]
        return any(pattern in path for pattern in critical_patterns)
    
    def _is_business_impact_path(self, path: str) -> bool:
        """判断是否为业务影响路径"""
        business_patterns = [
            'portfolio',
            'liquidity',
            'alerts.throttling',
            'business'
        ]
        return any(pattern in path for pattern in business_patterns)
    
    def _assess_diff_risk(self, diff: Dict) -> Dict[str, Any]:
        """评估差异风险"""
        risk_assessment = {
            "overall_risk": "LOW",
            "risk_factors": [],
            "recommendations": []
        }
        
        # 评估风险因素
        if 'values_changed' in diff:
            for path, details in diff['values_changed'].items():
                if self._is_critical_path(path):
                    risk_assessment["risk_factors"].append(f"关键参数变更: {path}")
                    risk_assessment["overall_risk"] = "CRITICAL"
                elif self._is_business_impact_path(path):
                    risk_assessment["risk_factors"].append(f"业务参数变更: {path}")
                    if risk_assessment["overall_risk"] == "LOW":
                        risk_assessment["overall_risk"] = "MEDIUM"
        
        # 生成建议
        if risk_assessment["overall_risk"] == "CRITICAL":
            risk_assessment["recommendations"].extend([
                "需要风险委员会审批",
                "建议在非交易时段部署",
                "准备回滚方案",
                "进行全面测试"
            ])
        elif risk_assessment["overall_risk"] == "MEDIUM":
            risk_assessment["recommendations"].extend([
                "需要业务团队确认",
                "建议进行影响评估",
                "准备回滚方案"
            ])
        
        return risk_assessment
    
    def compare_environments(
        self, 
        env1_config_dir: str, 
        env2_config_dir: str
    ) -> Dict[str, Any]:
        """对比不同环境的配置"""
        comparison_results = {}
        
        for config_type in ConfigType:
            env1_file = Path(env1_config_dir) / f"{config_type.value}.yaml"
            env2_file = Path(env2_config_dir) / f"{config_type.value}.yaml"
            
            if env1_file.exists() and env2_file.exists():
                with open(env1_file, 'r') as f1, open(env2_file, 'r') as f2:
                    config1 = yaml.safe_load(f1)
                    config2 = yaml.safe_load(f2)
                
                comparison_results[config_type.value] = self.compare_configs(config1, config2)
            else:
                comparison_results[config_type.value] = {
                    "error": f"Configuration files not found: {env1_file} or {env2_file}"
                }
        
        return comparison_results


# 全局配置管理器实例
config_manager = ConfigManager()
config_comparator = ConfigComparator()


# 便捷函数
def get_redline_config(path: Optional[str] = None) -> Any:
    """获取红线配置"""
    return config_manager.get_config(ConfigType.REDLINES, path)


def get_app_config(path: Optional[str] = None) -> Any:
    """获取应用配置"""  
    return config_manager.get_config(ConfigType.APPLICATION, path)


def set_app_config(path: str, value: Any, changed_by: str = "system") -> bool:
    """设置应用配置"""
    return config_manager.set_config(ConfigType.APPLICATION, path, value, changed_by)


def compare_configs_between_environments(env1_dir: str, env2_dir: str) -> Dict[str, Any]:
    """对比不同环境的配置"""
    return config_comparator.compare_environments(env1_dir, env2_dir)