"""
InvestIQ Platform - API包
API路由和端点定义
"""

# API版本信息
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# API标签定义
API_TAGS = {
    "scoring": "评分引擎",
    "gatekeeper": "四闸门校验", 
    "liquidity": "流动性检查",
    "portfolio": "投资组合",
    "alerts": "告警管理",
    "evidence": "证据管理",
    "system": "系统管理",
    "ai": "AI智能分析",
    "analysis": "智能分析应用",
    "data": "数据采集管理",
    "performance": "性能监控"
}

# API描述
API_DESCRIPTIONS = {
    "scoring": "行业和个股评分计算服务",
    "gatekeeper": "投资决策四闸门校验服务",
    "liquidity": "流动性和容量检查服务", 
    "portfolio": "投资组合构建和管理服务",
    "alerts": "告警和事件管理服务",
    "evidence": "证据链和审计服务",
    "system": "系统健康检查和监控服务",
    "ai": "AI大模型推理、情感分析、时间序列预测等智能分析服务",
    "analysis": "政策分析、公司分析、市场情感分析、趋势预测等应用级智能分析服务",
    "data": "多源数据采集、数据源管理、数据缓存和文件操作服务",
    "performance": "系统性能监控、模型性能统计、硬件利用率分析和优化建议服务"
}
