# InvestIQ Main API Service

InvestIQ平台的主API服务，作为微服务架构的API网关和业务协调器。

## 🎯 功能

- **API网关**: 统一的REST API入口
- **业务协调**: 协调各个AI微服务
- **用户管理**: 认证、授权、用户数据
- **数据聚合**: 整合各服务的分析结果
- **缓存层**: Redis缓存优化

## 🚀 启动

### 开发环境
```bash
cd services/main-api
pip install -e .
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 生产环境  
```bash
docker build -t investiq-main-api .
docker run -p 8000:8000 investiq-main-api
```

## 📡 依赖服务

- PostgreSQL (数据库)
- Redis (缓存)
- llm-service (LLM推理)
- sentiment-service (情感分析)
- timeseries-service (时序预测)

## 🔗 API端点

- `GET /health` - 健康检查
- `POST /api/v1/auth/login` - 用户登录
- `GET /api/v1/analysis/{symbol}` - 股票分析
- `POST /api/v1/portfolio/optimize` - 投资组合优化