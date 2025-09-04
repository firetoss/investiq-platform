# InvestIQ Platform

智能投资决策平台 - 基于AI增强的行业选择与组合支持系统

## 项目概述

InvestIQ是一个专为个人投资者设计的智能投资决策平台，集成了传统量化分析和现代AI技术，运行在NVIDIA Jetson Orin AGX设备上。

### 核心功能

- **四闸门投资决策系统**: 政策→行业→个股的智能评分体系
- **AI增强分析**: 大模型驱动的市场情报和投资建议
- **GPU加速计算**: 充分利用Jetson算力进行高性能计算
- **实时监控与告警**: 智能风险管控和投资机会识别
- **多模态分析**: 整合文本、图像、数据的综合分析能力

### 技术架构

```
┌─────────────────┐
│   Web UI        │ (React/Vue.js)
├─────────────────┤
│   FastAPI       │ (Python后端)
├─────────────────┤
│   AI Engine     │ (LLM + GPU加速)
├─────────────────┤
│   PostgreSQL    │ (时态数据库)
│   Redis         │ (缓存+队列)
│   MinIO         │ (对象存储)
└─────────────────┘
```

### 硬件要求

- **推荐**: NVIDIA Jetson Orin AGX 64GB
- **最低**: 32GB内存，支持CUDA的GPU
- **存储**: 500GB+ NVMe SSD

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd investiq-platform

# 安装依赖
pip install -r backend/requirements.txt
npm install --prefix frontend
```

### 2. 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑配置文件
vim .env
```

### 3. 启动服务

```bash
# 使用Docker Compose启动
docker-compose up -d

# 或手动启动
cd backend && python -m uvicorn app.main:app --reload
cd frontend && npm start
```

### 4. 访问应用

- Web界面: http://localhost:3000
- API文档: http://localhost:8000/docs
- 监控面板: http://localhost:3001

## 项目结构

```
investiq-platform/
├── backend/                 # Python后端
│   ├── app/
│   │   ├── api/            # API路由
│   │   ├── core/           # 核心配置
│   │   ├── models/         # 数据模型
│   │   ├── services/       # 业务逻辑
│   │   ├── utils/          # 工具函数
│   │   └── ai/             # AI模块
│   ├── migrations/         # 数据库迁移
│   ├── scripts/           # 脚本工具
│   └── tests/             # 测试用例
├── frontend/               # 前端应用
│   ├── src/
│   │   ├── components/    # React组件
│   │   ├── pages/         # 页面组件
│   │   ├── services/      # API服务
│   │   └── utils/         # 工具函数
│   ├── public/            # 静态资源
│   └── dist/              # 构建输出
├── models/                 # AI模型文件
│   ├── llm/               # 大语言模型
│   └── traditional/       # 传统ML模型
├── data/                   # 数据文件
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── models/            # 模型数据
├── deploy/                 # 部署配置
│   ├── docker/            # Docker文件
│   ├── scripts/           # 部署脚本
│   └── configs/           # 配置文件
├── docs/                   # 文档
└── tests/                  # 集成测试
```

## 核心模块

### 1. 评分引擎 (Scoring Engine)

实现四闸门投资决策方法论：

- **行业评分**: 政策强度、落地证据、市场确认、风险评估
- **个股评分**: 质量、估值、动量、政策契合度、护城河
- **四闸门校验**: 行业、公司、估值、执行四重验证

### 2. AI增强模块 (AI Enhancement)

- **大模型集成**: Llama 2/FinBERT本地部署
- **多模态分析**: 文本+图像的财报分析
- **实时情报**: AI驱动的市场信息聚合
- **强化学习**: 投资组合优化

### 3. GPU加速计算 (GPU Acceleration)

- **CUDA并行计算**: 批量评分和技术指标计算
- **TensorRT优化**: 模型推理加速
- **内存管理**: 高效的GPU内存使用

### 4. 数据管理 (Data Management)

- **时态数据库**: Point-in-Time查询支持
- **数据快照**: 历史数据版本管理
- **实时同步**: 市场数据实时更新

## 开发指南

### 代码规范

- Python: 遵循PEP 8规范
- JavaScript: 使用ESLint + Prettier
- Git: 使用Conventional Commits

### 测试策略

- 单元测试: pytest (Python) + Jest (JavaScript)
- 集成测试: 端到端API测试
- 性能测试: GPU计算性能基准测试

### 部署流程

1. 开发环境: 本地Docker开发
2. 测试环境: CI/CD自动部署
3. 生产环境: Jetson设备部署

## 许可证

MIT License

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your-email@example.com]
- 问题反馈: [GitHub Issues]

---

**注意**: 本项目专为Jetson Orin AGX设备优化，充分利用GPU算力进行智能投资分析。
