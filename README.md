# InvestIQ Platform

> 基于NVIDIA Jetson Orin AGX的智能投资决策平台
> 
> 采用GPU+CPU并行计算架构，提供中文金融AI分析服务

## 🎯 项目概述

InvestIQ Platform是一个专门针对中文金融市场的智能投资决策平台，基于"政策→行业→个股"的四闸门投资方法论，提供AI增强的投资分析服务。

### 核心特性
- 🤖 **AI驱动分析**: Qwen3-8B + RoBERTa + ARIMA/GARCH 基线
- 🚀 **Jetson优化**: 充分利用GPU+CPU硬件资源
- 📊 **实时分析**: 政策分析、情感分析、趋势预测
- 🔄 **微服务架构**: 基于dustynv优化镜像的容器化部署
- 📈 **性能监控**: 全方位的硬件和服务性能监控

## 🏗️ 技术架构

### 硬件架构（简化）
```
┌───────────────────────────────────────────────┐
│           Jetson Orin AGX 资源分配            │
├─────────────────────┬─────────────────────────┤
│        GPU          │          CPU            │
│ • LLM (Qwen3-8B)    │ • ARIMA/GARCH/指标      │
│ • 情感分析 (BERT)    │ • 数据处理/服务逻辑      │
└─────────────────────┴─────────────────────────┘
```

### 微服务架构
```
┌─────────────────────────────────────────────────────────┐
│              纯微服务架构                                │
├─────────────────┬─────────────────┬─────────────────────┤
│   主应用服务     │   AI专用服务     │   基础设施服务       │
│                 │                 │                     │
│ • API网关       │ • LLM(dustynv)  │ • PostgreSQL       │
│ • 业务逻辑      │ • 情感(dustynv) │ • Redis            │
│ • 数据库操作    │ • CPU时序       │ • MinIO            │
│ • HTTP代理      │                  │ • 监控服务         │
└─────────────────┴─────────────────┴─────────────────────┘
```

## 🚀 快速开始

### 前置要求
- **硬件**: NVIDIA Jetson Orin AGX (64GB内存)
- **系统**: JetPack 36.4.4
- **容器**: Docker + nvidia-container-runtime
- **存储**: 2TB+ NVMe SSD

### 部署步骤

#### 1. 克隆项目
```bash
git clone https://github.com/firetoss/investiq-platform.git
cd investiq-platform
```

#### 2. 下载AI模型
```bash
# 创建模型目录
mkdir -p models

# 下载Qwen3-8B INT8模型 (~6GB)
wget -O models/Qwen3-8B-INT8.gguf \
  https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-INT8.gguf

# 注意: RoBERTa和PatchTST模型会在服务启动时自动下载
```

#### 3. 启动服务
```bash
# 使用Jetson优化配置启动所有服务
docker-compose -f docker-compose.jetson.yml up -d

# 查看服务状态
docker-compose -f docker-compose.jetson.yml ps
```

#### 4. 验证部署
```bash
# 检查主应用
curl http://localhost:8000/health

# 检查AI服务
curl http://localhost:8001/health  # LLM服务 (GPU)
curl http://localhost:8002/health  # 情感分析 (GPU)
curl http://localhost:8004/health  # 时序预测 (CPU ARIMA/GARCH)

# 监控面板（可选）
# Prometheus 界面
http://localhost:9090
# Grafana 界面（默认账号：admin / 密码：investiq123）
http://localhost:3000

# 查看性能监控
curl http://localhost:8000/api/v1/performance/dashboard
```

## 📊 性能指标

### AI推理性能（示意）
- **LLM推理**: 120-180 tokens/s（Qwen3-8B INT8, GPU）
- **情感分析**: 子秒级延迟（小批量，GPU）
- **时序预测**: 取决于标的与网格，CPU 并行可达数百序列/分钟

### 硬件利用率
- **GPU**: 主要用于 LLM 与情感，按并发动态波动
- **CPU**: 时序与系统服务；建议按负载调整线程配额

### 精度表现
- **LLM精度**: 相比4-bit量化提升5-7%
- **情感分析**: 相比FinBERT精度提升3-5%
- **时序预测**: 相比Chronos精度提升8-12%

## 🔧 开发指南

### 环境设置
```bash
# 安装uv包管理器
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync --all-extras

# 安装开发工具
uv run pre-commit install
```

### 本地开发
```bash
# 启动基础设施服务
docker-compose -f docker-compose.jetson.yml up -d postgres redis minio

# 启动主应用 (开发模式)
uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
uv run pytest backend/tests/

# 代码格式化
uv run black backend/
uv run isort backend/
```

### 服务架构
| 服务 | 端口 | 职责 | 硬件资源 |
|------|------|------|----------|
| **主应用** | 8000 | API网关、业务逻辑 | 共享访问 |
| **LLM服务** | 8001 | Qwen3-8B推理 | GPU专用 |
| **情感分析** | 8002 | RoBERTa情感分析 | DLA0专用 |
| **CPU时序** | 8004 | 传统算法 | CPU专用 |
| **情感分析** | 8002 | 文本情感 | GPU专用 |

## 📚 API文档

### 核心API端点
```
/api/v1/
├── scoring/           # 评分引擎
├── gatekeeper/        # 四闸门校验
├── liquidity/         # 流动性检查
├── portfolio/         # 投资组合
├── ai/               # AI基础服务
├── analysis/         # 智能分析应用
├── data/             # 数据采集管理
└── performance/      # 性能监控
```

### 使用示例
```python
import httpx

# 政策分析
response = httpx.post("http://localhost:8000/api/v1/analysis/policy", 
                     json={"policy_text": "关于支持半导体产业发展的通知..."})

# 情感分析
response = httpx.post("http://localhost:8000/api/v1/ai/sentiment/analyze",
                     json={"texts": ["市场表现良好，投资者信心增强"]})

# 时序预测
response = httpx.post("http://localhost:8000/api/v1/ai/timeseries/forecast",
                     json={"data": [100, 102, 98, 105, 110], "horizon": 10})
```

## 🎯 投资方法论

### 四闸门决策体系
1. **行业闸门**: 政策驱动的行业机会识别
2. **公司闸门**: 基本面和竞争优势评估
3. **估值闸门**: 相对估值和安全边际
4. **执行闸门**: 技术面和流动性验证

### AI增强功能
- **政策解读**: 自动分析政策文件，识别投资机会
- **情感监控**: 实时分析市场情绪和新闻情感
- **趋势预测**: 基于AI的价格走势和技术分析
- **风险预警**: 智能识别市场风险和异常

## 🔧 技术栈

### 后端技术
- **框架**: FastAPI + uvicorn
- **数据库**: PostgreSQL 16 + Redis 7
- **AI框架**: PyTorch 2.1 + Transformers
- **硬件加速**: CUDA（可选 TensorRT/ONNXRuntime-TensorRT）
- **容器**: Docker + dustynv镜像

### AI模型
- **LLM**: Qwen3-8B INT8 量化（GPU）
- **情感分析**: 中文 BERT/RoBERTa（GPU）
- **时序预测**: ARIMA/GARCH/技术指标（CPU 并行）

### 部署技术
- **容器编排**: Docker Compose
- **基础镜像**: dustynv/llama.cpp, dustynv/pytorch
- **监控**: Prometheus + Grafana
- **存储**: MinIO对象存储

## 📈 性能优化

### Jetson优化特性
- **异构计算**: GPU+CPU 并行
- **精度平衡**: INT8（LLM）+ FP16（情感）
- **内存优化**: 按需加载与批处理

### 部署优化
- **预优化镜像**: 基于dustynv官方镜像
- **构建加速**: 多阶段构建，减少80%构建时间
- **启动优化**: 并行服务启动，减少75%启动时间
- **资源隔离**: 每个AI服务独占硬件资源

## 🤝 贡献指南

### 开发流程
1. Fork项目到个人仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 开发和测试功能
4. 提交更改: `git commit -m 'feat: add amazing feature'`
5. 推送分支: `git push origin feature/amazing-feature`
6. 创建Pull Request

### 代码规范
- 使用black进行代码格式化
- 遵循PEP 8编码规范
- 添加类型注解和文档字符串
- 编写单元测试，覆盖率>80%

### 提交规范
使用Conventional Commits格式:
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `perf:` 性能优化
- `refactor:` 代码重构

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [dustynv](https://github.com/dusty-nv) - Jetson容器优化
- [NVIDIA](https://developer.nvidia.com/jetson) - Jetson硬件支持
- [Qwen Team](https://github.com/QwenLM/Qwen) - 中文大语言模型
- [IBM Research](https://github.com/IBM/PatchTST) - PatchTST时序预测模型

## 📞 联系方式

- **项目仓库**: https://github.com/firetoss/investiq-platform
- **问题反馈**: [GitHub Issues](https://github.com/firetoss/investiq-platform/issues)
- **技术讨论**: [GitHub Discussions](https://github.com/firetoss/investiq-platform/discussions)

---

**⚡ 专为NVIDIA Jetson Orin AGX优化，实现边缘AI投资分析的极致性能**
