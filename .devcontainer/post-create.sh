#!/bin/bash
# InvestIQ开发容器初始化脚本
# 在Dev Container创建后执行，设置开发环境

set -e

echo "🚀 初始化InvestIQ开发环境..."

# 设置工作目录
cd /app

# 安装Python依赖
echo "📦 安装Python依赖..."
uv sync --all-extras

# 安装开发工具
echo "🔧 安装开发工具..."
uv run pre-commit install

# 设置Git配置 (如果需要)
echo "⚙️ 配置Git..."
git config --global --add safe.directory /app
git config --global init.defaultBranch main

# 创建必要的目录
echo "📁 创建开发目录..."
mkdir -p logs tmp data/cache models

# 设置权限
chmod +x scripts/*.sh 2>/dev/null || echo "脚本目录不存在，跳过权限设置"

# 验证Python环境
echo "🐍 验证Python环境..."
python --version
uv --version

# 验证关键依赖
echo "📚 验证关键依赖..."
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')" || echo "❌ FastAPI未安装"
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch未安装"
python -c "import httpx; print(f'✅ HTTPX可用')" || echo "❌ HTTPX未安装"

# 创建开发配置文件
echo "📝 创建开发配置..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ 创建了.env配置文件"
fi

# 运行代码质量检查
echo "🔍 运行代码质量检查..."
uv run black --check backend/ || echo "⚠️ 代码格式需要调整"
uv run isort --check-only backend/ || echo "⚠️ 导入排序需要调整"

# 运行基础测试
echo "🧪 运行基础测试..."
uv run pytest backend/tests/ --tb=short -q || echo "⚠️ 部分测试失败，请检查"

# 显示开发环境信息
echo ""
echo "🎉 InvestIQ开发环境初始化完成！"
echo ""
echo "📋 开发环境信息:"
echo "  - Python: $(python --version)"
echo "  - 工作目录: /app"
echo "  - 主应用端口: 8000"
echo "  - 调试端口: 5678"
echo "  - 数据库端口: 5433"
echo "  - Redis端口: 6380"
echo ""
echo "🚀 快速开始:"
echo "  1. 启动主应用: uv run uvicorn backend.app.main:app --reload --host 0.0.0.0"
echo "  2. 运行测试: uv run pytest"
echo "  3. 格式化代码: uv run black backend/"
echo "  4. API文档: http://localhost:8000/docs"
echo ""
echo "📚 有用的命令:"
echo "  - 进入开发工具容器: docker-compose -f docker-compose.dev.yml exec dev-tools bash"
echo "  - 查看服务状态: docker-compose -f docker-compose.dev.yml ps"
echo "  - 查看日志: docker-compose -f docker-compose.dev.yml logs -f"
echo ""
