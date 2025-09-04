#!/bin/bash
# InvestIQ开发环境快速设置脚本
# 支持本地开发和容器开发两种模式

set -e

echo "🚀 InvestIQ Platform 开发环境设置"
echo "=================================="

# 检查当前目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 显示菜单
echo ""
echo "请选择开发模式:"
echo "1) Docker容器开发 (推荐)"
echo "2) 本地开发"
echo "3) VSCode Dev Container"
echo "4) 仅启动基础设施服务"
echo ""
read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "🐳 启动Docker容器开发环境..."
        
        # 检查Docker
        if ! command -v docker &> /dev/null; then
            echo "❌ Docker未安装，请先安装Docker"
            exit 1
        fi
        
        # 启动开发服务
        echo "启动开发服务..."
        docker-compose -f docker-compose.dev.yml up -d
        
        # 等待服务启动
        echo "等待服务启动..."
        sleep 15
        
        # 验证服务
        echo "验证开发环境..."
        curl -f http://localhost:8000/health 2>/dev/null && echo "✅ 主应用已就绪" || echo "⚠️ 主应用未就绪"
        
        echo ""
        echo "🎉 Docker开发环境已启动！"
        echo "📝 访问 http://localhost:8000/docs 查看API文档"
        echo "🔧 进入开发容器: docker-compose -f docker-compose.dev.yml exec investiq-dev bash"
        ;;
        
    2)
        echo "💻 设置本地开发环境..."
        
        # 检查uv
        if ! command -v uv &> /dev/null; then
            echo "安装uv包管理器..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
        fi
        
        # 安装依赖
        echo "安装Python依赖..."
        uv sync --all-extras
        
        # 安装开发工具
        echo "安装开发工具..."
        uv run pre-commit install
        
        # 启动基础设施服务
        echo "启动基础设施服务..."
        docker-compose -f docker-compose.dev.yml up -d postgres-dev redis-dev minio-dev
        
        # 创建环境配置
        if [ ! -f ".env" ]; then
            cp .env.example .env
            echo "✅ 创建了.env配置文件"
        fi
        
        echo ""
        echo "🎉 本地开发环境已设置！"
        echo "🚀 启动主应用: uv run uvicorn backend.app.main:app --reload"
        echo "🧪 运行测试: uv run pytest"
        ;;
        
    3)
        echo "📦 准备VSCode Dev Container..."
        
        if [ ! -f ".devcontainer/devcontainer.json" ]; then
            echo "❌ Dev Container配置文件不存在"
            exit 1
        fi
        
        echo "✅ Dev Container配置已就绪"
        echo ""
        echo "📝 在VSCode中:"
        echo "1. 安装 'Dev Containers' 扩展"
        echo "2. 打开命令面板 (Ctrl+Shift+P)"
        echo "3. 选择 'Dev Containers: Reopen in Container'"
        echo "4. 等待容器构建和初始化完成"
        ;;
        
    4)
        echo "🏗️ 仅启动基础设施服务..."
        
        docker-compose -f docker-compose.dev.yml up -d postgres-dev redis-dev minio-dev
        
        echo ""
        echo "✅ 基础设施服务已启动："
        echo "  - PostgreSQL: localhost:5433"
        echo "  - Redis: localhost:6380"
        echo "  - MinIO: localhost:9002"
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "📚 有用的命令:"
echo "  - 查看服务状态: docker-compose -f docker-compose.dev.yml ps"
echo "  - 查看日志: docker-compose -f docker-compose.dev.yml logs -f"
echo "  - 停止服务: docker-compose -f docker-compose.dev.yml down"
echo "  - 重启服务: docker-compose -f docker-compose.dev.yml restart"
echo ""
echo "🔧 开发工具:"
echo "  - 代码格式化: uv run black backend/"
echo "  - 导入排序: uv run isort backend/"
echo "  - 类型检查: uv run mypy backend/"
echo "  - 运行测试: uv run pytest backend/tests/"
echo ""
echo "📖 更多信息请查看 CONTRIBUTING.md"
