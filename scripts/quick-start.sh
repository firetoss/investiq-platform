#!/bin/bash

# InvestIQ Platform - 快速启动脚本
# 用于快速启动和测试系统

set -e

echo "🚀 InvestIQ Platform 快速启动脚本"
echo "=================================="

# 检查必要工具
check_requirements() {
    echo "📋 检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查uv
    if ! command -v uv &> /dev/null; then
        echo "⚠️  uv未安装，正在安装..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    
    echo "✅ 系统要求检查完成"
}

# 创建必要目录
create_directories() {
    echo "📁 创建必要目录..."
    
    mkdir -p logs
    mkdir -p data/{raw,processed,models}
    mkdir -p models/{llm,traditional}
    mkdir -p tmp
    
    echo "✅ 目录创建完成"
}

# 检查环境文件
check_env_file() {
    echo "🔧 检查环境配置..."
    
    if [ ! -f .env ]; then
        echo "⚠️  .env文件不存在，从模板复制..."
        cp .env.example .env
        echo "✅ 环境文件已创建，请根据需要修改配置"
    else
        echo "✅ 环境文件已存在"
    fi
}

# 安装Python依赖
install_dependencies() {
    echo "📦 安装Python依赖..."
    
    # 检查是否在虚拟环境中
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "✅ 已在虚拟环境中: $VIRTUAL_ENV"
    else
        echo "⚠️  未检测到虚拟环境，uv将自动管理"
    fi
    
    # 安装依赖
    uv sync --dev
    
    echo "✅ 依赖安装完成"
}

# 启动Docker服务
start_docker_services() {
    echo "🐳 启动Docker服务..."
    
    # 停止可能存在的旧服务
    docker-compose down 2>/dev/null || true
    
    # 构建镜像
    echo "🔨 构建Docker镜像..."
    docker-compose build
    
    # 启动服务
    echo "🚀 启动服务..."
    docker-compose up -d
    
    echo "⏳ 等待服务启动..."
    sleep 15
    
    # 检查服务状态
    echo "📊 服务状态:"
    docker-compose ps
}

# 健康检查
health_check() {
    echo "🏥 执行健康检查..."
    
    max_retries=30
    retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ API服务健康检查通过"
            break
        else
            echo "⏳ 等待API服务启动... ($((retry_count + 1))/$max_retries)"
            sleep 2
            retry_count=$((retry_count + 1))
        fi
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "❌ API服务启动超时"
        echo "📋 查看日志:"
        docker-compose logs api
        exit 1
    fi
    
    # 详细健康检查
    echo "📊 详细健康状态:"
    curl -s http://localhost:8000/health | python3 -m json.tool || echo "健康检查API暂时不可用"
}

# 运行API测试
run_api_tests() {
    echo "🧪 运行API测试..."
    
    # 测试评分API
    echo "测试行业评分API..."
    curl -s -X POST "http://localhost:8000/api/v1/scoring/industry" \
        -H "Content-Type: application/json" \
        -d '{
            "industry_id": "semiconductor",
            "score_p": 85.0,
            "score_e": 80.0,
            "score_m": 75.0,
            "score_r_neg": 15.0
        }' | python3 -m json.tool || echo "评分API测试失败"
    
    # 测试四闸门API
    echo "测试四闸门校验API..."
    curl -s -X POST "http://localhost:8000/api/v1/gatekeeper/check" \
        -H "Content-Type: application/json" \
        -d '{
            "industry_score": 75.0,
            "equity_score": 72.0,
            "valuation_percentile": 0.6,
            "above_200dma": true
        }' | python3 -m json.tool || echo "四闸门API测试失败"
    
    echo "✅ API测试完成"
}

# 显示访问信息
show_access_info() {
    echo ""
    echo "🎉 InvestIQ Platform 启动成功!"
    echo "================================"
    echo ""
    echo "📱 访问地址:"
    echo "  🌐 API文档:     http://localhost:8000/docs"
    echo "  🏥 健康检查:    http://localhost:8000/health"
    echo "  ℹ️  系统信息:    http://localhost:8000/info"
    echo "  📊 Grafana:     http://localhost:3001 (admin/admin123)"
    echo "  💾 MinIO:       http://localhost:9001 (investiq/investiq123)"
    echo "  🌸 Flower:      http://localhost:5555"
    echo ""
    echo "🛠️  常用命令:"
    echo "  查看日志:       make logs"
    echo "  停止服务:       make docker-down"
    echo "  重启服务:       make restart"
    echo "  GPU检查:        make gpu-check"
    echo "  性能测试:       make benchmark"
    echo ""
    echo "📚 开发指南:"
    echo "  1. 修改代码后，容器会自动重载"
    echo "  2. 数据库和Redis数据会持久化"
    echo "  3. 日志文件保存在 ./logs/ 目录"
    echo "  4. 模型文件保存在 ./models/ 目录"
    echo ""
}

# 主函数
main() {
    echo "开始时间: $(date)"
    
    check_requirements
    create_directories
    check_env_file
    install_dependencies
    start_docker_services
    health_check
    run_api_tests
    show_access_info
    
    echo "完成时间: $(date)"
    echo ""
    echo "🎯 InvestIQ Platform 已准备就绪!"
}

# 错误处理
trap 'echo "❌ 启动过程中发生错误，请检查日志"; docker-compose logs' ERR

# 执行主函数
main "$@"
