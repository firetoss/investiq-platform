#!/bin/bash
# InvestIQ Platform - 快速启动脚本
# 一键启动开发或生产环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
InvestIQ Platform 快速启动脚本

使用方法:
    $0 [命令] [选项]

命令:
    dev         启动开发环境 (默认)
    prod        启动生产环境 (Jetson)
    stop        停止所有服务
    clean       清理容器和卷
    logs        查看服务日志
    health      检查服务健康状态
    update      更新并重启服务

选项:
    -h, --help  显示此帮助信息
    -v          详细输出
    --no-gpu    禁用GPU支持 (仅开发环境)

示例:
    $0 dev              # 启动开发环境
    $0 prod             # 启动生产环境
    $0 logs frontend-dev # 查看前端开发服务日志
    $0 health           # 检查所有服务状态
    $0 clean            # 清理所有容器和数据
EOF
}

# 检查Docker是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker 未运行或无权限访问"
        log_info "请确保Docker已启动并且当前用户有权限使用Docker"
        exit 1
    fi
}

# 检查GPU支持
check_gpu() {
    if command -v nvidia-docker > /dev/null 2>&1 || docker info | grep -q nvidia; then
        log_success "检测到NVIDIA GPU支持"
        return 0
    else
        log_warning "未检测到NVIDIA GPU支持，将使用CPU模式"
        return 1
    fi
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    mkdir -p models data logs
    log_success "目录创建完成"
}

# 检查并下载AI模型
check_models() {
    log_info "检查AI模型..."
    
    if [ ! -f "models/Qwen3-8B-INT8.gguf" ]; then
        log_warning "Qwen3-8B模型未找到"
        read -p "是否下载模型文件? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "下载Qwen3-8B模型 (约6GB)..."
            wget -O models/Qwen3-8B-INT8.gguf \
                https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-INT8.gguf
            log_success "模型下载完成"
        else
            log_warning "跳过模型下载，LLM服务可能无法正常工作"
        fi
    else
        log_success "AI模型检查通过"
    fi
}

# 启动开发环境
start_dev() {
    local no_gpu=${1:-false}
    
    log_info "启动开发环境..."
    
    create_directories
    
    local compose_file="docker-compose.dev.yml"
    local gpu_flag=""
    
    if [ "$no_gpu" = false ] && check_gpu; then
        gpu_flag="--runtime=nvidia"
    else
        log_warning "在CPU模式下启动"
        # 临时创建无GPU版本的compose文件
        sed 's/runtime: nvidia/#runtime: nvidia/' "$compose_file" > "${compose_file}.nogpu"
        compose_file="${compose_file}.nogpu"
    fi
    
    # 启动服务
    docker-compose -f "$compose_file" up -d
    
    # 清理临时文件
    [ -f "${compose_file}.nogpu" ] && rm "${compose_file}.nogpu"
    
    log_success "开发环境启动完成!"
    log_info "服务访问地址:"
    echo "  - 前端开发服务: http://localhost:3000"
    echo "  - 后端API服务: http://localhost:8000"
    echo "  - API文档: http://localhost:8000/docs"
    echo "  - MinIO控制台: http://localhost:9001"
    echo ""
    log_info "使用 '$0 logs' 查看服务日志"
    log_info "使用 '$0 health' 检查服务状态"
}

# 启动生产环境
start_prod() {
    log_info "启动生产环境 (Jetson)..."
    
    create_directories
    check_models
    
    if ! check_gpu; then
        log_error "生产环境需要NVIDIA GPU支持"
        exit 1
    fi
    
    docker-compose -f docker-compose.jetson.yml up -d
    
    log_success "生产环境启动完成!"
    log_info "服务访问地址:"
    echo "  - 主应用: http://localhost:8000"
    echo "  - LLM服务: http://localhost:8001"
    echo "  - 情感分析: http://localhost:8002" 
    echo "  - 时序预测: http://localhost:8004"
    echo "  - Grafana监控: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
}

# 停止服务
stop_services() {
    log_info "停止所有服务..."
    
    # 停止开发环境
    if [ -f "docker-compose.dev.yml" ]; then
        docker-compose -f docker-compose.dev.yml down
    fi
    
    # 停止生产环境
    if [ -f "docker-compose.jetson.yml" ]; then
        docker-compose -f docker-compose.jetson.yml down
    fi
    
    log_success "所有服务已停止"
}

# 清理容器和数据
clean_all() {
    log_warning "这将删除所有容器、网络和数据卷!"
    read -p "确认继续? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "清理容器和数据..."
        
        # 停止并删除容器
        docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
        docker-compose -f docker-compose.jetson.yml down -v 2>/dev/null || true
        
        # 删除相关镜像
        docker images | grep investiq | awk '{print $3}' | xargs -r docker rmi -f
        
        # 清理未使用的资源
        docker system prune -f
        
        log_success "清理完成"
    else
        log_info "清理操作已取消"
    fi
}

# 查看日志
show_logs() {
    local service=${1:-}
    
    if [ -z "$service" ]; then
        log_info "可用服务:"
        echo "开发环境: investiq-dev, frontend-dev, postgres-dev, redis-dev"
        echo "生产环境: investiq-app, llm-service, sentiment-service, cpu-timeseries"
        read -p "请输入服务名称: " service
    fi
    
    # 尝试开发环境
    if docker-compose -f docker-compose.dev.yml ps | grep -q "$service"; then
        docker-compose -f docker-compose.dev.yml logs -f "$service"
    # 尝试生产环境
    elif docker-compose -f docker-compose.jetson.yml ps | grep -q "$service"; then
        docker-compose -f docker-compose.jetson.yml logs -f "$service"
    else
        log_error "服务 '$service' 未找到"
        exit 1
    fi
}

# 健康检查
health_check() {
    log_info "检查服务健康状态..."
    
    # 检查开发环境
    if docker-compose -f docker-compose.dev.yml ps 2>/dev/null | grep -q "Up"; then
        log_info "开发环境服务状态:"
        docker-compose -f docker-compose.dev.yml ps
        echo ""
    fi
    
    # 检查生产环境
    if docker-compose -f docker-compose.jetson.yml ps 2>/dev/null | grep -q "Up"; then
        log_info "生产环境服务状态:"
        docker-compose -f docker-compose.jetson.yml ps
        echo ""
    fi
    
    # 检查API健康状态
    if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "主应用 API 健康检查通过"
    else
        log_warning "主应用 API 健康检查失败"
    fi
    
    if curl -f -s http://localhost:3000 > /dev/null 2>&1; then
        log_success "前端服务健康检查通过"
    else
        log_warning "前端服务健康检查失败"
    fi
}

# 更新服务
update_services() {
    log_info "更新服务..."
    
    # 拉取最新代码
    git pull
    
    # 重新构建镜像
    docker-compose -f docker-compose.dev.yml build --no-cache
    docker-compose -f docker-compose.jetson.yml build --no-cache
    
    log_success "服务更新完成，请重新启动服务"
}

# 主函数
main() {
    local command=${1:-dev}
    local verbose=false
    local no_gpu=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v)
                verbose=true
                set -x
                shift
                ;;
            --no-gpu)
                no_gpu=true
                shift
                ;;
            dev|prod|stop|clean|logs|health|update)
                command=$1
                shift
                ;;
            *)
                if [ "$command" = "logs" ]; then
                    service_name=$1
                    shift
                else
                    log_error "未知选项: $1"
                    show_help
                    exit 1
                fi
                ;;
        esac
    done
    
    # 检查Docker
    check_docker
    
    # 执行命令
    case $command in
        dev)
            start_dev "$no_gpu"
            ;;
        prod)
            start_prod
            ;;
        stop)
            stop_services
            ;;
        clean)
            clean_all
            ;;
        logs)
            show_logs "$service_name"
            ;;
        health)
            health_check
            ;;
        update)
            update_services
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi