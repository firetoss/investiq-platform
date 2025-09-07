#!/bin/bash

# InvestIQ Platform 启动脚本
# 用于快速启动和验证容器化平台

set -e

echo "🚀 Starting InvestIQ Platform..."
echo "=================================="

# 检查依赖
echo "📋 Checking dependencies..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker dependencies checked"

# 创建必要的目录
echo "📁 Creating required directories..."
mkdir -p models data/postgres data/redis data/minio logs

# 检查端口占用
echo "🔍 Checking port availability..."
PORTS=(80 443 3000 5432 6379 9000 9001 8001 8002 8003 8004 8005 8006 8007 8011 8012 8013)
OCCUPIED_PORTS=()

for port in "${PORTS[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        OCCUPIED_PORTS+=($port)
    fi
done

if [ ${#OCCUPIED_PORTS[@]} -ne 0 ]; then
    echo "⚠️  Warning: The following ports are already in use:"
    printf '%s\n' "${OCCUPIED_PORTS[@]}"
    echo "   This may cause conflicts. Consider stopping conflicting services."
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Startup cancelled"
        exit 1
    fi
fi

# 检测Docker Compose命令
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# 清理旧容器（如果存在）
echo "🧹 Cleaning up old containers..."
$DOCKER_COMPOSE down --remove-orphans 2>/dev/null || true

# 构建并启动基础设施服务
echo "🏗️  Building and starting infrastructure services..."
$DOCKER_COMPOSE up -d postgres redis minio

# 等待基础服务启动
echo "⏳ Waiting for infrastructure services to be ready..."
echo "   - PostgreSQL..."

# 等待PostgreSQL
for i in {1..30}; do
    if $DOCKER_COMPOSE exec -T postgres pg_isready -U investiq -d investiq > /dev/null 2>&1; then
        echo "   ✅ PostgreSQL is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ PostgreSQL failed to start within 30 seconds"
        echo "   Checking logs..."
        $DOCKER_COMPOSE logs postgres
        exit 1
    fi
    sleep 1
done

# 等待Redis
echo "   - Redis..."
for i in {1..30}; do
    if $DOCKER_COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "   ✅ Redis is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ Redis failed to start within 30 seconds"
        $DOCKER_COMPOSE logs redis
        exit 1
    fi
    sleep 1
done

# 等待MinIO
echo "   - MinIO..."
for i in {1..30}; do
    if curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        echo "   ✅ MinIO is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "   ❌ MinIO failed to start within 30 seconds"
        $DOCKER_COMPOSE logs minio
        exit 1
    fi
    sleep 1
done

echo "🎉 Infrastructure services are ready!"

# 显示服务状态
echo ""
echo "📊 Service Status:"
echo "=================="
echo "🗄️  PostgreSQL:  http://localhost:5432"
echo "🔄 Redis:        http://localhost:6379" 
echo "📦 MinIO:        http://localhost:9000 (admin: minioadmin/minioadmin123)"
echo "   MinIO Console: http://localhost:9001"

# 显示连接信息
echo ""
echo "🔗 Connection Information:"
echo "=========================="
echo "Database: postgresql://investiq:password@localhost:5432/investiq"
echo "Redis:    redis://localhost:6379/0"
echo "MinIO:    http://localhost:9000"

echo ""
echo "✅ InvestIQ Platform infrastructure is ready!"
echo "🔧 Next steps:"
echo "   1. Verify database tables: docker-compose exec postgres psql -U investiq -d investiq -c '\\dt'"
echo "   2. Check Redis: docker-compose exec redis redis-cli info"
echo "   3. Access MinIO console: http://localhost:9001"
echo ""
echo "⏹️  To stop: ./scripts/stop.sh"
echo "🧹 To cleanup: docker-compose down -v"