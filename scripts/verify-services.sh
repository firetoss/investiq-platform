#!/bin/bash

# InvestIQ Platform 验证脚本
# 用于验证各服务是否正常运行

echo "🔍 Verifying InvestIQ Platform Services..."
echo "=========================================="

# 检测Docker Compose命令
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 验证函数
verify_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "   Checking $service_name... "
    
    if curl -s -f "$url" > /dev/null; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

# 验证端口
verify_port() {
    local service_name=$1
    local port=$2
    
    echo -n "   Checking $service_name port $port... "
    
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✅ Open${NC}"
        return 0
    else
        echo -e "${RED}❌ Closed${NC}"
        return 1
    fi
}

# 验证数据库
verify_database() {
    echo -n "   Checking PostgreSQL connection... "
    
    if $DOCKER_COMPOSE exec -T postgres pg_isready -U investiq -d investiq > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        
        echo -n "   Checking database tables... "
        table_count=$($DOCKER_COMPOSE exec -T postgres psql -U investiq -d investiq -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs)
        
        if [ "$table_count" -gt 0 ]; then
            echo -e "${GREEN}✅ $table_count tables found${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  No tables found${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Connection failed${NC}"
        return 1
    fi
}

# 验证Redis
verify_redis() {
    echo -n "   Checking Redis connection... "
    
    if $DOCKER_COMPOSE exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        return 0
    else
        echo -e "${RED}❌ Connection failed${NC}"
        return 1
    fi
}

# 开始验证
echo "🏗️  Infrastructure Services:"
echo "=========================="

failed_checks=0

# 验证基础设施
if ! verify_port "PostgreSQL" 5432; then
    ((failed_checks++))
fi

if ! verify_port "Redis" 6379; then
    ((failed_checks++))
fi

if ! verify_port "MinIO API" 9000; then
    ((failed_checks++))
fi

if ! verify_port "MinIO Console" 9001; then
    ((failed_checks++))
fi

# 验证数据库连接
echo ""
echo "🗄️  Database Verification:"
echo "========================"

if ! verify_database; then
    ((failed_checks++))
fi

# 验证Redis连接
echo ""
echo "🔄 Redis Verification:"
echo "===================="

if ! verify_redis; then
    ((failed_checks++))
fi

# 显示容器状态
echo ""
echo "📦 Container Status:"
echo "=================="
$DOCKER_COMPOSE ps

# 验证Docker网络
echo ""
echo "🔗 Network Status:"
echo "================="
echo -n "   Checking investiq_network... "
if docker network ls | grep -q investiq_network; then
    echo -e "${GREEN}✅ Network exists${NC}"
else
    echo -e "${RED}❌ Network not found${NC}"
    ((failed_checks++))
fi

# 总结
echo ""
echo "📊 Verification Summary:"
echo "======================="

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Platform is ready.${NC}"
    echo ""
    echo "🎯 Quick Tests:"
    echo "   Database: $DOCKER_COMPOSE exec postgres psql -U investiq -d investiq -c 'SELECT version();'"
    echo "   Redis:    $DOCKER_COMPOSE exec redis redis-cli info server"
    echo "   MinIO:    curl -I http://localhost:9000/minio/health/live"
    echo ""
    echo "🌐 Web Interfaces:"
    echo "   MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
    exit 0
else
    echo -e "${RED}❌ $failed_checks checks failed!${NC}"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check logs: $DOCKER_COMPOSE logs [service-name]"
    echo "   2. Restart services: $DOCKER_COMPOSE restart"
    echo "   3. Full restart: $DOCKER_COMPOSE down && $DOCKER_COMPOSE up -d"
    echo ""
    exit 1
fi