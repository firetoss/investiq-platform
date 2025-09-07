#!/bin/bash

# InvestIQ Platform 停止脚本

echo "🛑 Stopping InvestIQ Platform..."
echo "================================"

# 停止所有服务
echo "📦 Stopping containers..."
docker-compose down --remove-orphans

# 显示清理选项
echo ""
echo "🧹 Cleanup Options:"
echo "   To remove volumes (⚠️  data will be lost):"
echo "   docker-compose down -v"
echo ""
echo "   To remove images:"
echo "   docker-compose down --rmi all"
echo ""
echo "   To remove everything:"
echo "   docker-compose down -v --rmi all --remove-orphans"

echo ""
echo "✅ InvestIQ Platform stopped successfully!"