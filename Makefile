# InvestIQ Platform - Makefile
# 简化开发和部署流程

.PHONY: help install dev build test clean docker-build docker-up docker-down logs

# 默认目标
help:
	@echo "InvestIQ Platform - 可用命令:"
	@echo ""
	@echo "开发环境:"
	@echo "  install     - 安装依赖 (使用uv)"
	@echo "  dev         - 启动开发服务器"
	@echo "  test        - 运行测试"
	@echo "  lint        - 代码检查"
	@echo "  format      - 代码格式化"
	@echo ""
	@echo "Docker环境:"
	@echo "  docker-build - 构建Docker镜像"
	@echo "  docker-up   - 启动Docker服务"
	@echo "  docker-down - 停止Docker服务"
	@echo "  logs        - 查看服务日志"
	@echo ""
	@echo "数据库:"
	@echo "  db-init     - 初始化数据库"
	@echo "  db-migrate  - 运行数据库迁移"
	@echo "  db-reset    - 重置数据库 (开发环境)"
	@echo ""
	@echo "工具:"
	@echo "  clean       - 清理临时文件"
	@echo "  gpu-check   - 检查GPU状态"
	@echo "  benchmark   - 运行性能基准测试"

# 安装依赖
install:
	@echo "安装Python依赖..."
	uv sync --dev --extra gpu
	@echo "依赖安装完成"

# 启动开发服务器
dev:
	@echo "启动InvestIQ开发服务器..."
	uv run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 运行测试
test:
	@echo "运行测试..."
	uv run pytest backend/tests/ -v --cov=backend --cov-report=html

# 代码检查
lint:
	@echo "运行代码检查..."
	uv run flake8 backend/
	uv run mypy backend/

# 代码格式化
format:
	@echo "格式化代码..."
	uv run black backend/
	uv run isort backend/

# 构建Docker镜像
docker-build:
	@echo "构建Docker镜像..."
	docker-compose build

# 启动Docker服务
docker-up:
	@echo "启动Docker服务..."
	docker-compose up -d
	@echo "等待服务启动..."
	sleep 10
	@echo "服务状态:"
	docker-compose ps

# 停止Docker服务
docker-down:
	@echo "停止Docker服务..."
	docker-compose down

# 查看服务日志
logs:
	docker-compose logs -f

# 初始化数据库
db-init:
	@echo "初始化数据库..."
	uv run python -c "import asyncio; from backend.app.core.database import create_tables; asyncio.run(create_tables())"

# 运行数据库迁移
db-migrate:
	@echo "运行数据库迁移..."
	uv run alembic upgrade head

# 重置数据库 (仅开发环境)
db-reset:
	@echo "重置数据库 (仅开发环境)..."
	uv run python -c "import asyncio; from backend.app.core.database import drop_tables, create_tables; asyncio.run(drop_tables()); asyncio.run(create_tables())"

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# 检查GPU状态
gpu-check:
	@echo "检查GPU状态..."
	uv run python -c "from backend.app.utils.gpu import diagnose_gpu; import json; print(json.dumps(diagnose_gpu(), indent=2, ensure_ascii=False))"

# 运行性能基准测试
benchmark:
	@echo "运行性能基准测试..."
	uv run python -c "from backend.app.utils.gpu import benchmark_gpu; import json; print(json.dumps(benchmark_gpu(), indent=2, ensure_ascii=False))"

# 生成API文档
docs:
	@echo "生成API文档..."
	uv run python -c "from backend.app.main import app; import json; print(json.dumps(app.openapi(), indent=2, ensure_ascii=False))" > docs/openapi.json

# 健康检查
health:
	@echo "执行健康检查..."
	curl -s http://localhost:8000/health | python -m json.tool

# 快速启动 (一键启动所有服务)
quick-start: docker-up
	@echo "等待服务完全启动..."
	sleep 30
	@echo "InvestIQ Platform 已启动!"
	@echo ""
	@echo "访问地址:"
	@echo "  API文档: http://localhost:8000/docs"
	@echo "  健康检查: http://localhost:8000/health"
	@echo "  系统信息: http://localhost:8000/info"
	@echo "  Grafana: http://localhost:3001 (admin/admin123)"
	@echo "  MinIO: http://localhost:9001 (investiq/investiq123)"

# 完整重启
restart: docker-down docker-up

# 开发环境设置
setup-dev: install db-init
	@echo "开发环境设置完成!"
	@echo "运行 'make dev' 启动开发服务器"

# 生产部署检查
prod-check:
	@echo "生产部署检查..."
	@echo "检查配置文件..."
	@test -f .env || (echo "错误: .env 文件不存在" && exit 1)
	@echo "检查Docker配置..."
	docker-compose config
	@echo "检查通过!"
