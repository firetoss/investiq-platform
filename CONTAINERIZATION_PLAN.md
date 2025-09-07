# InvestIQ平台 - 项目骨架与容器化实现计划

## 第一阶段：项目架构设计与目录结构

### 1.1 整体项目结构
```
investiq-platform/
├── services/                    # 微服务目录
│   ├── scoring-service/        # 评分引擎服务
│   ├── gatekeeper-service/     # 四闸门校验服务
│   ├── portfolio-service/      # 组合管理服务
│   ├── liquidity-service/      # 流动性校验服务
│   ├── alert-service/          # 告警服务
│   ├── evidence-service/       # 证据链服务
│   ├── audit-service/          # 审计服务
│   ├── llm-service/           # LLM服务
│   ├── sentiment-service/      # 情感分析服务
│   └── timeseries-service/     # 时序预测服务
├── shared/                      # 共享库
│   ├── models/                 # 数据模型定义
│   ├── utils/                  # 工具函数
│   └── config/                 # 配置管理
├── frontend/                    # React前端应用
├── gateway/                     # API网关配置
├── deploy/                      # 部署配置
│   ├── docker/                 # Docker配置
│   └── k8s/                   # Kubernetes配置（可选）
├── scripts/                     # 脚本工具
├── docs/                       # 文档
├── tests/                      # 集成测试
└── docker-compose.yml          # 主编排文件
```

### 1.2 各服务技术栈统一
- **后端服务**：FastAPI + SQLAlchemy + Pydantic
- **数据库**：PostgreSQL 16（主库）+ Redis 7（缓存）
- **消息队列**：Redis Streams + Celery
- **文件存储**：MinIO
- **容器化**：Docker + Docker Compose
- **API文档**：OpenAPI 3.0自动生成

## 第二阶段：共享库开发

### 2.1 数据模型定义 (shared/models/)
```python
# shared/models/base.py
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, String
from datetime import datetime

Base = declarative_base()

class BaseModel(Base):
    __abstract__ = True
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(50))

# shared/models/industry.py
class IndustryScoreSnapshot(BaseModel):
    __tablename__ = 'industry_score_snapshots'
    
    id = Column(String, primary_key=True)
    industry_id = Column(String, nullable=False)
    P = Column(Float)  # 政策强度
    E = Column(Float)  # 落地证据  
    M = Column(Float)  # 市场确认
    R_neg = Column(Float)  # 风险扣分
    score = Column(Float)  # 总分
    # ...其他字段

# shared/models/equity.py
class EquityScoreSnapshot(BaseModel):
    __tablename__ = 'equity_score_snapshots'
    
    id = Column(String, primary_key=True) 
    ticker = Column(String, nullable=False)
    Q = Column(Float)  # 质量
    V = Column(Float)  # 估值
    M = Column(Float)  # 动量
    C = Column(Float)  # 政策契合
    S = Column(Float)  # 份额护城河
    R_neg = Column(Float)  # 红旗扣分
    # ...其他字段
```

### 2.2 配置管理 (shared/config/)
```python
# shared/config/settings.py
from pydantic import BaseSettings
from typing import Dict, Any

class DatabaseSettings(BaseSettings):
    host: str = "postgres"
    port: int = 5432
    username: str = "investiq"
    password: str = "password"
    database: str = "investiq"
    
class RedisSettings(BaseSettings):
    host: str = "redis"
    port: int = 6379
    db: int = 0
    
class AppSettings(BaseSettings):
    timezone: str = "Asia/Shanghai"
    currency_default: str = "CNY"
    debug: bool = False
    
    # 各服务端口配置
    scoring_service_port: int = 8001
    gatekeeper_service_port: int = 8002
    portfolio_service_port: int = 8003
    # ...其他服务端口
```

### 2.3 通用工具 (shared/utils/)
```python
# shared/utils/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid

def create_app(title: str, version: str = "1.0.0") -> FastAPI:
    app = FastAPI(
        title=title,
        version=version,
        docs_url="/docs",
        openapi_url="/openapi.json"
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app

# shared/utils/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from shared.config.settings import DatabaseSettings

def get_database_url(settings: DatabaseSettings) -> str:
    return f"postgresql://{settings.username}:{settings.password}@{settings.host}:{settings.port}/{settings.database}"

def create_db_engine(settings: DatabaseSettings):
    engine = create_engine(get_database_url(settings))
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal
```

## 第三阶段：核心业务服务开发

### 3.1 评分服务 (services/scoring-service/)
```python
# services/scoring-service/main.py
from fastapi import FastAPI, Depends
from shared.utils.api import create_app
from shared.config.settings import AppSettings
from .routers import industry, equity

app = create_app("Scoring Service", "1.0.0")
settings = AppSettings()

# 路由注册
app.include_router(industry.router, prefix="/api/v1/industry", tags=["industry"])
app.include_router(equity.router, prefix="/api/v1/equity", tags=["equity"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "scoring-service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.scoring_service_port)

# services/scoring-service/routers/industry.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class IndustryScoreRequest(BaseModel):
    industry_id: str
    P: float
    E: float  
    M: float
    R_neg: float
    evidences: List[dict] = []

class IndustryScoreResponse(BaseModel):
    industry_id: str
    score: float
    snapshot_id: str

@router.post("/score", response_model=IndustryScoreResponse)
async def calculate_industry_score(request: IndustryScoreRequest):
    # 行业评分公式：Score = 0.35*P + 0.25*E + 0.25*M + 0.15*(100-R_neg)
    score = 0.35 * request.P + 0.25 * request.E + 0.25 * request.M + 0.15 * (100 - request.R_neg)
    
    # 生成快照ID
    snapshot_id = f"ind_{request.industry_id}_{int(time.time())}"
    
    # TODO: 保存到数据库
    
    return IndustryScoreResponse(
        industry_id=request.industry_id,
        score=round(score, 2),
        snapshot_id=snapshot_id
    )
```

### 3.2 闸门服务 (services/gatekeeper-service/)
```python
# services/gatekeeper-service/main.py
from fastapi import FastAPI
from shared.utils.api import create_app
from .routers import gates

app = create_app("Gatekeeper Service", "1.0.0")

app.include_router(gates.router, prefix="/api/v1", tags=["gates"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "gatekeeper-service"}

# services/gatekeeper-service/routers/gates.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class GateCheckRequest(BaseModel):
    industry_score: float
    equity_score: float
    valuation_percentile: float
    peg: Optional[float] = None
    above_200dma: bool
    
class GateCheckResponse(BaseModel):
    pass_all: bool
    failed_gates: List[str]
    details: dict

@router.post("/gate/check", response_model=GateCheckResponse)
async def check_gates(request: GateCheckRequest):
    failed_gates = []
    details = {}
    
    # Gate 1: Industry Score >= 70
    if request.industry_score < 70:
        failed_gates.append("industry")
        details["industry"] = f"Score {request.industry_score} < 70"
    
    # Gate 2: Equity Score >= 70
    if request.equity_score < 70:
        failed_gates.append("equity")
        details["equity"] = f"Score {request.equity_score} < 70"
        
    # Gate 3: Valuation Percentile <= 70% (or 80% for growth with PEG <= 1.5)
    valuation_threshold = 70
    if request.peg and request.peg <= 1.5:
        valuation_threshold = 80
        
    if request.valuation_percentile > valuation_threshold:
        failed_gates.append("valuation")
        details["valuation"] = f"Percentile {request.valuation_percentile}% > {valuation_threshold}%"
    
    # Gate 4: Above 200DMA
    if not request.above_200dma:
        failed_gates.append("execution")
        details["execution"] = "Price below 200DMA"
    
    return GateCheckResponse(
        pass_all=len(failed_gates) == 0,
        failed_gates=failed_gates,
        details=details
    )
```

### 3.3 流动性服务 (services/liquidity-service/)
```python
# services/liquidity-service/main.py
from fastapi import FastAPI
from shared.utils.api import create_app
from .routers import liquidity

app = create_app("Liquidity Service", "1.0.0")
app.include_router(liquidity.router, prefix="/api/v1", tags=["liquidity"])

# services/liquidity-service/routers/liquidity.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class LiquidityCheckRequest(BaseModel):
    ticker: str
    target_position: float
    currency: str
    ADV20: float  # 20日平均成交额
    turnover: float  # 日换手率
    free_float_mkt_cap: float  # 自由流通市值
    r: float = 0.10  # 参与率，A股默认10%，H股8%
    D: int = 5  # 退出天数，核心5天，战术3天

class LiquidityCheckResponse(BaseModel):
    adv_min: float
    absolute_floor_pass: bool
    free_float_cap_pass: bool
    used_participation_rate: float
    used_exit_days: int
    free_float_utilization_pct: float
    notes: List[str]

@router.post("/check", response_model=LiquidityCheckResponse)
async def check_liquidity(request: LiquidityCheckRequest):
    notes = []
    
    # 最小ADV要求：ADV_min = P_i / (r * D)
    adv_min = request.target_position / (request.r * request.D)
    
    # 绝对底线检查
    if request.currency == "CNY":  # A股
        absolute_floor_pass = request.ADV20 >= 30_000_000 and request.turnover >= 0.005
        if not absolute_floor_pass:
            notes.append(f"A股底线：ADV20需≥3000万，换手≥0.5%")
    else:  # H股
        absolute_floor_pass = request.ADV20 >= 20_000_000 and request.turnover >= 0.003
        if not absolute_floor_pass:
            notes.append(f"H股底线：ADV20需≥2000万，换手≥0.3%")
    
    # 自由流通占用检查
    free_float_utilization_pct = (request.target_position / request.free_float_mkt_cap) * 100
    free_float_cap_pass = free_float_utilization_pct <= 2.0
    
    if not free_float_cap_pass:
        notes.append(f"自由流通占用{free_float_utilization_pct:.2f}%超过2%限制")
    
    return LiquidityCheckResponse(
        adv_min=adv_min,
        absolute_floor_pass=absolute_floor_pass,
        free_float_cap_pass=free_float_cap_pass,
        used_participation_rate=request.r,
        used_exit_days=request.D,
        free_float_utilization_pct=free_float_utilization_pct,
        notes=notes
    )
```

## 第四阶段：容器化配置

### 4.1 各服务Dockerfile模板
```dockerfile
# services/scoring-service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制共享库
COPY shared/ /app/shared/
COPY services/scoring-service/ /app/

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 4.2 主docker-compose.yml
```yaml
version: '3.8'

services:
  # 数据库服务
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: investiq
      POSTGRES_USER: investiq
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deploy/sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U investiq"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MinIO对象存储
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 核心业务服务
  scoring-service:
    build:
      context: .
      dockerfile: services/scoring-service/Dockerfile
    ports:
      - "8001:8001"
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_HOST=postgres
      - REDIS_HOST=redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  gatekeeper-service:
    build:
      context: .
      dockerfile: services/gatekeeper-service/Dockerfile  
    ports:
      - "8002:8002"
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_HOST=postgres
      - REDIS_HOST=redis

  portfolio-service:
    build:
      context: .
      dockerfile: services/portfolio-service/Dockerfile
    ports:
      - "8003:8003"
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_HOST=postgres
      - REDIS_HOST=redis

  liquidity-service:
    build:
      context: .
      dockerfile: services/liquidity-service/Dockerfile
    ports:
      - "8004:8004"
    depends_on:
      - postgres
      - redis

  alert-service:
    build:
      context: .
      dockerfile: services/alert-service/Dockerfile
    ports:
      - "8005:8005"
    depends_on:
      - postgres
      - redis

  evidence-service:
    build:
      context: .
      dockerfile: services/evidence-service/Dockerfile
    ports:
      - "8006:8006"
    depends_on:
      - postgres
      - redis
      - minio

  audit-service:
    build:
      context: .
      dockerfile: services/audit-service/Dockerfile
    ports:
      - "8007:8007"
    depends_on:
      - postgres
      - redis

  # AI模型服务（基础版本，后续优化）
  llm-service:
    build:
      context: .
      dockerfile: services/llm-service/Dockerfile
    ports:
      - "8011:8011"
    environment:
      - MODEL_PATH=/models
      - DEVICE=cuda
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  sentiment-service:
    build:
      context: .
      dockerfile: services/sentiment-service/Dockerfile
    ports:
      - "8012:8012"

  timeseries-service:
    build:
      context: .
      dockerfile: services/timeseries-service/Dockerfile
    ports:
      - "8013:8013"

  # API网关
  gateway:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./gateway/Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - scoring-service
      - gatekeeper-service
      - portfolio-service
      - liquidity-service

  # 前端应用
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - gateway

volumes:
  postgres_data:
  redis_data:
  minio_data:
  caddy_data:
  caddy_config:

networks:
  default:
    name: investiq_network
```

### 4.3 API网关配置
```caddyfile
# gateway/Caddyfile
{
    auto_https off
}

localhost {
    # API路由
    handle_path /api/v1/scoring/* {
        reverse_proxy scoring-service:8001
    }
    
    handle_path /api/v1/gates/* {
        reverse_proxy gatekeeper-service:8002
    }
    
    handle_path /api/v1/portfolio/* {
        reverse_proxy portfolio-service:8003
    }
    
    handle_path /api/v1/liquidity/* {
        reverse_proxy liquidity-service:8004
    }
    
    handle_path /api/v1/alerts/* {
        reverse_proxy alert-service:8005
    }
    
    handle_path /api/v1/evidence/* {
        reverse_proxy evidence-service:8006
    }
    
    handle_path /api/v1/audit/* {
        reverse_proxy audit-service:8007
    }
    
    # AI服务路由
    handle_path /api/v1/llm/* {
        reverse_proxy llm-service:8011
    }
    
    handle_path /api/v1/sentiment/* {
        reverse_proxy sentiment-service:8012
    }
    
    handle_path /api/v1/timeseries/* {
        reverse_proxy timeseries-service:8013
    }
    
    # 前端应用
    handle /* {
        reverse_proxy frontend:80
    }
}
```

## 第五阶段：启动脚本与验证

### 5.1 启动脚本
```bash
#!/bin/bash
# scripts/start.sh

echo "启动InvestIQ平台..."

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
mkdir -p models data/postgres data/redis data/minio

# 构建并启动所有服务
docker-compose up --build -d

echo "等待服务启动..."
sleep 30

# 健康检查
echo "检查服务状态..."
curl -f http://localhost:8001/health || echo "评分服务未就绪"
curl -f http://localhost:8002/health || echo "闸门服务未就绪"
curl -f http://localhost:8003/health || echo "组合服务未就绪"
curl -f http://localhost:8004/health || echo "流动性服务未就绪"

echo "InvestIQ平台启动完成！"
echo "前端访问地址：http://localhost:3000"
echo "API文档地址：http://localhost:8001/docs"
```

### 5.2 服务验证脚本
```bash
#!/bin/bash
# scripts/verify-services.sh

echo "验证各服务是否正常运行..."

services=(
    "localhost:8001/health:评分服务"
    "localhost:8002/health:闸门服务"
    "localhost:8003/health:组合服务"
    "localhost:8004/health:流动性服务"
    "localhost:8005/health:告警服务"
    "localhost:8006/health:证据服务"
    "localhost:8007/health:审计服务"
)

for service in "${services[@]}"; do
    IFS=':' read -r url name <<< "$service"
    if curl -s -f "http://$url" > /dev/null; then
        echo "✅ $name 运行正常"
    else
        echo "❌ $name 运行异常"
    fi
done

# 测试基础API调用
echo "测试评分API..."
curl -X POST "http://localhost:8001/api/v1/industry/score" \
  -H "Content-Type: application/json" \
  -d '{
    "industry_id": "semiconductor",
    "P": 80,
    "E": 75,
    "M": 70,
    "R_neg": 20
  }'
```

### 5.3 各服务依赖文件结构

#### 5.3.1 共享依赖文件
```requirements.txt
# shared/requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
pydantic==2.5.0
python-multipart==0.0.6
```

#### 5.3.2 各服务特定依赖
```python
# services/scoring-service/requirements.txt
-r ../../shared/requirements.txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# services/llm-service/requirements.txt  
-r ../../shared/requirements.txt
transformers==4.35.0
torch==2.1.0
accelerate==0.24.1
vllm==0.2.2

# services/sentiment-service/requirements.txt
-r ../../shared/requirements.txt
transformers==4.35.0
torch==2.1.0
```

## 第六阶段：基础服务框架

### 6.1 服务基础框架代码结构
```
services/[service-name]/
├── main.py              # 服务入口
├── routers/            # 路由模块
│   ├── __init__.py
│   └── [domain].py
├── models/             # 服务特定模型
├── services/           # 业务逻辑层
├── repositories/       # 数据访问层
├── schemas/            # Pydantic模式
├── dependencies/       # 依赖注入
├── config/            # 服务配置
├── tests/             # 单元测试
├── Dockerfile
└── requirements.txt
```

### 6.2 Docker构建优化配置
```dockerfile
# 多阶段构建模板
FROM python:3.11-slim as base

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base as dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM dependencies as application
COPY shared/ /app/shared/
COPY services/[SERVICE_NAME]/ /app/

# 非root用户
RUN useradd --create-home --shell /bin/bash app
USER app

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

## 实现优先级与验证目标

### 优先级顺序
1. **第一优先级**：基础设施服务（postgres, redis, minio）正常启动
2. **第二优先级**：核心业务服务（scoring, gatekeeper, portfolio, liquidity）健康检查通过
3. **第三优先级**：AI服务基础框架就绪（可暂不加载模型）
4. **第四优先级**：API网关路由正常，服务间通信畅通
5. **第五优先级**：前端应用可访问，基础页面渲染

### 验证标准
- ✅ 所有容器都能正常启动且健康检查通过
- ✅ 各服务的 `/health` 端点返回正常状态
- ✅ API文档页面可正常访问（如 http://localhost:8001/docs）
- ✅ 基础的API调用能返回预期结果
- ✅ 前端应用能正常访问（即使功能未完善）
- ✅ 数据库连接正常，基础表结构创建成功
- ✅ Redis缓存服务可用
- ✅ MinIO对象存储服务可用
- ✅ API网关路由转发正常

### 关键检查点
1. **容器启动检查**：`docker-compose ps` 显示所有服务为 `Up` 状态
2. **健康检查通过**：所有服务健康检查通过，无重复重启
3. **端口可达性**：所有暴露端口可以正常访问
4. **服务间通信**：通过API网关可以正常访问各个微服务
5. **数据库初始化**：数据库表结构创建完成，连接池正常
6. **日志检查**：`docker-compose logs` 无严重错误日志

### 成功标准
完成后应该能够：
- 通过 `http://localhost` 访问前端应用
- 通过 `http://localhost:8001/docs` 等访问各服务API文档
- 各服务的健康检查端点返回200状态
- 可以通过API网关调用各个微服务接口
- 数据库、Redis、MinIO等基础服务正常运行