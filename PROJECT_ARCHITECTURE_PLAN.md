# InvestIQ 行业选择与组合支持平台 - 总体架构规划

## 项目概述

基于PRD文档和产品设计文档，实现一个完整的行业选择与组合支持平台，支持A股+H股的中周期长多策略。

### 核心业务逻辑
1. **四闸门方法论**：政策→行业→个股的评分体系
2. **行业评分**：P(政策)+E(证据)+M(市场)+R-(风险) 的100分制评分
3. **个股评分**：Q(质量)+V(估值)+M(动量)+C(契合)+S(份额)-R-(红旗) 的权重评分
4. **流动性与容量校验**：基于参与率、退出天数的公式化门槛
5. **组合管理**：A/B/C三层分级，融资约束，回撤断路器

### 技术架构要求
1. **边缘部署**：Jetson Orin AGX 64GB设备
2. **微服务架构**：主业务与模型服务隔离
3. **容器化**：Docker Compose部署
4. **单用户模式**：简化RBAC设计
5. **SOTA模型**：LLM、情感分析、时序预测使用最新模型

## 阶段一：架构设计与基础设施 (Week 1-2)

### 1.1 微服务架构设计
- **核心业务服务**：
  - `scoring-service`：行业/个股评分引擎
  - `gatekeeper-service`：四闸门校验服务  
  - `portfolio-service`：组合构建与管理
  - `liquidity-service`：流动性与容量校验
  - `alert-service`：监控告警中心
  - `evidence-service`：证据链管理
  - `audit-service`：审计轨迹记录

- **AI模型服务**：
  - `llm-service`：大语言模型服务（用于备忘录生成、文本分析）
  - `sentiment-service`：情感分析服务（政策文本、公告分析）
  - `timeseries-service`：时序预测服务（价格趋势、技术指标）

### 1.2 技术栈选型
- **后端**：FastAPI + Pydantic + SQLAlchemy
- **前端**：React 18 + TypeScript + Vite
- **数据库**：PostgreSQL 16 + Redis 7
- **消息队列**：Redis Streams + Celery
- **存储**：MinIO（证据文件存储）
- **搜索**：Meilisearch（全文检索）
- **监控**：Prometheus + Grafana + Loki
- **容器化**：Docker + Docker Compose
- **API网关**：Caddy（TLS终止、路由）

### 1.3 Jetson优化配置
- ARM64架构适配
- GPU内存管理优化
- 模型量化与推理优化
- 容器资源限制配置

## 阶段二：数据模型与API设计 (Week 2-3)

### 2.1 核心数据模型
```python
# 行业评分快照
class IndustryScoreSnapshot:
    industry_id, P, E, M, R_neg, score, ts, editor, method, as_of, source

# 个股评分快照  
class EquityScoreSnapshot:
    ticker, Q, V, M, C, S, R_neg, score, ts, editor, as_of, is_partial, is_stale, confidence

# 证据链
class EvidenceItem:
    entity_type, entity_id, type, url, source, ts, hash, note

# 组合持仓
class Position:
    portfolio_id, equity_id, tier, target_pct, actual_pct, entry_plan_json

# 决策日志（不可变审计链）
class DecisionLog:
    id, user, action, payload_hash, prev_hash, before, after, ts, request_id
```

### 2.2 OpenAPI规范实现
- 基于PRD中的API规范设计RESTful接口
- 实现幂等性（Idempotency-Key）
- 时间旅行查询（asOf参数）
- 标准化错误码与响应头

## 阶段三：评分引擎开发 (Week 3-4)

### 3.1 行业评分引擎
- **公式实现**：Score^Ind = 0.35P + 0.25E + 0.25M + 0.15×(100-R^-)
- **政策强度评估**：基于LLM的政策文本分析
- **落地证据收集**：招标/验收公告自动抓取与分析
- **市场确认指标**：200DMA占比、景气扩散度计算

### 3.2 个股评分引擎
- **质量评分Q(30%)**：ROE、现金流质量、毛利趋势、杠杆分析
- **估值评分V(20%)**：历史分位计算、同业对比、PEG计算
- **动量评分M(25%)**：中期趋势、EPS预期修正分析
- **政策契合C(15%)**：资质订单与政策条款映射
- **份额护城河S(10%)**：市占率、定价权分析
- **红旗检测R-**：应收存货、审计意见、质押风险

### 3.3 估值分位系统
- **回退链实现**：P/E → EV/EBITDA → EV/Sales → Z-Score
- **5年历史窗口**：滚动计算历史分位数
- **置信度模型**：覆盖度×新鲜度×数据源权重

## 阶段四：闸门系统与流动性校验 (Week 4-5)

### 4.1 四闸门实现
- **Gate-Industry**：行业分≥70，三证据二满足
- **Gate-Company**：个股分≥70，无红旗
- **Gate-Valuation**：估值分位≤70%（成长股≤80%，PEG≤1.5）
- **Gate-Execution**：200DMA在上，三段式建仓准备

### 4.2 流动性与容量校验
- **参与率配置**：A股10%，H股8%
- **公式校验**：ADV_min ≥ P_i/(r×D)
- **绝对底线**：A股ADV20≥3000万，H股ADV20≥2000万
- **Board Lot拦截**：H股非整板提醒与建议

## 阶段五：AI模型服务开发 (Week 5-6)

### 5.1 LLM服务
- **模型选择**：Qwen2.5-32B-Instruct（量化版本）
- **功能**：备忘录生成、政策文本分析、证据总结
- **优化**：vLLM推理加速、量化部署

### 5.2 情感分析服务  
- **模型选择**：FinBERT-Chinese或最新中文金融情感模型
- **应用场景**：公告情感分析、政策倾向判断
- **实时处理**：支持批量和流式处理

### 5.3 时序预测服务
- **模型选择**：TimesFM、Chronos或最新Transformer-based时序模型
- **技术指标**：200DMA计算、趋势预测、动量分析
- **GPU优化**：利用Jetson的GPU加速推理

## 阶段六：组合管理与监控 (Week 6-7)

### 6.1 组合构建服务
- **A/B/C分层逻辑**：基于评分的自动分层
- **三段式建仓**：40/30/30比例分批建仓
- **融资约束**：最大1.10x杠杆，单标≤30%
- **再平衡建议**：基于评分变化的动态调整

### 6.2 回撤断路器
- **三级断路器**：-10%/-20%/-30%触发不同动作
- **锁存机制**：回撤型告警的锁存与恢复逻辑
- **自动执行**：去杠杆、减仓、清仓的自动化流程

### 6.3 监控告警系统
- **事件类型**：event/kpi/trend/drawdown四类
- **节流机制**：按类型配置不同间隔
- **升级路径**：P3→P2→P1的自动升级
- **通知路由**：邮件、webhook通知

## 阶段七：前端界面开发 (Week 7-8)

### 7.1 React技术栈
- **核心框架**：React 18 + TypeScript
- **构建工具**：Vite（快速开发与构建）
- **UI组件库**：Ant Design（企业级UI组件）
- **状态管理**：Zustand（轻量级状态管理）
- **路由**：React Router v6
- **图表可视化**：
  - ECharts for React（丰富的金融图表）
  - D3.js（自定义可视化）
  - TradingView Charting Library（专业K线图）
- **表格组件**：Ant Design Table + Virtual Scrolling
- **HTTP客户端**：Axios + React Query（数据获取与缓存）
- **样式方案**：Tailwind CSS + Ant Design
- **表单处理**：React Hook Form + Zod（类型安全验证）

### 7.2 核心页面组件

#### 7.2.1 行业评分台 (`/industry`)
```tsx
// 主要功能
- IndustryScoreTable: 行业评分表格（P/E/M/R-子分展示）
- EvidenceDrawer: 证据链侧边抽屉
- EventCalendar: 事件日历组件
- ScoreVisualization: 评分可视化图表
- BulkGateCheck: 批量闸门检查
```

#### 7.2.2 个股筛选器 (`/equity`)
```tsx
// 主要功能
- EquityFilter: 多维度筛选器（市值/ADV/评分维度）
- EquityTable: 个股列表表格
- ScoreBreakdown: 评分详细拆分
- LiquidityCheck: 流动性校验组件
- BasketManager: 建仓篮子管理
```

#### 7.2.3 组合看板 (`/portfolio`)
```tsx
// 主要功能
- PositionOverview: A/B/C分层仓位概览
- CircuitBreakerStatus: 断路器状态指示器
- LiquidityValidator: 容量校验展示
- RebalanceRecommendations: 再平衡建议
- PortfolioSimulator: 组合模拟模式
```

#### 7.2.4 告警中心 (`/alerts`)
```tsx
// 主要功能
- AlertsTable: 告警列表（类型/严重度/状态）
- AlertDetail: 告警详情抽屉
- ThrottleStatus: 节流状态展示
- EscalationFlow: 升级流程可视化
- AlertHistory: 历史告警记录
```

#### 7.2.5 备忘录生成 (`/memo`)
```tsx
// 主要功能
- MemoEditor: 备忘录编辑器（论点/证据/KPI树）
- TemplateSelector: 模板选择器
- EvidenceLinker: 证据链接器
- ExportOptions: 导出选项（PDF/Markdown）
- PreviewPanel: 实时预览面板
```

### 7.3 React特有优化

#### 7.3.1 性能优化
```tsx
// 虚拟化处理大量数据
import { FixedSizeList as List } from 'react-window';

// 记忆化组件防止不必要重渲染
const ExpensiveComponent = React.memo(({ data }) => {
  // 组件逻辑
});

// 懒加载路由
const IndustryPage = lazy(() => import('./pages/Industry'));
```

#### 7.3.2 实时数据处理
```tsx
// WebSocket连接用于实时更新
const useRealTimeData = (endpoint: string) => {
  const [data, setData] = useState(null);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://api/${endpoint}`);
    ws.onmessage = (event) => setData(JSON.parse(event.data));
    return () => ws.close();
  }, [endpoint]);
  
  return data;
};
```

#### 7.3.3 状态管理架构
```tsx
// Zustand store设计
interface AppState {
  // 行业评分状态
  industryScores: IndustryScore[];
  selectedIndustry: string | null;
  
  // 个股筛选状态
  equityFilters: EquityFilters;
  equityResults: EquityData[];
  
  // 组合状态
  portfolio: PortfolioData;
  positions: Position[];
  
  // 告警状态
  alerts: Alert[];
  activeAlerts: Alert[];
  
  // 全局状态
  user: UserInfo;
  settings: AppSettings;
}
```

## 阶段八：数据集成与ETL (Week 8-9)

### 8.1 数据源集成
- **公开数据源**：交易所官网、政府网站、行业协会
- **数据抓取**：遵循robots.txt、速率限制
- **数据清洗**：标准化格式、去重、质量检查
- **增量更新**：支持增量抓取和全量刷新

### 8.2 EOD批处理
- **200DMA计算**：每日收盘后批量更新
- **估值分位更新**：5年滚动窗口计算
- **快照固化**：PIT数据不可变存储
- **KPI监控**：T+30分钟完成率95%目标

## 阶段九：部署与运维 (Week 9-10)

### 9.1 Jetson部署优化
- **Docker Compose配置**：针对ARM64优化
- **资源限制**：合理分配64GB内存
- **GPU利用**：AI服务的GPU加速配置
- **存储优化**：数据分层存储策略

### 9.2 监控体系
- **Prometheus指标**：业务指标+系统指标
- **Grafana仪表盘**：实时监控+告警可视化  
- **日志聚合**：Loki日志收集+查询
- **链路追踪**：OpenTelemetry分布式追踪

### 9.3 前端部署优化
- **生产构建**：Vite优化构建配置
- **静态资源CDN**：本地MinIO作为CDN
- **Service Worker**：离线缓存支持
- **Bundle分析**：webpack-bundle-analyzer优化包体积

## 阶段十：测试与验收 (Week 10-11)

### 10.1 前端测试
- **单元测试**：Jest + React Testing Library
- **组件测试**：Storybook组件文档与测试
- **端到端测试**：Playwright自动化测试
- **性能测试**：React DevTools Profiler

### 10.2 集成测试  
- **端到端流程**：从评分到建仓的完整流程
- **回撤断路器**：各级断路器触发测试
- **AI模型测试**：模型推理准确性验证

### 10.3 性能测试
- **API响应时间**：P99 ≤ 200ms目标
- **前端渲染性能**：大量数据表格渲染优化
- **内存使用**：Jetson设备资源占用监控

## React前端项目结构

```
frontend/
├── src/
│   ├── components/          # 共用组件
│   │   ├── ui/             # 基础UI组件
│   │   ├── charts/         # 图表组件
│   │   ├── tables/         # 表格组件
│   │   └── forms/          # 表单组件
│   ├── pages/              # 页面组件
│   │   ├── Industry/       # 行业评分页
│   │   ├── Equity/         # 个股筛选页
│   │   ├── Portfolio/      # 组合管理页
│   │   ├── Alerts/         # 告警中心页
│   │   └── Memo/          # 备忘录页
│   ├── hooks/              # 自定义Hooks
│   ├── stores/             # Zustand状态管理
│   ├── services/           # API服务层
│   ├── types/              # TypeScript类型定义
│   ├── utils/              # 工具函数
│   └── constants/          # 常量定义
├── public/                 # 静态资源
├── tests/                  # 测试文件
├── docs/                   # 文档
└── package.json
```

## 关键技术风险与应对

1. **Jetson资源限制**：采用模型量化、内存优化、服务分离
2. **AI模型部署**：使用TensorRT、ONNX优化推理性能
3. **数据质量**：建立完善的数据校验和监控机制
4. **系统稳定性**：实现服务降级、熔断、重试机制
5. **合规要求**：严格遵循数据抓取规范和审计要求
6. **React性能**：大数据量表格虚拟化、组件懒加载、状态优化

## 交付物

1. **完整源代码**：微服务架构的所有组件
2. **React前端应用**：完整的单页应用（SPA）
3. **Docker配置**：适配Jetson的完整部署配置
4. **API文档**：基于OpenAPI 3.0的完整接口文档
5. **组件文档**：Storybook组件库文档
6. **运维手册**：部署、监控、故障排除指南
7. **用户手册**：系统使用和功能说明
8. **测试报告**：包含性能、功能、安全测试结果