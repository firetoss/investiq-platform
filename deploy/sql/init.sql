-- InvestIQ Platform Database Initialization Script
-- 创建数据库表结构和基础数据

-- 设置时区
SET timezone = 'Asia/Shanghai';

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 创建行业评分快照表
CREATE TABLE IF NOT EXISTS industry_score_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    industry_id VARCHAR(100) NOT NULL,
    industry_name VARCHAR(200),
    P DECIMAL(5,2), -- 政策强度
    E DECIMAL(5,2), -- 落地证据
    M DECIMAL(5,2), -- 市场确认
    R_neg DECIMAL(5,2), -- 风险扣分
    score DECIMAL(5,2) NOT NULL, -- 总分
    method VARCHAR(50) DEFAULT 'standard',
    as_of DATE NOT NULL,
    source VARCHAR(200),
    is_partial BOOLEAN DEFAULT FALSE,
    is_stale BOOLEAN DEFAULT FALSE,
    confidence DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- 创建个股基础信息表
CREATE TABLE IF NOT EXISTS equities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL UNIQUE,
    exchange VARCHAR(10) NOT NULL,
    name VARCHAR(200) NOT NULL,
    market_cap BIGINT,
    free_float_cap BIGINT,
    adv20 BIGINT, -- 20日平均成交额
    board_lot INTEGER DEFAULT 100,
    currency VARCHAR(3) DEFAULT 'CNY',
    industry_id VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建个股评分快照表
CREATE TABLE IF NOT EXISTS equity_score_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    Q DECIMAL(5,2), -- 质量评分
    V DECIMAL(5,2), -- 估值评分
    M DECIMAL(5,2), -- 动量评分
    C DECIMAL(5,2), -- 政策契合评分
    S DECIMAL(5,2), -- 份额护城河评分
    R_neg DECIMAL(5,2), -- 红旗扣分
    score DECIMAL(5,2) NOT NULL, -- 总分
    as_of DATE NOT NULL,
    is_partial BOOLEAN DEFAULT FALSE,
    is_stale BOOLEAN DEFAULT FALSE,
    confidence DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    FOREIGN KEY (ticker) REFERENCES equities(ticker)
);

-- 创建估值分位数快照表
CREATE TABLE IF NOT EXISTS valuation_percentile_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(20) NOT NULL,
    window_years INTEGER DEFAULT 5,
    percentile DECIMAL(5,2) NOT NULL,
    as_of DATE NOT NULL,
    method VARCHAR(50) DEFAULT 'PE',
    is_partial BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES equities(ticker)
);

-- 创建证据项表
CREATE TABLE IF NOT EXISTS evidence_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type VARCHAR(50) NOT NULL, -- 'industry' or 'equity'
    entity_id VARCHAR(100) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'policy', 'evidence', 'market', 'quality'等
    url TEXT,
    source VARCHAR(200),
    title VARCHAR(500),
    content TEXT,
    hash VARCHAR(64), -- SHA-256 hash
    note TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- 创建组合表
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    capital BIGINT NOT NULL, -- 初始资金
    leverage_max DECIMAL(3,2) DEFAULT 1.10,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

-- 创建持仓表
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    tier VARCHAR(1) NOT NULL, -- 'A', 'B', 'C'
    target_pct DECIMAL(5,4), -- 目标仓位百分比
    actual_pct DECIMAL(5,4), -- 实际仓位百分比
    entry_plan_json JSONB, -- 建仓计划详情
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
    FOREIGN KEY (ticker) REFERENCES equities(ticker)
);

-- 创建告警表
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(50) NOT NULL, -- 'event', 'kpi', 'trend', 'drawdown'
    entity_type VARCHAR(50), -- 'industry', 'equity', 'portfolio'
    entity_id VARCHAR(100),
    severity VARCHAR(10) DEFAULT 'P3', -- 'P1', 'P2', 'P3'
    rule VARCHAR(200),
    message TEXT,
    status VARCHAR(20) DEFAULT 'new', -- 'new', 'acknowledged', 'resolved'
    hits INTEGER DEFAULT 1,
    assignee VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- 创建决策日志表（不可变审计链）
CREATE TABLE IF NOT EXISTS decision_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    payload_hash VARCHAR(64) NOT NULL, -- SHA-256
    prev_hash VARCHAR(64), -- 链式哈希
    before_data JSONB,
    after_data JSONB,
    request_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_industry_snapshots_industry_id ON industry_score_snapshots(industry_id);
CREATE INDEX IF NOT EXISTS idx_industry_snapshots_as_of ON industry_score_snapshots(as_of DESC);
CREATE INDEX IF NOT EXISTS idx_industry_snapshots_score ON industry_score_snapshots(score DESC);

CREATE INDEX IF NOT EXISTS idx_equities_ticker ON equities(ticker);
CREATE INDEX IF NOT EXISTS idx_equities_exchange ON equities(exchange);
CREATE INDEX IF NOT EXISTS idx_equities_industry ON equities(industry_id);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ticker ON equity_score_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_as_of ON equity_score_snapshots(as_of DESC);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_score ON equity_score_snapshots(score DESC);

CREATE INDEX IF NOT EXISTS idx_valuation_snapshots_ticker ON valuation_percentile_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_valuation_snapshots_as_of ON valuation_percentile_snapshots(as_of DESC);

CREATE INDEX IF NOT EXISTS idx_evidence_entity ON evidence_items(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_evidence_type ON evidence_items(type);
CREATE INDEX IF NOT EXISTS idx_evidence_created_at ON evidence_items(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_positions_portfolio_id ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_tier ON positions(tier);

CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(type);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_decision_logs_user ON decision_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_decision_logs_action ON decision_logs(action);
CREATE INDEX IF NOT EXISTS idx_decision_logs_created_at ON decision_logs(created_at DESC);

-- 创建触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表添加更新时间触发器
CREATE TRIGGER update_industry_score_snapshots_updated_at 
    BEFORE UPDATE ON industry_score_snapshots 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_equities_updated_at 
    BEFORE UPDATE ON equities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_equity_score_snapshots_updated_at 
    BEFORE UPDATE ON equity_score_snapshots 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at 
    BEFORE UPDATE ON portfolios 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at 
    BEFORE UPDATE ON positions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alerts_updated_at 
    BEFORE UPDATE ON alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入初始数据

-- 插入测试行业数据
INSERT INTO industry_score_snapshots (industry_id, industry_name, P, E, M, R_neg, score, as_of, created_by) 
VALUES 
    ('semiconductor', '半导体', 85, 75, 70, 20, 74.5, CURRENT_DATE, 'system'),
    ('new_energy', '新能源', 80, 70, 65, 25, 70.25, CURRENT_DATE, 'system'),
    ('ai_software', 'AI软件', 90, 80, 75, 15, 80.75, CURRENT_DATE, 'system')
ON CONFLICT DO NOTHING;

-- 插入测试个股数据
INSERT INTO equities (ticker, exchange, name, market_cap, free_float_cap, adv20, board_lot, currency, industry_id) 
VALUES 
    ('600519.SH', 'SSE', '贵州茅台', 2500000000000, 1250000000000, 250000000, 100, 'CNY', 'consumer'),
    ('000858.SZ', 'SZSE', '五粮液', 800000000000, 400000000000, 180000000, 100, 'CNY', 'consumer'),
    ('0700.HK', 'HKEX', '腾讯控股', 350000000000, 320000000000, 200000000, 100, 'HKD', 'internet'),
    ('0941.HK', 'HKEX', '中国移动', 180000000000, 90000000000, 120000000, 100, 'HKD', 'telecom')
ON CONFLICT (ticker) DO NOTHING;

-- 插入测试组合数据
INSERT INTO portfolios (name, capital, leverage_max, created_by) 
VALUES 
    ('示例组合', 5000000, 1.10, 'system')
ON CONFLICT DO NOTHING;

-- 创建用户表（简化版本，单用户模式）
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(200),
    role VARCHAR(50) DEFAULT 'analyst',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 插入默认用户
INSERT INTO users (username, email, role) 
VALUES ('admin', 'admin@investiq.local', 'admin')
ON CONFLICT (username) DO NOTHING;

-- 输出初始化完成信息
DO $$
BEGIN
    RAISE NOTICE 'InvestIQ Platform database initialization completed successfully';
    RAISE NOTICE 'Tables created: % industry_score_snapshots, equities, equity_score_snapshots, valuation_percentile_snapshots, evidence_items, portfolios, positions, alerts, decision_logs, users', 
    (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public');
END $$;