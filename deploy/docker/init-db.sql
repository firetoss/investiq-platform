-- InvestIQ Platform - 数据库初始化脚本
-- PostgreSQL数据库初始化和优化配置

-- 设置时区
SET timezone = 'Asia/Shanghai';

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 创建数据库用户 (如果不存在)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'investiq') THEN
        CREATE ROLE investiq WITH LOGIN PASSWORD 'investiq123';
    END IF;
END
$$;

-- 授权
GRANT ALL PRIVILEGES ON DATABASE investiq TO investiq;
GRANT ALL ON SCHEMA public TO investiq;

-- 设置数据库参数优化
ALTER DATABASE investiq SET timezone = 'Asia/Shanghai';
ALTER DATABASE investiq SET log_statement = 'mod';
ALTER DATABASE investiq SET log_min_duration_statement = 1000;

-- 创建审计函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建序列号生成函数 (用于审计链)
CREATE OR REPLACE FUNCTION get_next_sequence_number()
RETURNS INTEGER AS $$
DECLARE
    next_seq INTEGER;
BEGIN
    SELECT COALESCE(MAX(sequence_number), 0) + 1 INTO next_seq FROM decision_logs;
    RETURN next_seq;
END;
$$ LANGUAGE plpgsql;

-- 创建哈希计算函数
CREATE OR REPLACE FUNCTION calculate_decision_log_hash(
    seq_num INTEGER,
    user_id TEXT,
    action TEXT,
    payload_hash TEXT,
    timestamp_val TIMESTAMPTZ,
    prev_hash TEXT DEFAULT NULL
)
RETURNS TEXT AS $$
DECLARE
    hash_input TEXT;
BEGIN
    hash_input := seq_num::TEXT || ':' || user_id || ':' || action || ':' || payload_hash || ':' || timestamp_val::TEXT;
    IF prev_hash IS NOT NULL THEN
        hash_input := prev_hash || ':' || hash_input;
    END IF;
    RETURN encode(digest(hash_input, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql;

-- 创建时间分区函数
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;

-- 创建索引优化函数
CREATE OR REPLACE FUNCTION optimize_table_indexes()
RETURNS VOID AS $$
BEGIN
    -- 重建统计信息
    ANALYZE;
    
    -- 记录优化时间
    INSERT INTO system_maintenance_log (operation, performed_at) 
    VALUES ('index_optimization', NOW())
    ON CONFLICT DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- 创建系统维护日志表
CREATE TABLE IF NOT EXISTS system_maintenance_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation VARCHAR(100) NOT NULL,
    details JSONB,
    performed_at TIMESTAMPTZ DEFAULT NOW()
);

-- 创建配置表
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    updated_by VARCHAR(100)
);

-- 插入默认配置
INSERT INTO system_config (key, value, description) VALUES
('scoring_weights_industry', '{"P": 0.35, "E": 0.25, "M": 0.25, "R": 0.15}', '行业评分权重配置'),
('scoring_weights_equity', '{"Q": 0.30, "V": 0.20, "M": 0.25, "C": 0.15, "S": 0.10}', '个股评分权重配置'),
('gate_thresholds', '{"industry": 70, "equity": 70, "valuation_percentile_max": 0.7, "growth_percentile_max": 0.8, "peg_max": 1.5}', '四闸门阈值配置'),
('liquidity_config', '{"participation_rate": {"A": 0.10, "H": 0.08}, "exit_days": {"core": 5, "tactical": 3}}', '流动性配置'),
('circuit_breaker_levels', '{"level_1": -0.10, "level_2": -0.20, "level_3": -0.30}', '回撤断路器级别')
ON CONFLICT (key) DO NOTHING;

-- 创建性能监控视图
CREATE OR REPLACE VIEW performance_summary AS
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public'
ORDER BY tablename, attname;

-- 创建数据质量检查函数
CREATE OR REPLACE FUNCTION check_data_quality()
RETURNS TABLE(
    table_name TEXT,
    total_rows BIGINT,
    null_count BIGINT,
    null_percentage NUMERIC
) AS $$
BEGIN
    -- 这里可以添加具体的数据质量检查逻辑
    RETURN QUERY
    SELECT 
        'placeholder'::TEXT as table_name,
        0::BIGINT as total_rows,
        0::BIGINT as null_count,
        0::NUMERIC as null_percentage;
END;
$$ LANGUAGE plpgsql;

-- 设置默认权限
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO investiq;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO investiq;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO investiq;

-- 记录初始化完成
INSERT INTO system_maintenance_log (operation, details) 
VALUES ('database_initialization', '{"version": "1.0", "timestamp": "' || NOW()::TEXT || '"}');

-- 输出初始化完成信息
DO $$
BEGIN
    RAISE NOTICE 'InvestIQ数据库初始化完成';
    RAISE NOTICE '时区: %', current_setting('timezone');
    RAISE NOTICE '数据库版本: %', version();
END $$;
