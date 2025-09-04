/**
 * InvestIQ Platform - API Types
 * 后端API接口类型定义
 */

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  request_id?: string;
  snapshot_ts: string;
}

// 行业评分相关类型
export interface IndustryScore {
  industry_code: string;
  industry_name: string;
  overall_score: number;
  policy_score: number;
  growth_score: number;
  valuation_score: number;
  technical_score: number;
  policy_weight: number;
  growth_weight: number;
  valuation_weight: number;
  technical_weight: number;
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence_level: number;
  explanation: string;
  risk_factors: string[];
  updated_at: string;
}

export interface IndustryComparison {
  industries: IndustryScore[];
  comparison_metrics: {
    metric_name: string;
    values: Record<string, number>;
  }[];
  benchmark_data: {
    industry_code: string;
    benchmark_name: string;
    benchmark_value: number;
  }[];
}

// 个股评分相关类型
export interface EquityScore {
  ticker: string;
  company_name: string;
  industry_code: string;
  overall_score: number;
  policy_score: number;
  growth_score: number;
  valuation_score: number;
  technical_score: number;
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence_level: number;
  risk_factors: string[];
  key_metrics: Record<string, number>;
  updated_at: string;
}

export interface EquityScreeningRequest {
  industry_codes?: string[];
  min_score?: number;
  max_score?: number;
  recommendations?: string[];
  market_caps?: {
    min?: number;
    max?: number;
  };
  financial_filters?: {
    min_revenue_growth?: number;
    max_pe_ratio?: number;
    min_roe?: number;
  };
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

// 组合管理相关类型
export interface PortfolioPosition {
  ticker: string;
  company_name: string;
  tier: 'A' | 'B' | 'C';
  weight: number;
  shares: number;
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

export interface Portfolio {
  portfolio_id: string;
  name: string;
  description?: string;
  total_capital: number;
  current_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  max_drawdown: number;
  sharpe_ratio: number;
  positions: PortfolioPosition[];
  tier_allocation: {
    A: { target: number; current: number; };
    B: { target: number; current: number; };
    C: { target: number; current: number; };
  };
  risk_metrics: {
    var_95: number;
    beta: number;
    volatility: number;
  };
  circuit_breaker_status: {
    level: 0 | 1 | 2 | 3;
    triggered_at?: string;
    recovery_threshold?: number;
  };
  created_at: string;
  updated_at: string;
}

export interface PortfolioConstructRequest {
  name: string;
  description?: string;
  total_capital: number;
  candidates: {
    ticker: string;
    suggested_tier: 'A' | 'B' | 'C';
    max_weight?: number;
  }[];
  constraints?: {
    max_single_position?: number;
    sector_concentration_limit?: number;
    min_diversification_score?: number;
  };
  risk_tolerance?: 'conservative' | 'balanced' | 'aggressive';
}

// 告警系统相关类型
export interface Alert {
  alert_id: string;
  alert_type: 'event' | 'kpi' | 'trend' | 'drawdown';
  severity: 'P1' | 'P2' | 'P3';
  status: 'new' | 'acknowledged' | 'in_progress' | 'resolved' | 'closed';
  entity_type?: string;
  entity_id?: string;
  ticker?: string;
  title: string;
  message: string;
  rule_name?: string;
  current_value?: number;
  threshold_value?: number;
  assignee?: string;
  hits_count: number;
  is_latched: boolean;
  triggered_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  next_allowed_at?: string;
  is_overdue: boolean;
  can_trigger_now: boolean;
}

export interface AlertRule {
  rule_id: string;
  name: string;
  description?: string;
  alert_type: 'event' | 'kpi' | 'trend' | 'drawdown';
  entity_type?: string;
  condition: Record<string, any>;
  threshold_config?: Record<string, any>;
  severity: 'P1' | 'P2' | 'P3';
  message_template?: string;
  throttle_config?: Record<string, any>;
  is_enabled: boolean;
  created_at: string;
  updated_at: string;
}

// 证据管理相关类型
export interface EvidenceItem {
  evidence_id: string;
  entity_type: string;
  entity_id: string;
  evidence_type: 'policy' | 'order' | 'financial' | 'news' | 'announcement' | 'research' | 'market_data' | 'other';
  title: string;
  description?: string;
  source: 'official' | 'authorized' | 'public';
  source_url?: string;
  source_name?: string;
  content?: string;
  content_hash?: string;
  file_path?: string;
  file_size?: number;
  file_type?: string;
  tags?: string[];
  keywords?: string[];
  confidence_score?: number;
  relevance_score?: number;
  quality_score?: number;
  evidence_date?: string;
  published_at?: string;
  collected_at: string;
  is_verified: boolean;
  is_active: boolean;
  verified_by?: string;
  verified_at?: string;
}

// 通用分页类型
export interface PaginationParams {
  page?: number;
  page_size?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    page: number;
    page_size: number;
    total_count: number;
    total_pages: number;
  };
}

// 仪表板数据类型
export interface DashboardMetrics {
  industry_coverage: number;
  equity_coverage: number;
  portfolio_count: number;
  active_alerts: number;
  system_health: 'healthy' | 'warning' | 'critical';
}