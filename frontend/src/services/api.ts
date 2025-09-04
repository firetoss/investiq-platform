/**
 * InvestIQ Platform - API Client
 * 后端API接口封装
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { 
  ApiResponse, 
  IndustryScore, 
  IndustryComparison,
  EquityScore,
  EquityScreeningRequest,
  Portfolio,
  PortfolioConstructRequest,
  Alert,
  AlertRule,
  EvidenceItem,
  PaginatedResponse 
} from '@/types/api';

class ApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api/v1') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        // 添加请求ID
        config.headers['X-Request-ID'] = `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // 通用请求方法
  private async request<T>(config: AxiosRequestConfig): Promise<T> {
    const response = await this.client.request(config);
    return response.data;
  }

  // 行业评分API
  async getIndustryScores(params?: { 
    days?: number; 
    min_score?: number; 
    industry_codes?: string[]; 
  }): Promise<ApiResponse<IndustryScore[]>> {
    return this.request({
      method: 'GET',
      url: '/scoring/industries',
      params,
    });
  }

  async compareIndustries(
    industry_codes: string[], 
    metrics?: string[]
  ): Promise<ApiResponse<IndustryComparison>> {
    return this.request({
      method: 'POST',
      url: '/scoring/industries/compare',
      data: { industry_codes, metrics },
    });
  }

  async getIndustryHistory(
    industry_code: string, 
    days: number = 30
  ): Promise<ApiResponse<IndustryScore[]>> {
    return this.request({
      method: 'GET',
      url: `/scoring/industries/${industry_code}/history`,
      params: { days },
    });
  }

  // 个股评分API
  async getEquityScore(ticker: string): Promise<ApiResponse<EquityScore>> {
    return this.request({
      method: 'GET',
      url: `/scoring/equities/${ticker}`,
    });
  }

  async screenEquities(
    filters: EquityScreeningRequest
  ): Promise<ApiResponse<PaginatedResponse<EquityScore>>> {
    return this.request({
      method: 'POST',
      url: '/scoring/equities/screen',
      data: filters,
    });
  }

  async getEquityHistory(
    ticker: string, 
    days: number = 30
  ): Promise<ApiResponse<EquityScore[]>> {
    return this.request({
      method: 'GET',
      url: `/scoring/equities/${ticker}/history`,
      params: { days },
    });
  }

  // 组合管理API
  async getPortfolios(): Promise<ApiResponse<Portfolio[]>> {
    return this.request({
      method: 'GET',
      url: '/portfolio/list',
    });
  }

  async getPortfolio(portfolio_id: string): Promise<ApiResponse<Portfolio>> {
    return this.request({
      method: 'GET',
      url: `/portfolio/${portfolio_id}`,
    });
  }

  async constructPortfolio(
    request: PortfolioConstructRequest
  ): Promise<ApiResponse<Portfolio>> {
    return this.request({
      method: 'POST',
      url: '/portfolio/construct',
      data: request,
    });
  }

  async rebalancePortfolio(
    portfolio_id: string,
    candidates?: { ticker: string; suggested_tier: 'A' | 'B' | 'C'; }[]
  ): Promise<ApiResponse<Portfolio>> {
    return this.request({
      method: 'POST',
      url: `/portfolio/${portfolio_id}/rebalance`,
      data: { candidates },
    });
  }

  async getPortfolioMetrics(
    portfolio_id: string,
    start_date?: string,
    end_date?: string
  ): Promise<ApiResponse<any>> {
    return this.request({
      method: 'GET',
      url: `/portfolio/${portfolio_id}/metrics`,
      params: { start_date, end_date },
    });
  }

  // 告警系统API
  async getAlerts(params?: {
    page?: number;
    page_size?: number;
    alert_type?: string;
    severity?: string;
    status?: string;
    assignee?: string;
    entity_type?: string;
    ticker?: string;
    start_date?: string;
    end_date?: string;
  }): Promise<ApiResponse<PaginatedResponse<Alert>>> {
    return this.request({
      method: 'GET',
      url: '/alerts/list',
      params,
    });
  }

  async createAlert(alert: {
    alert_type: string;
    entity_type?: string;
    entity_id?: string;
    ticker?: string;
    title: string;
    message: string;
    rule_name?: string;
    severity?: string;
    current_value?: number;
    threshold_value?: number;
    details?: Record<string, any>;
  }): Promise<ApiResponse<Alert>> {
    return this.request({
      method: 'POST',
      url: '/alerts/create',
      data: alert,
    });
  }

  async acknowledgeAlert(
    alert_id: string,
    assignee?: string
  ): Promise<ApiResponse<Alert>> {
    return this.request({
      method: 'POST',
      url: `/alerts/${alert_id}/acknowledge`,
      params: { assignee },
    });
  }

  async updateAlert(
    alert_id: string,
    updates: {
      status?: string;
      assignee?: string;
      resolution_note?: string;
    }
  ): Promise<ApiResponse<Alert>> {
    return this.request({
      method: 'PUT',
      url: `/alerts/${alert_id}`,
      data: updates,
    });
  }

  async getAlertRules(params?: {
    enabled_only?: boolean;
    alert_type?: string;
  }): Promise<ApiResponse<{ rules: AlertRule[]; total_count: number; enabled_count: number; }>> {
    return this.request({
      method: 'GET',
      url: '/alerts/rules',
      params,
    });
  }

  async getAlertDashboard(days: number = 7): Promise<ApiResponse<any>> {
    return this.request({
      method: 'GET',
      url: '/alerts/dashboard',
      params: { days },
    });
  }

  // 证据管理API
  async attachEvidence(evidence: {
    entity_type: string;
    entity_id: string;
    evidence_type: string;
    title: string;
    description?: string;
    source: string;
    source_url?: string;
    source_name?: string;
    content?: string;
    evidence_date?: string;
    published_at?: string;
    tags?: string[];
    keywords?: string[];
    metadata?: Record<string, any>;
  }): Promise<ApiResponse<EvidenceItem>> {
    return this.request({
      method: 'POST',
      url: '/evidence/attach',
      data: evidence,
    });
  }

  async uploadEvidenceFile(
    file: File,
    metadata: {
      entity_type: string;
      entity_id: string;
      evidence_type: string;
      title: string;
      description?: string;
      source?: string;
      tags?: string;
    }
  ): Promise<ApiResponse<EvidenceItem>> {
    const formData = new FormData();
    formData.append('file', file);
    Object.entries(metadata).forEach(([key, value]) => {
      if (value !== undefined) {
        formData.append(key, value);
      }
    });

    return this.request({
      method: 'POST',
      url: '/evidence/upload',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  }

  async getEvidenceByEntity(
    entity_type: string,
    entity_id: string,
    params?: {
      evidence_type?: string;
      source?: string;
      verified_only?: boolean;
      active_only?: boolean;
      start_date?: string;
      end_date?: string;
      page?: number;
      page_size?: number;
    }
  ): Promise<ApiResponse<PaginatedResponse<EvidenceItem>>> {
    return this.request({
      method: 'GET',
      url: '/evidence/by-entity',
      params: { entity_type, entity_id, ...params },
    });
  }

  async verifyEvidence(
    evidence_id: string,
    verification: {
      verified_by: string;
      verification_note?: string;
      confidence_score?: number;
      relevance_score?: number;
      quality_score?: number;
    }
  ): Promise<ApiResponse<EvidenceItem>> {
    return this.request({
      method: 'PUT',
      url: `/evidence/${evidence_id}/verify`,
      data: verification,
    });
  }

  async searchEvidence(params: {
    query: string;
    entity_type?: string;
    evidence_type?: string;
    source?: string;
    verified_only?: boolean;
    min_score?: number;
    page?: number;
    page_size?: number;
  }): Promise<ApiResponse<PaginatedResponse<EvidenceItem>>> {
    return this.request({
      method: 'GET',
      url: '/evidence/search',
      params,
    });
  }

  // 系统健康检查
  async healthCheck(): Promise<{ status: string; [key: string]: any }> {
    const [scoring, portfolio, alerts, evidence] = await Promise.allSettled([
      this.request({ method: 'GET', url: '/scoring/health' }),
      this.request({ method: 'GET', url: '/portfolio/health' }),
      this.request({ method: 'GET', url: '/alerts/health' }),
      this.request({ method: 'GET', url: '/evidence/health' }),
    ]);

    return {
      status: 'healthy',
      services: {
        scoring: scoring.status === 'fulfilled' ? 'healthy' : 'error',
        portfolio: portfolio.status === 'fulfilled' ? 'healthy' : 'error',
        alerts: alerts.status === 'fulfilled' ? 'healthy' : 'error',
        evidence: evidence.status === 'fulfilled' ? 'healthy' : 'error',
      },
    };
  }
}

export const apiClient = new ApiClient();
export default apiClient;