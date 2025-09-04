/**
 * InvestIQ Platform - 组合管理看板
 * 提供投资组合创建、管理和监控功能
 */

import React, { useState, useMemo } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Button, 
  Space, 
  Typography, 
  Tag,
  Progress,
  Statistic,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Divider,
  Alert,
  Badge,
  Tooltip,
  Descriptions,
  Switch
} from 'antd';
import { 
  PlusOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  PieChartOutlined,
  BarChartOutlined,
  ExclamationCircleOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  DollarOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { Portfolio, PortfolioPosition, PortfolioConstructRequest } from '@/types/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Option } = Select;

export default function PortfoliosPage() {
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [createForm] = Form.useForm();
  const queryClient = useQueryClient();

  // 获取组合列表
  const { data: portfolioData, isLoading, refetch } = useQuery({
    queryKey: ['portfolios'],
    queryFn: () => apiClient.getPortfolios(),
    refetchInterval: 30000, // 30秒刷新
  });

  // 获取组合详情
  const { data: portfolioDetail } = useQuery({
    queryKey: ['portfolio-detail', selectedPortfolio?.portfolio_id],
    queryFn: () => selectedPortfolio ? 
      apiClient.getPortfolio(selectedPortfolio.portfolio_id) : null,
    enabled: !!selectedPortfolio,
  });

  // 获取行业选项
  const { data: industryOptions } = useQuery({
    queryKey: ['industry-options'],
    queryFn: () => apiClient.getIndustryScores({ days: 1 }),
  });

  // 创建组合
  const createPortfolioMutation = useMutation({
    mutationFn: (data: PortfolioConstructRequest) => apiClient.constructPortfolio(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
      setCreateModalVisible(false);
      createForm.resetFields();
    },
  });

  // 重平衡组合
  const rebalanceMutation = useMutation({
    mutationFn: ({ portfolioId }: { portfolioId: string }) => 
      apiClient.rebalancePortfolio(portfolioId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
      queryClient.invalidateQueries({ queryKey: ['portfolio-detail'] });
    },
  });

  // 显示组合详情
  const showPortfolioDetail = (portfolio: Portfolio) => {
    setSelectedPortfolio(portfolio);
    setDetailModalVisible(true);
  };

  // 处理创建组合
  const handleCreatePortfolio = (values: any) => {
    const requestData: PortfolioConstructRequest = {
      name: values.name,
      description: values.description,
      total_capital: values.total_capital,
      candidates: values.candidates?.map((item: any) => ({
        ticker: item.ticker,
        suggested_tier: item.tier,
        max_weight: item.max_weight / 100,
      })) || [],
      constraints: {
        max_single_position: values.max_single_position / 100,
        sector_concentration_limit: values.sector_limit / 100,
        min_diversification_score: values.min_diversification,
      },
      risk_tolerance: values.risk_tolerance,
    };
    
    createPortfolioMutation.mutate(requestData);
  };

  // 获取断路器状态颜色
  const getCircuitBreakerColor = (level: number) => {
    const colors = ['success', 'warning', 'error', 'error'];
    return colors[level] as 'success' | 'warning' | 'error';
  };

  // 获取断路器状态文本
  const getCircuitBreakerText = (level: number) => {
    const texts = ['正常', '一级预警', '二级断路', '三级断路'];
    return texts[level] || '未知';
  };

  // 组合列表表格列
  const portfolioColumns: ColumnsType<Portfolio> = [
    {
      title: '组合名称',
      key: 'name',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Text strong>{record.name}</Text>
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => showPortfolioDetail(record)}
            />
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.description}
          </Text>
        </Space>
      ),
    },
    {
      title: '总资产',
      dataIndex: 'current_value',
      key: 'current_value',
      width: 120,
      render: (value: number) => (
        <Statistic
          value={value}
          precision={0}
          prefix="¥"
          valueStyle={{ fontSize: 14 }}
          formatter={(val) => `${(Number(val) / 10000).toFixed(1)}万`}
        />
      ),
    },
    {
      title: '收益率',
      key: 'pnl',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text strong style={{ 
            color: record.total_pnl_pct >= 0 ? '#52c41a' : '#f5222d',
            fontSize: 16
          }}>
            {record.total_pnl_pct >= 0 ? '+' : ''}{(record.total_pnl_pct * 100).toFixed(2)}%
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.total_pnl >= 0 ? '+' : ''}¥{(record.total_pnl / 10000).toFixed(1)}万
          </Text>
        </Space>
      ),
    },
    {
      title: '最大回撤',
      dataIndex: 'max_drawdown',
      key: 'max_drawdown',
      width: 100,
      render: (drawdown: number) => (
        <Text style={{ color: Math.abs(drawdown) > 0.15 ? '#f5222d' : '#faad14' }}>
          {(drawdown * 100).toFixed(1)}%
        </Text>
      ),
    },
    {
      title: '夏普比率',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      width: 100,
      render: (ratio: number) => (
        <Text style={{ color: ratio > 1 ? '#52c41a' : ratio > 0.5 ? '#faad14' : '#f5222d' }}>
          {ratio.toFixed(2)}
        </Text>
      ),
    },
    {
      title: '持仓数',
      key: 'position_count',
      width: 80,
      render: (_, record) => (
        <Badge count={record.positions.length} style={{ backgroundColor: '#1890ff' }} />
      ),
    },
    {
      title: '断路器状态',
      key: 'circuit_breaker',
      width: 120,
      render: (_, record) => (
        <Badge 
          status={getCircuitBreakerColor(record.circuit_breaker_status.level)} 
          text={getCircuitBreakerText(record.circuit_breaker_status.level)}
        />
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 100,
      render: (time: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {dayjs(time).format('MM-DD')}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 120,
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="text" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => showPortfolioDetail(record)}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<ReloadOutlined />}
            loading={rebalanceMutation.isPending}
            onClick={() => rebalanceMutation.mutate({ portfolioId: record.portfolio_id })}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<SettingOutlined />}
          />
        </Space>
      ),
    },
  ];

  // 持仓明细表格列
  const positionColumns: ColumnsType<PortfolioPosition> = [
    {
      title: '股票',
      key: 'stock',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.ticker}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.company_name}
          </Text>
        </Space>
      ),
    },
    {
      title: '分层',
      dataIndex: 'tier',
      key: 'tier',
      render: (tier: string) => {
        const colors = { A: 'red', B: 'orange', C: 'blue' };
        return <Tag color={colors[tier as keyof typeof colors]}>{tier}层</Tag>;
      },
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      render: (weight: number) => (
        <Progress 
          percent={weight * 100} 
          format={(percent) => `${percent?.toFixed(1)}%`}
          size="small"
        />
      ),
    },
    {
      title: '持仓股数',
      dataIndex: 'shares',
      key: 'shares',
      render: (shares: number) => shares.toLocaleString(),
    },
    {
      title: '成本价',
      dataIndex: 'entry_price',
      key: 'entry_price',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '现价',
      dataIndex: 'current_price',
      key: 'current_price',
      render: (price: number) => `¥${price.toFixed(2)}`,
    },
    {
      title: '市值',
      dataIndex: 'market_value',
      key: 'market_value',
      render: (value: number) => `¥${(value / 10000).toFixed(1)}万`,
    },
    {
      title: '盈亏',
      key: 'pnl',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text style={{ 
            color: record.unrealized_pnl_pct >= 0 ? '#52c41a' : '#f5222d',
            fontWeight: 'bold'
          }}>
            {record.unrealized_pnl_pct >= 0 ? '+' : ''}{(record.unrealized_pnl_pct * 100).toFixed(2)}%
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.unrealized_pnl >= 0 ? '+' : ''}¥{(record.unrealized_pnl / 10000).toFixed(1)}万
          </Text>
        </Space>
      ),
    },
  ];

  // 组合分层分布饼图
  const tierDistributionOption = useMemo(() => {
    if (!selectedPortfolio) return {};
    
    const tiers = selectedPortfolio.tier_allocation;
    return {
      title: {
        text: '分层配置',
        left: 'center',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c}% ({d}%)'
      },
      series: [{
        name: '分层配置',
        type: 'pie',
        radius: ['40%', '70%'],
        data: [
          { value: tiers.A.current * 100, name: 'A层(核心)', itemStyle: { color: '#f5222d' } },
          { value: tiers.B.current * 100, name: 'B层(重要)', itemStyle: { color: '#faad14' } },
          { value: tiers.C.current * 100, name: 'C层(战术)', itemStyle: { color: '#1890ff' } },
        ],
        label: {
          formatter: '{b}\n{c}%'
        }
      }]
    };
  }, [selectedPortfolio]);

  // 收益趋势图
  const profitTrendOption = {
    title: {
      text: '收益趋势',
      textStyle: { fontSize: 14 }
    },
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 30 }, (_, i) => 
        dayjs().subtract(29 - i, 'day').format('MM/DD')
      )
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}%'
      }
    },
    series: [{
      name: '累计收益率',
      type: 'line',
      data: Array.from({ length: 30 }, () => 
        (Math.random() - 0.5) * 20 + (Math.random() * 10 - 5)
      ),
      smooth: true,
      itemStyle: { color: '#1890ff' },
      areaStyle: { opacity: 0.1 }
    }],
    grid: { left: 40, right: 20, top: 50, bottom: 30 }
  };

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
            <div>
              <Title level={2} style={{ margin: 0 }}>
                组合管理看板
              </Title>
              <Text type="secondary">
                基于A/B/C分层策略的投资组合构建与风险管理
              </Text>
            </div>
            <Button 
              type="primary" 
              size="large"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建组合
            </Button>
          </Space>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 统计概览 */}
          <Row gutter={[24, 16]}>
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="组合总数"
                  value={portfolioData?.data?.length || 0}
                  prefix={<PieChartOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="总资产"
                  value={portfolioData?.data?.reduce((sum, p) => sum + p.current_value, 0) || 0}
                  prefix={<DollarOutlined />}
                  formatter={(value) => `¥${(Number(value) / 100000000).toFixed(1)}亿`}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="平均收益率"
                  value={portfolioData?.data?.length ? 
                    (portfolioData.data.reduce((sum, p) => sum + p.total_pnl_pct, 0) / portfolioData.data.length * 100).toFixed(2) : 0
                  }
                  suffix="%"
                  prefix={<TrendingUpOutlined />}
                  valueStyle={{ color: '#fa8c16' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="风险预警"
                  value={portfolioData?.data?.filter(p => p.circuit_breaker_status.level > 0).length || 0}
                  prefix={<ExclamationCircleOutlined />}
                  valueStyle={{ color: '#f5222d' }}
                />
              </Card>
            </Col>
          </Row>

          {/* 组合列表 */}
          <Card 
            title={
              <Space>
                <BarChartOutlined />
                我的组合
              </Space>
            }
            extra={
              <Button 
                icon={<ReloadOutlined />}
                onClick={() => refetch()}
                loading={isLoading}
              >
                刷新
              </Button>
            }
          >
            <Table
              columns={portfolioColumns}
              dataSource={portfolioData?.data || []}
              rowKey="portfolio_id"
              loading={isLoading}
              scroll={{ x: 1000 }}
              size="middle"
              pagination={false}
            />
          </Card>
        </Space>

        {/* 创建组合弹窗 */}
        <Modal
          title="创建新组合"
          open={createModalVisible}
          onCancel={() => setCreateModalVisible(false)}
          width={800}
          footer={null}
        >
          <Form
            form={createForm}
            layout="vertical"
            onFinish={handleCreatePortfolio}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="name"
                  label="组合名称"
                  rules={[{ required: true, message: '请输入组合名称' }]}
                >
                  <Input placeholder="如：稳健成长组合" />
                </Form.Item>
              </Col>
              
              <Col span={12}>
                <Form.Item
                  name="total_capital"
                  label="总资金(万元)"
                  rules={[{ required: true, message: '请输入总资金' }]}
                >
                  <InputNumber 
                    placeholder="100" 
                    style={{ width: '100%' }}
                    min={1}
                    max={100000}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="description"
              label="组合描述"
            >
              <Input.TextArea placeholder="描述组合的投资理念和策略..." rows={3} />
            </Form.Item>

            <Divider orientation="left">风险约束</Divider>
            
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  name="max_single_position"
                  label="单一持仓上限(%)"
                  initialValue={15}
                >
                  <InputNumber 
                    min={1} 
                    max={30} 
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="sector_limit"
                  label="行业集中度上限(%)"
                  initialValue={40}
                >
                  <InputNumber 
                    min={10} 
                    max={80} 
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="risk_tolerance"
                  label="风险偏好"
                  initialValue="balanced"
                >
                  <Select>
                    <Option value="conservative">稳健</Option>
                    <Option value="balanced">平衡</Option>
                    <Option value="aggressive">激进</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Divider orientation="left">候选股票池</Divider>
            
            <Alert
              message="提示"
              description="系统将根据当前行业评分和个股质量自动筛选候选股票，您也可以手动添加特定股票"
              type="info"
              style={{ marginBottom: 16 }}
            />

            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => setCreateModalVisible(false)}>
                取消
              </Button>
              <Button 
                type="primary" 
                htmlType="submit"
                loading={createPortfolioMutation.isPending}
              >
                创建组合
              </Button>
            </Space>
          </Form>
        </Modal>

        {/* 组合详情弹窗 */}
        <Modal
          title={selectedPortfolio ? `${selectedPortfolio.name} - 详细信息` : '组合详情'}
          open={detailModalVisible}
          onCancel={() => setDetailModalVisible(false)}
          width={1200}
          footer={[
            <Button key="close" onClick={() => setDetailModalVisible(false)}>
              关闭
            </Button>,
            <Button 
              key="rebalance" 
              type="primary"
              icon={<ReloadOutlined />}
              loading={rebalanceMutation.isPending}
              onClick={() => selectedPortfolio && 
                rebalanceMutation.mutate({ portfolioId: selectedPortfolio.portfolio_id })
              }
            >
              重新平衡
            </Button>,
          ]}
        >
          {portfolioDetail?.data && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {/* 组合概况 */}
              <Row gutter={[16, 16]}>
                <Col xs={24} md={8}>
                  <Card size="small" title="基本信息">
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="总资产">
                        ¥{(portfolioDetail.data.current_value / 10000).toFixed(1)}万
                      </Descriptions.Item>
                      <Descriptions.Item label="总收益">
                        <Text style={{ 
                          color: portfolioDetail.data.total_pnl >= 0 ? '#52c41a' : '#f5222d',
                          fontWeight: 'bold'
                        }}>
                          {portfolioDetail.data.total_pnl >= 0 ? '+' : ''}
                          {(portfolioDetail.data.total_pnl_pct * 100).toFixed(2)}%
                        </Text>
                      </Descriptions.Item>
                      <Descriptions.Item label="最大回撤">
                        <Text style={{ color: '#f5222d' }}>
                          {(portfolioDetail.data.max_drawdown * 100).toFixed(2)}%
                        </Text>
                      </Descriptions.Item>
                      <Descriptions.Item label="夏普比率">
                        {portfolioDetail.data.sharpe_ratio.toFixed(3)}
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
                
                <Col xs={24} md={8}>
                  <Card size="small" title="风险指标">
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="VaR(95%)">
                        {(portfolioDetail.data.risk_metrics.var_95 * 100).toFixed(2)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="Beta系数">
                        {portfolioDetail.data.risk_metrics.beta.toFixed(3)}
                      </Descriptions.Item>
                      <Descriptions.Item label="波动率">
                        {(portfolioDetail.data.risk_metrics.volatility * 100).toFixed(2)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="断路器">
                        <Badge 
                          status={getCircuitBreakerColor(portfolioDetail.data.circuit_breaker_status.level)} 
                          text={getCircuitBreakerText(portfolioDetail.data.circuit_breaker_status.level)}
                        />
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
                
                <Col xs={24} md={8}>
                  <Card size="small" title="分层配置">
                    <ReactECharts option={tierDistributionOption} style={{ height: '200px' }} />
                  </Card>
                </Col>
              </Row>

              {/* 收益趋势图 */}
              <Card title="收益趋势" size="small">
                <ReactECharts option={profitTrendOption} style={{ height: '300px' }} />
              </Card>

              {/* 持仓明细 */}
              <Card title="持仓明细" size="small">
                <Table
                  columns={positionColumns}
                  dataSource={portfolioDetail.data.positions}
                  rowKey="ticker"
                  size="small"
                  pagination={false}
                  scroll={{ x: 800 }}
                />
              </Card>
            </Space>
          )}
        </Modal>
      </div>
    </MainLayout>
  );
}