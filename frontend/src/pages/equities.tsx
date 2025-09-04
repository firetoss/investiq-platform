/**
 * InvestIQ Platform - 个股筛选页面
 * 提供个股评分筛选和分析功能
 */

import React, { useState, useMemo } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Form,
  Input,
  Select, 
  Button, 
  Space, 
  Typography, 
  Tag,
  Slider,
  InputNumber,
  Collapse,
  Tooltip,
  Badge,
  Modal,
  Descriptions,
  Progress
} from 'antd';
import { 
  SearchOutlined,
  FilterOutlined,
  ReloadOutlined,
  EyeOutlined,
  StarOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  InfoCircleOutlined,
  DownloadOutlined,
  PlusOutlined
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { EquityScore, EquityScreeningRequest } from '@/types/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Option } = Select;
const { Panel } = Collapse;

export default function EquitiesPage() {
  const [form] = Form.useForm();
  const [selectedEquities, setSelectedEquities] = useState<string[]>([]);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedEquity, setSelectedEquity] = useState<EquityScore | null>(null);
  const [screeningParams, setScreeningParams] = useState<EquityScreeningRequest>({
    page: 1,
    page_size: 50,
    sort_by: 'overall_score',
    sort_order: 'desc',
  });

  // 获取筛选结果
  const { data: equityData, isLoading, refetch } = useQuery({
    queryKey: ['equity-screening', screeningParams],
    queryFn: () => apiClient.screenEquities(screeningParams),
    refetchInterval: 5 * 60 * 1000,
  });

  // 获取行业列表（用于筛选）
  const { data: industryOptions } = useQuery({
    queryKey: ['industry-options'],
    queryFn: () => apiClient.getIndustryScores({ days: 1 }),
  });

  // 获取个股详情
  const { data: equityDetail } = useQuery({
    queryKey: ['equity-detail', selectedEquity?.ticker],
    queryFn: () => selectedEquity ? apiClient.getEquityScore(selectedEquity.ticker) : null,
    enabled: !!selectedEquity,
  });

  // 处理筛选
  const handleScreen = (values: any) => {
    const filters: EquityScreeningRequest = {
      ...screeningParams,
      page: 1,
      industry_codes: values.industry_codes,
      min_score: values.score_range?.[0],
      max_score: values.score_range?.[1],
      recommendations: values.recommendations,
      market_caps: {
        min: values.market_cap_range?.[0] ? values.market_cap_range[0] * 100000000 : undefined,
        max: values.market_cap_range?.[1] ? values.market_cap_range[1] * 100000000 : undefined,
      },
      financial_filters: {
        min_revenue_growth: values.min_revenue_growth,
        max_pe_ratio: values.max_pe_ratio,
        min_roe: values.min_roe,
      },
    };
    setScreeningParams(filters);
  };

  // 重置筛选
  const handleReset = () => {
    form.resetFields();
    setScreeningParams({
      page: 1,
      page_size: 50,
      sort_by: 'overall_score',
      sort_order: 'desc',
    });
  };

  // 显示个股详情
  const showEquityDetail = (equity: EquityScore) => {
    setSelectedEquity(equity);
    setDetailModalVisible(true);
  };

  // 获取推荐标签颜色
  const getRecommendationColor = (recommendation: string) => {
    const colors = {
      'STRONG_BUY': '#52c41a',
      'BUY': '#1890ff',
      'HOLD': '#faad14',
      'SELL': '#f5222d',
      'STRONG_SELL': '#a8071a',
    };
    return colors[recommendation as keyof typeof colors] || '#d9d9d9';
  };

  // 表格列配置
  const columns: ColumnsType<EquityScore> = [
    {
      title: '股票',
      key: 'stock',
      fixed: 'left',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Text strong>{record.ticker}</Text>
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => showEquityDetail(record)}
            />
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.company_name.length > 12 ? 
              `${record.company_name.substring(0, 12)}...` : 
              record.company_name
            }
          </Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {record.industry_code}
          </Text>
        </Space>
      ),
    },
    {
      title: (
        <Space>
          综合评分
          <Tooltip title="综合四个维度的加权评分">
            <InfoCircleOutlined />
          </Tooltip>
        </Space>
      ),
      dataIndex: 'overall_score',
      key: 'overall_score',
      width: 120,
      sorter: true,
      render: (score: number) => (
        <Space direction="vertical" size={0} style={{ textAlign: 'center' }}>
          <Text strong style={{ 
            fontSize: 16, 
            color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' 
          }}>
            {score.toFixed(1)}
          </Text>
          <Progress
            percent={score}
            size="small"
            showInfo={false}
            strokeColor={score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d'}
          />
        </Space>
      ),
    },
    {
      title: '政策',
      dataIndex: 'policy_score',
      key: 'policy_score',
      width: 80,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(0)}
        </Text>
      ),
    },
    {
      title: '成长',
      dataIndex: 'growth_score',
      key: 'growth_score',
      width: 80,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(0)}
        </Text>
      ),
    },
    {
      title: '估值',
      dataIndex: 'valuation_score',
      key: 'valuation_score',
      width: 80,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(0)}
        </Text>
      ),
    },
    {
      title: '技术',
      dataIndex: 'technical_score',
      key: 'technical_score',
      width: 80,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(0)}
        </Text>
      ),
    },
    {
      title: '投资建议',
      dataIndex: 'recommendation',
      key: 'recommendation',
      width: 100,
      render: (recommendation: string) => {
        const textMap = {
          'STRONG_BUY': '强买',
          'BUY': '买入',
          'HOLD': '持有',
          'SELL': '卖出',
          'STRONG_SELL': '强卖',
        };
        return (
          <Tag color={getRecommendationColor(recommendation)}>
            {textMap[recommendation as keyof typeof textMap] || recommendation}
          </Tag>
        );
      },
    },
    {
      title: '置信度',
      dataIndex: 'confidence_level',
      key: 'confidence_level',
      width: 100,
      render: (confidence: number) => (
        <Progress
          percent={confidence}
          size="small"
          format={(percent) => `${percent}%`}
          strokeColor={confidence >= 80 ? '#52c41a' : confidence >= 60 ? '#faad14' : '#f5222d'}
        />
      ),
    },
    {
      title: '关键指标',
      key: 'key_metrics',
      width: 150,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: 11 }}>
            PE: {record.key_metrics?.pe_ratio?.toFixed(1) || 'N/A'}
          </Text>
          <Text style={{ fontSize: 11 }}>
            PB: {record.key_metrics?.pb_ratio?.toFixed(1) || 'N/A'}
          </Text>
          <Text style={{ fontSize: 11 }}>
            ROE: {record.key_metrics?.roe ? `${(record.key_metrics.roe * 100).toFixed(1)}%` : 'N/A'}
          </Text>
        </Space>
      ),
    },
    {
      title: '风险',
      dataIndex: 'risk_factors',
      key: 'risk_factors',
      width: 120,
      render: (risks: string[]) => (
        <Space direction="vertical" size={0}>
          {risks?.slice(0, 2).map((risk, index) => (
            <Badge key={index} status="warning" text={
              <Text style={{ fontSize: 11 }}>
                {risk.length > 8 ? `${risk.substring(0, 8)}...` : risk}
              </Text>
            } />
          ))}
          {risks?.length > 2 && (
            <Text type="secondary" style={{ fontSize: 10 }}>
              +{risks.length - 2} 个
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 100,
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="text" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => showEquityDetail(record)}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<StarOutlined />}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<PlusOutlined />}
          />
        </Space>
      ),
    },
  ];

  // 个股详情雷达图
  const equityRadarOption = useMemo(() => {
    if (!equityDetail?.data) return {};
    
    const data = equityDetail.data;
    return {
      title: {
        text: `${data.company_name} 评分分析`,
        textStyle: { fontSize: 16 }
      },
      tooltip: {},
      radar: {
        indicator: [
          { name: '政策', max: 100 },
          { name: '成长', max: 100 },
          { name: '估值', max: 100 },
          { name: '技术', max: 100 }
        ]
      },
      series: [{
        type: 'radar',
        data: [{
          value: [
            data.policy_score,
            data.growth_score,
            data.valuation_score,
            data.technical_score
          ],
          name: data.ticker,
          itemStyle: { color: '#1890ff' },
          areaStyle: { opacity: 0.3 }
        }]
      }]
    };
  }, [equityDetail?.data]);

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Title level={2} style={{ margin: 0 }}>
            个股筛选
          </Title>
          <Text type="secondary">
            基于四门方法论的个股投资价值筛选与分析
          </Text>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 筛选工具栏 */}
          <Card size="small" title={
            <Space>
              <FilterOutlined />
              筛选条件
            </Space>
          }>
            <Form 
              form={form} 
              onFinish={handleScreen}
              layout="vertical"
            >
              <Row gutter={[16, 16]}>
                <Col xs={24} sm={12} md={6}>
                  <Form.Item name="industry_codes" label="行业选择">
                    <Select
                      mode="multiple"
                      placeholder="选择行业"
                      allowClear
                      maxTagCount={2}
                    >
                      {industryOptions?.data?.map(industry => (
                        <Option key={industry.industry_code} value={industry.industry_code}>
                          {industry.industry_name}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
                
                <Col xs={24} sm={12} md={6}>
                  <Form.Item name="score_range" label="评分范围">
                    <Slider
                      range
                      min={0}
                      max={100}
                      marks={{0: '0', 50: '50', 100: '100'}}
                      tooltip={{ formatter: (value) => `${value}分` }}
                    />
                  </Form.Item>
                </Col>
                
                <Col xs={24} sm={12} md={6}>
                  <Form.Item name="recommendations" label="投资建议">
                    <Select
                      mode="multiple"
                      placeholder="选择建议"
                      allowClear
                    >
                      <Option value="STRONG_BUY">强买</Option>
                      <Option value="BUY">买入</Option>
                      <Option value="HOLD">持有</Option>
                      <Option value="SELL">卖出</Option>
                      <Option value="STRONG_SELL">强卖</Option>
                    </Select>
                  </Form.Item>
                </Col>
                
                <Col xs={24} sm={12} md={6}>
                  <Form.Item name="market_cap_range" label="市值范围(亿元)">
                    <Slider
                      range
                      min={0}
                      max={5000}
                      step={50}
                      marks={{0: '0', 1000: '1000', 5000: '5000+'}}
                      tooltip={{ formatter: (value) => `${value}亿` }}
                    />
                  </Form.Item>
                </Col>
              </Row>
              
              <Collapse ghost>
                <Panel header="高级筛选" key="advanced">
                  <Row gutter={[16, 16]}>
                    <Col xs={24} sm={8}>
                      <Form.Item name="min_revenue_growth" label="最小营收增长率(%)">
                        <InputNumber 
                          placeholder="如: 15"
                          style={{ width: '100%' }}
                          min={-100}
                          max={1000}
                        />
                      </Form.Item>
                    </Col>
                    
                    <Col xs={24} sm={8}>
                      <Form.Item name="max_pe_ratio" label="最大PE倍数">
                        <InputNumber 
                          placeholder="如: 30"
                          style={{ width: '100%' }}
                          min={0}
                          max={500}
                        />
                      </Form.Item>
                    </Col>
                    
                    <Col xs={24} sm={8}>
                      <Form.Item name="min_roe" label="最小ROE(%)">
                        <InputNumber 
                          placeholder="如: 10"
                          style={{ width: '100%' }}
                          min={0}
                          max={100}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>
              </Collapse>
              
              <Space>
                <Button type="primary" htmlType="submit" loading={isLoading}>
                  开始筛选
                </Button>
                <Button onClick={handleReset}>
                  重置条件
                </Button>
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={() => refetch()}
                  loading={isLoading}
                >
                  刷新数据
                </Button>
                <Button icon={<DownloadOutlined />}>
                  导出结果
                </Button>
              </Space>
            </Form>
          </Card>

          {/* 筛选结果 */}
          <Card title={
            <Space>
              筛选结果
              <Badge 
                count={equityData?.data?.pagination?.total_count || 0} 
                style={{ backgroundColor: '#1890ff' }}
              />
            </Space>
          }>
            <Table
              columns={columns}
              dataSource={equityData?.data?.items || []}
              rowKey="ticker"
              loading={isLoading}
              scroll={{ x: 1400 }}
              size="middle"
              pagination={{
                current: equityData?.data?.pagination?.page,
                pageSize: equityData?.data?.pagination?.page_size,
                total: equityData?.data?.pagination?.total_count,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => 
                  `第 ${range[0]}-${range[1]} 条，共 ${total} 只股票`,
                onChange: (page, pageSize) => {
                  setScreeningParams(prev => ({ ...prev, page, page_size: pageSize }));
                }
              }}
              rowSelection={{
                selectedRowKeys: selectedEquities,
                onChange: (keys) => setSelectedEquities(keys as string[]),
              }}
            />
          </Card>
        </Space>

        {/* 个股详情弹窗 */}
        <Modal
          title={selectedEquity ? `${selectedEquity.ticker} - ${selectedEquity.company_name}` : '个股详情'}
          open={detailModalVisible}
          onCancel={() => setDetailModalVisible(false)}
          width={800}
          footer={[
            <Button key="close" onClick={() => setDetailModalVisible(false)}>
              关闭
            </Button>,
            <Button key="add" type="primary" icon={<PlusOutlined />}>
              加入组合
            </Button>,
          ]}
        >
          {equityDetail?.data && (
            <Row gutter={[24, 24]}>
              <Col xs={24} md={12}>
                <ReactECharts option={equityRadarOption} style={{ height: '300px' }} />
              </Col>
              
              <Col xs={24} md={12}>
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="行业">{equityDetail.data.industry_code}</Descriptions.Item>
                  <Descriptions.Item label="综合评分">
                    <Text strong style={{ 
                      color: equityDetail.data.overall_score >= 80 ? '#52c41a' : 
                            equityDetail.data.overall_score >= 60 ? '#faad14' : '#f5222d' 
                    }}>
                      {equityDetail.data.overall_score.toFixed(1)}
                    </Text>
                  </Descriptions.Item>
                  <Descriptions.Item label="投资建议">
                    <Tag color={getRecommendationColor(equityDetail.data.recommendation)}>
                      {equityDetail.data.recommendation}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="置信度">
                    {equityDetail.data.confidence_level.toFixed(1)}%
                  </Descriptions.Item>
                </Descriptions>

                <Title level={5} style={{ marginTop: 16 }}>关键财务指标</Title>
                <Descriptions column={2} size="small">
                  <Descriptions.Item label="PE">
                    {equityDetail.data.key_metrics?.pe_ratio?.toFixed(2) || 'N/A'}
                  </Descriptions.Item>
                  <Descriptions.Item label="PB">
                    {equityDetail.data.key_metrics?.pb_ratio?.toFixed(2) || 'N/A'}
                  </Descriptions.Item>
                  <Descriptions.Item label="ROE">
                    {equityDetail.data.key_metrics?.roe ? 
                      `${(equityDetail.data.key_metrics.roe * 100).toFixed(1)}%` : 'N/A'}
                  </Descriptions.Item>
                  <Descriptions.Item label="ROA">
                    {equityDetail.data.key_metrics?.roa ? 
                      `${(equityDetail.data.key_metrics.roa * 100).toFixed(1)}%` : 'N/A'}
                  </Descriptions.Item>
                </Descriptions>
              </Col>
              
              <Col xs={24}>
                <Title level={5}>风险因素</Title>
                <Space wrap>
                  {equityDetail.data.risk_factors?.map((risk, index) => (
                    <Tag key={index} color="orange" style={{ marginBottom: 4 }}>
                      {risk}
                    </Tag>
                  ))}
                </Space>
              </Col>
            </Row>
          )}
        </Modal>
      </div>
    </MainLayout>
  );
}