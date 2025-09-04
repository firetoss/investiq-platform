/**
 * InvestIQ Platform - 行业评分台页面
 * 提供行业评分分析和比较功能
 */

import React, { useState, useMemo } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Table, 
  Select, 
  Button, 
  Space, 
  Typography, 
  Tag,
  Progress,
  Tooltip,
  Radio,
  DatePicker,
  Statistic,
  Input
} from 'antd';
import { 
  SearchOutlined,
  ReloadOutlined,
  BarChartOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  MinusOutlined,
  InfoCircleOutlined
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { IndustryScore } from '@/types/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';
import type { ColumnsType } from 'antd/es/table';
import type { Dayjs } from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

export default function IndustriesPage() {
  const [selectedIndustries, setSelectedIndustries] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [dateRange, setDateRange] = useState<[Dayjs, Dayjs]>([
    dayjs().subtract(7, 'days'),
    dayjs()
  ]);
  const [searchText, setSearchText] = useState('');
  const [sortBy, setSortBy] = useState<'overall_score' | 'policy_score' | 'growth_score' | 'valuation_score' | 'technical_score'>('overall_score');

  // 获取行业评分数据
  const { data: industryData, isLoading, refetch } = useQuery({
    queryKey: ['industry-scores', dateRange],
    queryFn: () => apiClient.getIndustryScores({
      days: dayjs().diff(dateRange[0], 'days'),
    }),
    refetchInterval: 5 * 60 * 1000, // 5分钟刷新
  });

  // 获取行业比较数据
  const { data: comparisonData } = useQuery({
    queryKey: ['industry-comparison', selectedIndustries],
    queryFn: () => selectedIndustries.length > 1 ? 
      apiClient.compareIndustries(selectedIndustries) : null,
    enabled: selectedIndustries.length > 1,
  });

  // 过滤和排序数据
  const filteredIndustries = useMemo(() => {
    let data = industryData?.data || [];
    
    // 搜索过滤
    if (searchText) {
      data = data.filter(item => 
        item.industry_name.toLowerCase().includes(searchText.toLowerCase()) ||
        item.industry_code.toLowerCase().includes(searchText.toLowerCase())
      );
    }
    
    // 排序
    data.sort((a, b) => b[sortBy] - a[sortBy]);
    
    return data;
  }, [industryData?.data, searchText, sortBy]);

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

  // 获取趋势图标
  const getTrendIcon = (current: number, previous: number = current - 5) => {
    if (current > previous) {
      return <TrendingUpOutlined style={{ color: '#52c41a' }} />;
    } else if (current < previous) {
      return <TrendingDownOutlined style={{ color: '#f5222d' }} />;
    }
    return <MinusOutlined style={{ color: '#8c8c8c' }} />;
  };

  // 表格列配置
  const columns: ColumnsType<IndustryScore> = [
    {
      title: '行业',
      dataIndex: 'industry_name',
      key: 'industry_name',
      fixed: 'left',
      width: 120,
      render: (text, record) => (
        <Space direction="vertical" size={0}>
          <Text strong>{text}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
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
          <Text strong style={{ fontSize: 18, color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
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
      title: '政策评分',
      dataIndex: 'policy_score',
      key: 'policy_score',
      width: 100,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(1)}
        </Text>
      ),
    },
    {
      title: '成长评分',
      dataIndex: 'growth_score',
      key: 'growth_score',
      width: 100,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(1)}
        </Text>
      ),
    },
    {
      title: '估值评分',
      dataIndex: 'valuation_score',
      key: 'valuation_score',
      width: 100,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(1)}
        </Text>
      ),
    },
    {
      title: '技术评分',
      dataIndex: 'technical_score',
      key: 'technical_score',
      width: 100,
      render: (score: number) => (
        <Text style={{ color: score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d' }}>
          {score.toFixed(1)}
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
      title: '趋势',
      key: 'trend',
      width: 80,
      render: (_, record) => getTrendIcon(record.overall_score),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 120,
      render: (time: string) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {dayjs(time).format('MM-DD HH:mm')}
        </Text>
      ),
    },
  ];

  // 雷达图配置
  const radarChartOption = useMemo(() => {
    if (!selectedIndustries.length || !industryData?.data) return {};
    
    const selectedData = industryData.data.filter(item => 
      selectedIndustries.includes(item.industry_code)
    );

    return {
      title: {
        text: '行业评分雷达图对比',
        textStyle: { fontSize: 16, fontWeight: 'normal' }
      },
      tooltip: {
        trigger: 'item'
      },
      legend: {
        data: selectedData.map(item => item.industry_name),
        bottom: 0
      },
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
        data: selectedData.map((item, index) => ({
          value: [
            item.policy_score,
            item.growth_score,
            item.valuation_score,
            item.technical_score
          ],
          name: item.industry_name,
          itemStyle: { color: ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1'][index % 5] }
        }))
      }]
    };
  }, [selectedIndustries, industryData?.data]);

  // 分布图配置
  const scatterChartOption = useMemo(() => {
    if (!industryData?.data) return {};

    return {
      title: {
        text: '行业评分分布图',
        textStyle: { fontSize: 16, fontWeight: 'normal' }
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          const data = industryData.data[params.dataIndex];
          return `${data.industry_name}<br/>综合评分: ${data.overall_score.toFixed(1)}<br/>置信度: ${data.confidence_level.toFixed(1)}%`;
        }
      },
      xAxis: {
        name: '综合评分',
        nameLocation: 'middle',
        nameGap: 25,
        type: 'value',
        min: 0,
        max: 100
      },
      yAxis: {
        name: '置信度',
        nameLocation: 'middle',
        nameGap: 40,
        type: 'value',
        min: 0,
        max: 100
      },
      series: [{
        type: 'scatter',
        data: industryData.data.map((item, index) => [
          item.overall_score,
          item.confidence_level,
          item.industry_name
        ]),
        symbolSize: (data: any) => Math.max(data[0] / 5, 10),
        itemStyle: {
          color: (params: any) => {
            const score = params.data[0];
            return score >= 80 ? '#52c41a' : score >= 60 ? '#faad14' : '#f5222d';
          }
        }
      }]
    };
  }, [industryData?.data]);

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Title level={2} style={{ margin: 0 }}>
            行业评分台
          </Title>
          <Text type="secondary">
            基于四门方法论的行业投资价值评分与分析
          </Text>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 统计概览 */}
          <Row gutter={[24, 16]}>
            <Col xs={24} sm={8}>
              <Card size="small">
                <Statistic
                  title="行业总数"
                  value={filteredIndustries.length}
                  prefix={<BarChartOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card size="small">
                <Statistic
                  title="平均评分"
                  value={filteredIndustries.length ? 
                    (filteredIndustries.reduce((sum, item) => sum + item.overall_score, 0) / filteredIndustries.length).toFixed(1) : 0
                  }
                  precision={1}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={8}>
              <Card size="small">
                <Statistic
                  title="强买推荐"
                  value={filteredIndustries.filter(item => item.recommendation === 'STRONG_BUY').length}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
          </Row>

          {/* 工具栏 */}
          <Card size="small">
            <Row gutter={[16, 16]} align="middle">
              <Col xs={24} sm={12} md={6}>
                <Input
                  placeholder="搜索行业"
                  prefix={<SearchOutlined />}
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  allowClear
                />
              </Col>
              
              <Col xs={24} sm={12} md={4}>
                <Select
                  value={sortBy}
                  onChange={setSortBy}
                  style={{ width: '100%' }}
                >
                  <Option value="overall_score">综合评分</Option>
                  <Option value="policy_score">政策评分</Option>
                  <Option value="growth_score">成长评分</Option>
                  <Option value="valuation_score">估值评分</Option>
                  <Option value="technical_score">技术评分</Option>
                </Select>
              </Col>
              
              <Col xs={24} sm={12} md={6}>
                <RangePicker
                  value={dateRange}
                  onChange={(dates) => dates && setDateRange(dates as [Dayjs, Dayjs])}
                  size="middle"
                  style={{ width: '100%' }}
                />
              </Col>
              
              <Col xs={24} sm={12} md={4}>
                <Radio.Group 
                  value={viewMode} 
                  onChange={(e) => setViewMode(e.target.value)}
                  buttonStyle="solid"
                  size="middle"
                >
                  <Radio.Button value="table">表格</Radio.Button>
                  <Radio.Button value="chart">图表</Radio.Button>
                </Radio.Group>
              </Col>
              
              <Col xs={24} sm={12} md={4}>
                <Button 
                  icon={<ReloadOutlined />} 
                  onClick={() => refetch()}
                  loading={isLoading}
                  style={{ width: '100%' }}
                >
                  刷新
                </Button>
              </Col>
            </Row>
          </Card>

          {/* 主要内容 */}
          {viewMode === 'table' ? (
            <Card>
              <Table
                columns={columns}
                dataSource={filteredIndustries}
                rowKey="industry_code"
                loading={isLoading}
                scroll={{ x: 1200 }}
                size="middle"
                pagination={{
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 个行业`,
                }}
                rowSelection={{
                  selectedRowKeys: selectedIndustries,
                  onChange: (keys) => setSelectedIndustries(keys as string[]),
                  getCheckboxProps: (record) => ({
                    name: record.industry_name,
                  }),
                }}
              />
            </Card>
          ) : (
            <Row gutter={[24, 24]}>
              <Col xs={24} lg={12}>
                <Card title="行业对比雷达图">
                  {selectedIndustries.length > 0 ? (
                    <ReactECharts option={radarChartOption} style={{ height: '400px' }} />
                  ) : (
                    <div style={{ textAlign: 'center', padding: '100px 0', color: '#999' }}>
                      请在表格中选择要对比的行业
                    </div>
                  )}
                </Card>
              </Col>
              
              <Col xs={24} lg={12}>
                <Card title="评分分布散点图">
                  <ReactECharts option={scatterChartOption} style={{ height: '400px' }} />
                </Card>
              </Col>
            </Row>
          )}
        </Space>
      </div>
    </MainLayout>
  );
}