/**
 * InvestIQ Platform - 首页概览面板
 * 提供系统整体概览和关键指标
 */

import React from 'react';
import { Row, Col, Card, Statistic, Progress, List, Badge, Avatar, Button, Space, Typography } from 'antd';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined,
  TrendingUpOutlined,
  AlertOutlined,
  BarChartOutlined,
  StockOutlined
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';

const { Title, Text } = Typography;

export default function HomePage() {
  // 获取系统健康状态
  const { data: healthData } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 30000, // 30秒刷新一次
  });

  // 获取告警仪表板数据
  const { data: alertDashboard } = useQuery({
    queryKey: ['alert-dashboard'],
    queryFn: () => apiClient.getAlertDashboard(7),
  });

  // 获取行业评分数据
  const { data: industryScores } = useQuery({
    queryKey: ['industry-scores-summary'],
    queryFn: () => apiClient.getIndustryScores({ days: 1 }),
  });

  // 获取组合数据
  const { data: portfolioData } = useQuery({
    queryKey: ['portfolios-summary'],
    queryFn: () => apiClient.getPortfolios(),
  });

  // 模拟市场概览数据
  const marketOverview = {
    上证指数: { value: 3245.67, change: 1.24, changePercent: 0.038 },
    深证成指: { value: 10234.56, change: -15.23, changePercent: -0.015 },
    创业板指: { value: 2156.78, change: 8.45, changePercent: 0.039 },
    恒生指数: { value: 18765.43, change: -45.67, changePercent: -0.024 },
  };

  // 趋势图配置
  const trendChartOption = {
    title: {
      text: '近7日告警趋势',
      textStyle: { fontSize: 14, fontWeight: 'normal' }
    },
    tooltip: {
      trigger: 'axis',
    },
    xAxis: {
      type: 'category',
      data: Array.from({ length: 7 }, (_, i) => dayjs().subtract(6 - i, 'day').format('MM/DD')),
    },
    yAxis: {
      type: 'value',
    },
    series: [
      {
        name: '告警数量',
        type: 'line',
        data: [12, 8, 15, 6, 10, 14, 9],
        smooth: true,
        itemStyle: { color: '#1890ff' },
        areaStyle: { opacity: 0.1 },
      },
    ],
    grid: { left: 40, right: 20, top: 50, bottom: 30 },
  };

  // 行业热力图配置
  const industryHeatmapOption = {
    title: {
      text: '行业评分热力图',
      textStyle: { fontSize: 14, fontWeight: 'normal' }
    },
    tooltip: {
      position: 'top',
      formatter: (params: any) => `${params.data[2]}: ${params.data[3]}分`
    },
    grid: {
      height: '70%',
      top: '10%'
    },
    xAxis: {
      type: 'category',
      data: ['政策', '成长', '估值', '技术'],
      splitArea: { show: true }
    },
    yAxis: {
      type: 'category',
      data: ['新能源', '医药', '消费', '科技', '金融'],
      splitArea: { show: true }
    },
    visualMap: {
      min: 0,
      max: 100,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: '5%'
    },
    series: [{
      name: '行业评分',
      type: 'heatmap',
      data: [
        [0, 0, '新能源-政策', 85],
        [1, 0, '新能源-成长', 78],
        [2, 0, '新能源-估值', 65],
        [3, 0, '新能源-技术', 72],
        [0, 1, '医药-政策', 70],
        [1, 1, '医药-成长', 82],
        [2, 1, '医药-估值', 88],
        [3, 1, '医药-技术', 75],
        // ... 更多数据
      ],
      label: {
        show: true
      }
    }]
  };

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Title level={2} style={{ margin: 0 }}>
            投资决策平台概览
          </Title>
          <Text type="secondary">
            实时监控投资决策的关键指标和系统状态
          </Text>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 系统状态统计 */}
          <Row gutter={[24, 24]}>
            <Col xs={24} sm={12} lg={6}>
              <Card className="stat-card">
                <Statistic
                  title="行业覆盖"
                  value={28}
                  suffix="个"
                  valueStyle={{ color: '#3f8600' }}
                  prefix={<BarChartOutlined />}
                />
                <Progress 
                  percent={93} 
                  size="small" 
                  showInfo={false}
                  strokeColor="#52c41a" 
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  较昨日 +2
                </Text>
              </Card>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <Card className="stat-card">
                <Statistic
                  title="个股追踪"
                  value={1247}
                  suffix="只"
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<StockOutlined />}
                />
                <Progress 
                  percent={87} 
                  size="small" 
                  showInfo={false}
                  strokeColor="#1890ff" 
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  数据覆盖率 87%
                </Text>
              </Card>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <Card className="stat-card">
                <Statistic
                  title="组合数量"
                  value={portfolioData?.data?.length || 0}
                  suffix="个"
                  valueStyle={{ color: '#722ed1' }}
                  prefix={<TrendingUpOutlined />}
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  总资产 ¥2.4亿
                </Text>
              </Card>
            </Col>
            
            <Col xs={24} sm={12} lg={6}>
              <Card className="stat-card">
                <Statistic
                  title="活跃告警"
                  value={alertDashboard?.data?.basic_stats?.new_alerts || 0}
                  suffix="条"
                  valueStyle={{ color: '#f5222d' }}
                  prefix={<AlertOutlined />}
                />
                <Text type="secondary" style={{ fontSize: 12 }}>
                  待处理 {alertDashboard?.data?.basic_stats?.new_alerts || 0} 条
                </Text>
              </Card>
            </Col>
          </Row>

          {/* 市场概览 */}
          <Card title="市场概览" size="small">
            <Row gutter={[16, 16]}>
              {Object.entries(marketOverview).map(([index, data]) => (
                <Col xs={12} sm={6} key={index}>
                  <Card size="small" style={{ textAlign: 'center' }}>
                    <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 4 }}>
                      {index}
                    </div>
                    <div style={{ fontSize: 18, fontWeight: 'bold', marginBottom: 4 }}>
                      {data.value.toLocaleString()}
                    </div>
                    <div style={{ 
                      color: data.change >= 0 ? '#52c41a' : '#f5222d',
                      fontSize: 12
                    }}>
                      {data.change >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                      {Math.abs(data.change)} ({Math.abs(data.changePercent * 100).toFixed(2)}%)
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>

          {/* 图表区域 */}
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card>
                <ReactECharts 
                  option={trendChartOption} 
                  style={{ height: '300px' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card>
                <ReactECharts 
                  option={industryHeatmapOption} 
                  style={{ height: '300px' }}
                />
              </Card>
            </Col>
          </Row>

          {/* 最新动态 */}
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card 
                title="最新告警" 
                size="small"
                extra={<Button type="link" size="small">查看全部</Button>}
              >
                <List
                  size="small"
                  dataSource={[
                    {
                      id: '1',
                      title: '新能源行业政策评分下降',
                      description: '受政策调整影响，评分从85分降至78分',
                      time: '2分钟前',
                      severity: 'P2',
                    },
                    {
                      id: '2',
                      title: '医药组合回撤超过阈值',
                      description: '当前回撤-12.5%，触发二级断路器',
                      time: '15分钟前',
                      severity: 'P1',
                    },
                    {
                      id: '3',
                      title: '科技股集中度预警',
                      description: '科技板块仓位占比达到35%',
                      time: '1小时前',
                      severity: 'P3',
                    },
                  ]}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Badge 
                            status={
                              item.severity === 'P1' ? 'error' : 
                              item.severity === 'P2' ? 'warning' : 'default'
                            } 
                          />
                        }
                        title={item.title}
                        description={
                          <Space direction="vertical" size={0}>
                            <Text type="secondary" style={{ fontSize: 12 }}>
                              {item.description}
                            </Text>
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              {item.time}
                            </Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card 
                title="系统状态" 
                size="small"
                extra={
                  <Badge 
                    status={healthData?.status === 'healthy' ? 'success' : 'error'} 
                    text={healthData?.status === 'healthy' ? '正常运行' : '异常'}
                  />
                }
              >
                <List
                  size="small"
                  dataSource={[
                    { name: '评分服务', status: 'success', response: '45ms' },
                    { name: '组合服务', status: 'success', response: '32ms' },
                    { name: '告警服务', status: 'success', response: '28ms' },
                    { name: '证据服务', status: 'success', response: '51ms' },
                    { name: '数据库', status: 'success', response: '12ms' },
                    { name: '缓存服务', status: 'warning', response: '156ms' },
                  ]}
                  renderItem={(item) => (
                    <List.Item
                      actions={[
                        <Text key="response" type="secondary" style={{ fontSize: 12 }}>
                          {item.response}
                        </Text>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Badge 
                            status={item.status as any} 
                          />
                        }
                        title={item.name}
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        </Space>
      </div>
    </MainLayout>
  );
}