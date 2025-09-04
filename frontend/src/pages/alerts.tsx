/**
 * InvestIQ Platform - 告警管理中心
 * 提供告警监控、处理和规则管理功能
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
  Select,
  Input,
  DatePicker,
  Badge,
  Modal,
  Form,
  Descriptions,
  Alert as AntAlert,
  Tabs,
  Progress,
  Statistic,
  List,
  Avatar,
  Dropdown,
  Tooltip,
  Switch,
  InputNumber,
  message
} from 'antd';
import { 
  BellOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  FilterOutlined,
  SettingOutlined,
  EyeOutlined,
  UserOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  BulbOutlined,
  TrendingUpOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { Alert, AlertRule } from '@/types/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';
import type { ColumnsType } from 'antd/es/table';
import type { Dayjs } from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

export default function AlertsPage() {
  const [selectedAlerts, setSelectedAlerts] = useState<string[]>([]);
  const [alertDetailVisible, setAlertDetailVisible] = useState(false);
  const [ruleModalVisible, setRuleModalVisible] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [selectedRule, setSelectedRule] = useState<AlertRule | null>(null);
  const [ruleForm] = Form.useForm();
  const [activeTab, setActiveTab] = useState('alerts');
  const [filterParams, setFilterParams] = useState({
    alert_type: undefined as string | undefined,
    severity: undefined as string | undefined,
    status: undefined as string | undefined,
    assignee: undefined as string | undefined,
    page: 1,
    page_size: 20,
  });

  const queryClient = useQueryClient();

  // 获取告警列表
  const { data: alertsData, isLoading: alertsLoading, refetch: refetchAlerts } = useQuery({
    queryKey: ['alerts', filterParams],
    queryFn: () => apiClient.getAlerts(filterParams),
    refetchInterval: 10000, // 10秒刷新
  });

  // 获取告警规则
  const { data: rulesData, isLoading: rulesLoading } = useQuery({
    queryKey: ['alert-rules'],
    queryFn: () => apiClient.getAlertRules({ enabled_only: false }),
  });

  // 获取告警仪表板数据
  const { data: dashboardData } = useQuery({
    queryKey: ['alert-dashboard'],
    queryFn: () => apiClient.getAlertDashboard(7),
  });

  // 确认告警
  const acknowledgeMutation = useMutation({
    mutationFn: ({ alertId, assignee }: { alertId: string; assignee?: string }) =>
      apiClient.acknowledgeAlert(alertId, assignee),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      message.success('告警确认成功');
    },
  });

  // 批量确认告警
  const batchAcknowledgeMutation = useMutation({
    mutationFn: ({ alertIds, assignee }: { alertIds: string[]; assignee?: string }) =>
      apiClient.batchAcknowledgeAlerts(alertIds, assignee),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      setSelectedAlerts([]);
      message.success('批量确认成功');
    },
  });

  // 更新告警
  const updateAlertMutation = useMutation({
    mutationFn: ({ alertId, updates }: { alertId: string; updates: any }) =>
      apiClient.updateAlert(alertId, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['alerts'] });
      setAlertDetailVisible(false);
      message.success('告警更新成功');
    },
  });

  // 获取严重程度颜色
  const getSeverityColor = (severity: string) => {
    const colors = { 'P1': 'red', 'P2': 'orange', 'P3': 'blue' };
    return colors[severity as keyof typeof colors] || 'default';
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      'new': 'red',
      'acknowledged': 'orange', 
      'in_progress': 'blue',
      'resolved': 'green',
      'closed': 'default'
    };
    return colors[status as keyof typeof colors] || 'default';
  };

  // 显示告警详情
  const showAlertDetail = (alert: Alert) => {
    setSelectedAlert(alert);
    setAlertDetailVisible(true);
  };

  // 处理筛选变更
  const handleFilterChange = (key: string, value: any) => {
    setFilterParams(prev => ({ ...prev, [key]: value, page: 1 }));
  };

  // 批量操作菜单
  const batchActionsMenu = [
    {
      key: 'acknowledge',
      label: '批量确认',
      icon: <CheckCircleOutlined />,
      onClick: () => {
        if (selectedAlerts.length === 0) {
          message.warning('请选择要确认的告警');
          return;
        }
        batchAcknowledgeMutation.mutate({
          alertIds: selectedAlerts,
          assignee: '系统管理员',
        });
      },
    },
    {
      key: 'assign',
      label: '批量分配',
      icon: <UserOutlined />,
    },
  ];

  // 告警列表表格列
  const alertColumns: ColumnsType<Alert> = [
    {
      title: '告警信息',
      key: 'alert_info',
      width: 250,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Badge 
              status={record.severity === 'P1' ? 'error' : record.severity === 'P2' ? 'warning' : 'processing'} 
            />
            <Text strong style={{ fontSize: 14 }}>
              {record.title}
            </Text>
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => showAlertDetail(record)}
            />
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.message.length > 50 ? `${record.message.substring(0, 50)}...` : record.message}
          </Text>
          <Space size="small">
            <Tag color={getSeverityColor(record.severity)} size="small">
              {record.severity}
            </Tag>
            <Tag color="blue" size="small">
              {record.alert_type}
            </Tag>
          </Space>
        </Space>
      ),
    },
    {
      title: '关联实体',
      key: 'entity',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          {record.ticker && <Text style={{ fontSize: 12 }}>{record.ticker}</Text>}
          {record.entity_type && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              {record.entity_type}
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const statusMap = {
          'new': '新建',
          'acknowledged': '已确认',
          'in_progress': '处理中',
          'resolved': '已解决',
          'closed': '已关闭',
        };
        return (
          <Tag color={getStatusColor(status)}>
            {statusMap[status as keyof typeof statusMap] || status}
          </Tag>
        );
      },
    },
    {
      title: '处理人',
      dataIndex: 'assignee',
      key: 'assignee',
      width: 100,
      render: (assignee: string) => (
        assignee ? (
          <Space>
            <Avatar size="small" icon={<UserOutlined />} />
            <Text style={{ fontSize: 12 }}>{assignee}</Text>
          </Space>
        ) : (
          <Text type="secondary" style={{ fontSize: 12 }}>未分配</Text>
        )
      ),
    },
    {
      title: '触发时间',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      width: 120,
      sorter: true,
      render: (time: string) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: 12 }}>
            {dayjs(time).format('MM-DD HH:mm')}
          </Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {dayjs(time).fromNow()}
          </Text>
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="text" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => showAlertDetail(record)}
          />
          {record.status === 'new' && (
            <Button 
              type="text" 
              size="small" 
              icon={<CheckCircleOutlined />}
              loading={acknowledgeMutation.isPending}
              onClick={() => acknowledgeMutation.mutate({
                alertId: record.alert_id,
                assignee: '系统管理员',
              })}
            />
          )}
          <Button 
            type="text" 
            size="small" 
            icon={<EditOutlined />}
          />
        </Space>
      ),
    },
  ];

  // 告警规则表格列
  const ruleColumns: ColumnsType<AlertRule> = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record) => (
        <Space direction="vertical" size={0}>
          <Text strong>{name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.description}
          </Text>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'alert_type',
      key: 'alert_type',
      render: (type: string) => <Tag>{type}</Tag>,
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>{severity}</Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'is_enabled',
      key: 'is_enabled',
      render: (enabled: boolean) => (
        <Switch checked={enabled} size="small" />
      ),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (time: string) => dayjs(time).format('MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space size="small">
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      ),
    },
  ];

  // 告警趋势图
  const alertTrendOption = useMemo(() => {
    const dailyStats = dashboardData?.data?.daily_trend || [];
    return {
      title: {
        text: '告警趋势',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type: 'category',
        data: dailyStats.map((item: any) => dayjs(item.date).format('MM/DD'))
      },
      yAxis: {
        type: 'value'
      },
      series: [{
        name: '告警数量',
        type: 'line',
        data: dailyStats.map((item: any) => item.count),
        smooth: true,
        itemStyle: { color: '#1890ff' },
        areaStyle: { opacity: 0.1 }
      }],
      grid: { left: 40, right: 20, top: 50, bottom: 30 }
    };
  }, [dashboardData]);

  // 告警分布饼图
  const alertDistributionOption = useMemo(() => {
    const typeStats = dashboardData?.data?.type_distribution || {};
    return {
      title: {
        text: '告警类型分布',
        textStyle: { fontSize: 14 }
      },
      tooltip: {
        trigger: 'item'
      },
      series: [{
        type: 'pie',
        radius: ['40%', '70%'],
        data: Object.entries(typeStats).map(([type, count]) => ({
          name: type,
          value: count
        })),
        label: {
          formatter: '{b}: {c} ({d}%)'
        }
      }]
    };
  }, [dashboardData]);

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
            <div>
              <Title level={2} style={{ margin: 0 }}>
                告警管理中心
              </Title>
              <Text type="secondary">
                实时监控系统告警，快速响应和处理风险事件
              </Text>
            </div>
            <Space>
              <Button 
                icon={<PlusOutlined />}
                onClick={() => setRuleModalVisible(true)}
              >
                新建规则
              </Button>
              <Button 
                icon={<ReloadOutlined />}
                onClick={() => refetchAlerts()}
                loading={alertsLoading}
              >
                刷新
              </Button>
            </Space>
          </Space>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 统计概览 */}
          <Row gutter={[24, 16]}>
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="活跃告警"
                  value={dashboardData?.data?.basic_stats?.new_alerts || 0}
                  prefix={<BellOutlined />}
                  valueStyle={{ color: '#f5222d' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="待处理"
                  value={dashboardData?.data?.basic_stats?.acknowledged_alerts || 0}
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{ color: '#faad14' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="已解决"
                  value={dashboardData?.data?.basic_stats?.resolved_alerts || 0}
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Statistic
                  title="解决率"
                  value={dashboardData?.data?.basic_stats?.resolution_rate || 0}
                  suffix="%"
                  prefix={<TrendingUpOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                  precision={1}
                />
              </Card>
            </Col>
          </Row>

          {/* 图表区域 */}
          <Row gutter={[24, 24]}>
            <Col xs={24} lg={12}>
              <Card title="告警趋势" size="small">
                <ReactECharts option={alertTrendOption} style={{ height: '300px' }} />
              </Card>
            </Col>
            
            <Col xs={24} lg={12}>
              <Card title="类型分布" size="small">
                <ReactECharts option={alertDistributionOption} style={{ height: '300px' }} />
              </Card>
            </Col>
          </Row>

          {/* 主内容区域 */}
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab="告警列表" key="alerts">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {/* 筛选栏 */}
                  <Row gutter={16} align="middle">
                    <Col xs={24} sm={4}>
                      <Select
                        placeholder="告警类型"
                        allowClear
                        style={{ width: '100%' }}
                        value={filterParams.alert_type}
                        onChange={(value) => handleFilterChange('alert_type', value)}
                      >
                        <Option value="event">事件</Option>
                        <Option value="kpi">KPI</Option>
                        <Option value="trend">趋势</Option>
                        <Option value="drawdown">回撤</Option>
                      </Select>
                    </Col>
                    
                    <Col xs={24} sm={4}>
                      <Select
                        placeholder="严重程度"
                        allowClear
                        style={{ width: '100%' }}
                        value={filterParams.severity}
                        onChange={(value) => handleFilterChange('severity', value)}
                      >
                        <Option value="P1">P1-严重</Option>
                        <Option value="P2">P2-重要</Option>
                        <Option value="P3">P3-一般</Option>
                      </Select>
                    </Col>
                    
                    <Col xs={24} sm={4}>
                      <Select
                        placeholder="处理状态"
                        allowClear
                        style={{ width: '100%' }}
                        value={filterParams.status}
                        onChange={(value) => handleFilterChange('status', value)}
                      >
                        <Option value="new">新建</Option>
                        <Option value="acknowledged">已确认</Option>
                        <Option value="in_progress">处理中</Option>
                        <Option value="resolved">已解决</Option>
                      </Select>
                    </Col>
                    
                    <Col xs={24} sm={6}>
                      <Input
                        placeholder="搜索告警内容"
                        allowClear
                        prefix={<FilterOutlined />}
                      />
                    </Col>
                    
                    <Col xs={24} sm={6}>
                      <Space>
                        <Dropdown
                          menu={{
                            items: batchActionsMenu,
                            onClick: ({ key }) => {
                              const action = batchActionsMenu.find(item => item.key === key);
                              action?.onClick?.();
                            }
                          }}
                          disabled={selectedAlerts.length === 0}
                        >
                          <Button>
                            批量操作 ({selectedAlerts.length})
                          </Button>
                        </Dropdown>
                      </Space>
                    </Col>
                  </Row>

                  {/* 告警表格 */}
                  <Table
                    columns={alertColumns}
                    dataSource={alertsData?.data?.alerts || []}
                    rowKey="alert_id"
                    loading={alertsLoading}
                    size="middle"
                    scroll={{ x: 1000 }}
                    rowSelection={{
                      selectedRowKeys: selectedAlerts,
                      onChange: setSelectedAlerts,
                    }}
                    pagination={{
                      current: filterParams.page,
                      pageSize: filterParams.page_size,
                      total: alertsData?.data?.pagination?.total_count,
                      showSizeChanger: true,
                      showQuickJumper: true,
                      showTotal: (total, range) => 
                        `第 ${range[0]}-${range[1]} 条，共 ${total} 条告警`,
                      onChange: (page, pageSize) => {
                        setFilterParams(prev => ({ ...prev, page, page_size: pageSize }));
                      },
                    }}
                  />
                </Space>
              </TabPane>

              <TabPane tab="规则管理" key="rules">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  <Space>
                    <Button type="primary" icon={<PlusOutlined />}>
                      新建规则
                    </Button>
                    <Button icon={<BulbOutlined />}>
                      规则模板
                    </Button>
                  </Space>

                  <Table
                    columns={ruleColumns}
                    dataSource={rulesData?.data?.rules || []}
                    rowKey="rule_id"
                    loading={rulesLoading}
                    size="middle"
                    pagination={{
                      showSizeChanger: true,
                      showTotal: (total) => `共 ${total} 条规则`,
                    }}
                  />
                </Space>
              </TabPane>
            </Tabs>
          </Card>
        </Space>

        {/* 告警详情弹窗 */}
        <Modal
          title="告警详情"
          open={alertDetailVisible}
          onCancel={() => setAlertDetailVisible(false)}
          width={800}
          footer={[
            <Button key="close" onClick={() => setAlertDetailVisible(false)}>
              关闭
            </Button>,
            selectedAlert?.status === 'new' && (
              <Button 
                key="acknowledge" 
                type="primary"
                icon={<CheckCircleOutlined />}
                loading={acknowledgeMutation.isPending}
                onClick={() => selectedAlert && acknowledgeMutation.mutate({
                  alertId: selectedAlert.alert_id,
                  assignee: '系统管理员',
                })}
              >
                确认告警
              </Button>
            ),
          ].filter(Boolean)}
        >
          {selectedAlert && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <AntAlert
                type={selectedAlert.severity === 'P1' ? 'error' : 
                      selectedAlert.severity === 'P2' ? 'warning' : 'info'}
                message={selectedAlert.title}
                description={selectedAlert.message}
                showIcon
              />

              <Descriptions column={2} bordered size="small">
                <Descriptions.Item label="告警ID">
                  {selectedAlert.alert_id}
                </Descriptions.Item>
                <Descriptions.Item label="告警类型">
                  <Tag>{selectedAlert.alert_type}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="严重程度">
                  <Tag color={getSeverityColor(selectedAlert.severity)}>
                    {selectedAlert.severity}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="当前状态">
                  <Tag color={getStatusColor(selectedAlert.status)}>
                    {selectedAlert.status}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="关联实体">
                  {selectedAlert.ticker || selectedAlert.entity_type || '无'}
                </Descriptions.Item>
                <Descriptions.Item label="处理人">
                  {selectedAlert.assignee || '未分配'}
                </Descriptions.Item>
                <Descriptions.Item label="当前值">
                  {selectedAlert.current_value || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="阈值">
                  {selectedAlert.threshold_value || 'N/A'}
                </Descriptions.Item>
                <Descriptions.Item label="触发时间">
                  {dayjs(selectedAlert.triggered_at).format('YYYY-MM-DD HH:mm:ss')}
                </Descriptions.Item>
                <Descriptions.Item label="确认时间">
                  {selectedAlert.acknowledged_at ? 
                    dayjs(selectedAlert.acknowledged_at).format('YYYY-MM-DD HH:mm:ss') : '未确认'}
                </Descriptions.Item>
                <Descriptions.Item label="是否超时">
                  <Badge 
                    status={selectedAlert.is_overdue ? 'error' : 'success'} 
                    text={selectedAlert.is_overdue ? '已超时' : '正常'} 
                  />
                </Descriptions.Item>
                <Descriptions.Item label="命中次数">
                  {selectedAlert.hits_count}
                </Descriptions.Item>
              </Descriptions>

              {selectedAlert.rule_name && (
                <Card title="规则信息" size="small">
                  <Text>规则名称: {selectedAlert.rule_name}</Text>
                </Card>
              )}
            </Space>
          )}
        </Modal>
      </div>
    </MainLayout>
  );
}