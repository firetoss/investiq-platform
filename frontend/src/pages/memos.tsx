/**
 * InvestIQ Platform - 投资备忘录工具
 * 提供投资决策记录和备忘录生成功能
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
  Modal,
  Form,
  Descriptions,
  List,
  Avatar,
  Tooltip,
  Divider,
  Upload,
  message,
  Switch,
  Rate,
  Progress,
  Tabs,
  Timeline,
  Alert as AntAlert,
  Drawer
} from 'antd';
import { 
  FileTextOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  UploadOutlined,
  BulbOutlined,
  StarOutlined,
  ClockCircleOutlined,
  UserOutlined,
  TagsOutlined,
  SearchOutlined,
  FilterOutlined,
  BookOutlined,
  ExportOutlined,
  ImportOutlined,
  CopyOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import MainLayout from '@/components/layout/MainLayout';
import { apiClient } from '@/services/api';
import { ReactECharts } from 'echarts-for-react';
import dayjs from 'dayjs';
import type { ColumnsType } from 'antd/es/table';
import type { UploadProps } from 'antd';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

interface Memo {
  id: string;
  title: string;
  content: string;
  type: 'investment' | 'research' | 'market' | 'strategy';
  status: 'draft' | 'published' | 'archived';
  tags: string[];
  priority: number;
  confidence: number;
  related_entities: {
    type: string;
    id: string;
    name: string;
  }[];
  attachments: {
    id: string;
    name: string;
    url: string;
    size: number;
  }[];
  created_by: string;
  created_at: string;
  updated_at: string;
  views: number;
  shares: number;
}

export default function MemosPage() {
  const [selectedMemos, setSelectedMemos] = useState<string[]>([]);
  const [memoModalVisible, setMemoModalVisible] = useState(false);
  const [viewModalVisible, setViewModalVisible] = useState(false);
  const [templateDrawerVisible, setTemplateDrawerVisible] = useState(false);
  const [selectedMemo, setSelectedMemo] = useState<Memo | null>(null);
  const [memoForm] = Form.useForm();
  const [activeTab, setActiveTab] = useState('list');
  const [filterParams, setFilterParams] = useState({
    type: undefined as string | undefined,
    status: undefined as string | undefined,
    tags: undefined as string[] | undefined,
    search: '',
    page: 1,
    page_size: 20,
  });

  const queryClient = useQueryClient();

  // 模拟备忘录数据
  const mockMemos: Memo[] = [
    {
      id: '1',
      title: '新能源行业投资机会分析',
      content: '基于最新政策和市场表现，新能源行业呈现以下投资机会...',
      type: 'investment',
      status: 'published',
      tags: ['新能源', '政策利好', 'A股'],
      priority: 5,
      confidence: 85,
      related_entities: [
        { type: 'industry', id: 'new_energy', name: '新能源' },
        { type: 'equity', id: '300750', name: '宁德时代' },
      ],
      attachments: [
        { id: '1', name: '政策文件.pdf', url: '#', size: 1024000 },
      ],
      created_by: '投资经理',
      created_at: '2024-01-15T10:30:00Z',
      updated_at: '2024-01-16T15:20:00Z',
      views: 156,
      shares: 8,
    },
    {
      id: '2',
      title: '医药板块回调风险提示',
      content: '近期医药板块出现明显回调，主要风险因素包括...',
      type: 'market',
      status: 'published',
      tags: ['医药', '风险提示', '回调'],
      priority: 4,
      confidence: 78,
      related_entities: [
        { type: 'industry', id: 'healthcare', name: '医药' },
      ],
      attachments: [],
      created_by: '风控经理',
      created_at: '2024-01-10T14:20:00Z',
      updated_at: '2024-01-10T14:20:00Z',
      views: 89,
      shares: 12,
    },
  ];

  // 获取备忘录列表（模拟）
  const { data: memosData, isLoading, refetch } = useQuery({
    queryKey: ['memos', filterParams],
    queryFn: async () => ({
      data: {
        items: mockMemos.filter(memo => {
          if (filterParams.type && memo.type !== filterParams.type) return false;
          if (filterParams.status && memo.status !== filterParams.status) return false;
          if (filterParams.search && !memo.title.toLowerCase().includes(filterParams.search.toLowerCase())) return false;
          return true;
        }),
        pagination: {
          page: filterParams.page,
          page_size: filterParams.page_size,
          total_count: mockMemos.length,
          total_pages: Math.ceil(mockMemos.length / filterParams.page_size),
        }
      }
    }),
    refetchInterval: 30000,
  });

  // 创建/更新备忘录
  const saveMemoMutation = useMutation({
    mutationFn: async (data: any) => {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      return { success: true };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memos'] });
      setMemoModalVisible(false);
      memoForm.resetFields();
      message.success('备忘录保存成功');
    },
  });

  // 显示备忘录详情
  const showMemoDetail = (memo: Memo) => {
    setSelectedMemo(memo);
    setViewModalVisible(true);
  };

  // 创建新备忘录
  const createNewMemo = () => {
    setSelectedMemo(null);
    memoForm.resetFields();
    setMemoModalVisible(true);
  };

  // 编辑备忘录
  const editMemo = (memo: Memo) => {
    setSelectedMemo(memo);
    memoForm.setFieldsValue({
      title: memo.title,
      content: memo.content,
      type: memo.type,
      tags: memo.tags,
      priority: memo.priority,
      confidence: memo.confidence,
    });
    setMemoModalVisible(true);
  };

  // 处理筛选变更
  const handleFilterChange = (key: string, value: any) => {
    setFilterParams(prev => ({ ...prev, [key]: value, page: 1 }));
  };

  // 保存备忘录
  const handleSaveMemo = (values: any) => {
    saveMemoMutation.mutate({
      ...values,
      id: selectedMemo?.id,
    });
  };

  // 获取类型颜色
  const getTypeColor = (type: string) => {
    const colors = {
      'investment': 'blue',
      'research': 'green',
      'market': 'orange',
      'strategy': 'purple',
    };
    return colors[type as keyof typeof colors] || 'default';
  };

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      'draft': 'default',
      'published': 'green',
      'archived': 'gray',
    };
    return colors[status as keyof typeof colors] || 'default';
  };

  // 备忘录表格列
  const memoColumns: ColumnsType<Memo> = [
    {
      title: '标题',
      key: 'title',
      width: 250,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Space>
            <Text strong style={{ fontSize: 14 }}>
              {record.title}
            </Text>
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => showMemoDetail(record)}
            />
          </Space>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.content.length > 60 ? `${record.content.substring(0, 60)}...` : record.content}
          </Text>
          <Space size="small" wrap>
            {record.tags.slice(0, 3).map(tag => (
              <Tag key={tag} size="small">{tag}</Tag>
            ))}
            {record.tags.length > 3 && <Text type="secondary" style={{ fontSize: 11 }}>+{record.tags.length - 3}</Text>}
          </Space>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 100,
      render: (type: string) => {
        const typeMap = {
          'investment': '投资分析',
          'research': '研究报告',
          'market': '市场观察',
          'strategy': '策略规划',
        };
        return (
          <Tag color={getTypeColor(type)}>
            {typeMap[type as keyof typeof typeMap] || type}
          </Tag>
        );
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => {
        const statusMap = {
          'draft': '草稿',
          'published': '已发布',
          'archived': '已归档',
        };
        return (
          <Tag color={getStatusColor(status)}>
            {statusMap[status as keyof typeof statusMap] || status}
          </Tag>
        );
      },
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 100,
      render: (priority: number) => (
        <Rate disabled value={priority} count={5} style={{ fontSize: 12 }} />
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
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
      title: '关联实体',
      key: 'related_entities',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          {record.related_entities.slice(0, 2).map((entity, index) => (
            <Tag key={index} size="small" color="blue">
              {entity.name}
            </Tag>
          ))}
          {record.related_entities.length > 2 && (
            <Text type="secondary" style={{ fontSize: 11 }}>
              +{record.related_entities.length - 2} 个
            </Text>
          )}
        </Space>
      ),
    },
    {
      title: '创建信息',
      key: 'created_info',
      width: 120,
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text style={{ fontSize: 12 }}>{record.created_by}</Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {dayjs(record.created_at).format('MM-DD HH:mm')}
          </Text>
          <Space size={8}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {record.views} 阅读
            </Text>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {record.shares} 分享
            </Text>
          </Space>
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      width: 150,
      render: (_, record) => (
        <Space size="small">
          <Button 
            type="text" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => showMemoDetail(record)}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<EditOutlined />}
            onClick={() => editMemo(record)}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<ShareAltOutlined />}
          />
          <Button 
            type="text" 
            size="small" 
            icon={<DownloadOutlined />}
          />
        </Space>
      ),
    },
  ];

  // 文件上传配置
  const uploadProps: UploadProps = {
    name: 'file',
    action: '/api/upload',
    headers: {
      authorization: 'authorization-text',
    },
    onChange(info) {
      if (info.file.status === 'done') {
        message.success(`${info.file.name} 文件上传成功`);
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} 文件上传失败`);
      }
    },
  };

  // 备忘录模板
  const templates = [
    {
      id: 'investment_analysis',
      name: '投资分析模板',
      description: '标准的投资分析备忘录模板',
      content: `# 投资分析备忘录

## 投资要点
- 核心投资逻辑
- 风险因素分析
- 预期收益评估

## 基本面分析
### 财务状况
### 经营状况
### 行业地位

## 技术面分析
### 价格走势
### 技术指标
### 支撑阻力位

## 投资建议
### 目标价位
### 仓位建议
### 时间周期

## 风险提示
`,
    },
    {
      id: 'market_outlook',
      name: '市场展望模板',
      description: '市场趋势分析模板',
      content: `# 市场展望备忘录

## 市场概况
### 主要指数表现
### 成交量变化
### 资金流向

## 政策环境
### 宏观政策
### 行业政策
### 监管动态

## 投资主线
### 主题投资机会
### 板块轮动预期
### 个股挖掘方向

## 风险关注
### 系统性风险
### 流动性风险
### 政策风险
`,
    },
  ];

  return (
    <MainLayout>
      <div>
        <div className="page-header">
          <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
            <div>
              <Title level={2} style={{ margin: 0 }}>
                投资备忘录
              </Title>
              <Text type="secondary">
                记录投资决策过程，构建知识库和经验积累
              </Text>
            </div>
            <Space>
              <Button 
                icon={<BookOutlined />}
                onClick={() => setTemplateDrawerVisible(true)}
              >
                模板库
              </Button>
              <Button 
                type="primary"
                icon={<PlusOutlined />}
                onClick={createNewMemo}
              >
                新建备忘录
              </Button>
            </Space>
          </Space>
        </div>

        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 统计概览 */}
          <Row gutter={[24, 16]}>
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Space direction="vertical" size={0}>
                  <Text type="secondary">总备忘录</Text>
                  <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                    {memosData?.data?.pagination?.total_count || 0}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    本月新增 +5
                  </Text>
                </Space>
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Space direction="vertical" size={0}>
                  <Text type="secondary">已发布</Text>
                  <Title level={3} style={{ margin: 0, color: '#52c41a' }}>
                    {mockMemos.filter(m => m.status === 'published').length}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    发布率 85%
                  </Text>
                </Space>
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Space direction="vertical" size={0}>
                  <Text type="secondary">草稿</Text>
                  <Title level={3} style={{ margin: 0, color: '#faad14' }}>
                    {mockMemos.filter(m => m.status === 'draft').length}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    待完善
                  </Text>
                </Space>
              </Card>
            </Col>
            
            <Col xs={24} sm={6}>
              <Card className="stat-card">
                <Space direction="vertical" size={0}>
                  <Text type="secondary">平均阅读量</Text>
                  <Title level={3} style={{ margin: 0, color: '#722ed1' }}>
                    {Math.round(mockMemos.reduce((sum, m) => sum + m.views, 0) / mockMemos.length) || 0}
                  </Title>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    较上月 +15%
                  </Text>
                </Space>
              </Card>
            </Col>
          </Row>

          {/* 主内容区域 */}
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab="备忘录列表" key="list">
                <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                  {/* 筛选栏 */}
                  <Row gutter={16} align="middle">
                    <Col xs={24} sm={4}>
                      <Select
                        placeholder="备忘录类型"
                        allowClear
                        style={{ width: '100%' }}
                        value={filterParams.type}
                        onChange={(value) => handleFilterChange('type', value)}
                      >
                        <Option value="investment">投资分析</Option>
                        <Option value="research">研究报告</Option>
                        <Option value="market">市场观察</Option>
                        <Option value="strategy">策略规划</Option>
                      </Select>
                    </Col>
                    
                    <Col xs={24} sm={4}>
                      <Select
                        placeholder="状态"
                        allowClear
                        style={{ width: '100%' }}
                        value={filterParams.status}
                        onChange={(value) => handleFilterChange('status', value)}
                      >
                        <Option value="draft">草稿</Option>
                        <Option value="published">已发布</Option>
                        <Option value="archived">已归档</Option>
                      </Select>
                    </Col>
                    
                    <Col xs={24} sm={6}>
                      <Input
                        placeholder="搜索标题或内容"
                        prefix={<SearchOutlined />}
                        allowClear
                        value={filterParams.search}
                        onChange={(e) => handleFilterChange('search', e.target.value)}
                      />
                    </Col>
                    
                    <Col xs={24} sm={4}>
                      <RangePicker size="middle" style={{ width: '100%' }} />
                    </Col>
                    
                    <Col xs={24} sm={6}>
                      <Space>
                        <Button icon={<FilterOutlined />}>
                          高级筛选
                        </Button>
                        <Button icon={<ExportOutlined />}>
                          批量导出
                        </Button>
                      </Space>
                    </Col>
                  </Row>

                  {/* 备忘录表格 */}
                  <Table
                    columns={memoColumns}
                    dataSource={memosData?.data?.items || []}
                    rowKey="id"
                    loading={isLoading}
                    size="middle"
                    scroll={{ x: 1200 }}
                    rowSelection={{
                      selectedRowKeys: selectedMemos,
                      onChange: setSelectedMemos,
                    }}
                    pagination={{
                      current: filterParams.page,
                      pageSize: filterParams.page_size,
                      total: memosData?.data?.pagination?.total_count,
                      showSizeChanger: true,
                      showQuickJumper: true,
                      showTotal: (total, range) => 
                        `第 ${range[0]}-${range[1]} 条，共 ${total} 份备忘录`,
                      onChange: (page, pageSize) => {
                        setFilterParams(prev => ({ ...prev, page, page_size: pageSize }));
                      },
                    }}
                  />
                </Space>
              </TabPane>

              <TabPane tab="知识图谱" key="knowledge">
                <div style={{ textAlign: 'center', padding: '100px 0', color: '#999' }}>
                  <BookOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                  <div>知识图谱功能开发中...</div>
                  <div style={{ fontSize: 12, marginTop: 8 }}>将展示备忘录之间的关联关系</div>
                </div>
              </TabPane>
            </Tabs>
          </Card>
        </Space>

        {/* 备忘录编辑弹窗 */}
        <Modal
          title={selectedMemo ? '编辑备忘录' : '新建备忘录'}
          open={memoModalVisible}
          onCancel={() => setMemoModalVisible(false)}
          width={900}
          footer={null}
          destroyOnClose
        >
          <Form
            form={memoForm}
            layout="vertical"
            onFinish={handleSaveMemo}
          >
            <Row gutter={16}>
              <Col span={16}>
                <Form.Item
                  name="title"
                  label="标题"
                  rules={[{ required: true, message: '请输入备忘录标题' }]}
                >
                  <Input placeholder="请输入备忘录标题" />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="type"
                  label="类型"
                  rules={[{ required: true, message: '请选择备忘录类型' }]}
                >
                  <Select placeholder="选择类型">
                    <Option value="investment">投资分析</Option>
                    <Option value="research">研究报告</Option>
                    <Option value="market">市场观察</Option>
                    <Option value="strategy">策略规划</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="content"
              label="内容"
              rules={[{ required: true, message: '请输入备忘录内容' }]}
            >
              <TextArea 
                rows={12} 
                placeholder="支持 Markdown 格式..."
                style={{ fontFamily: 'Monaco, Consolas, monospace' }}
              />
            </Form.Item>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="tags"
                  label="标签"
                >
                  <Select
                    mode="tags"
                    placeholder="添加标签"
                    style={{ width: '100%' }}
                  >
                    <Option value="新能源">新能源</Option>
                    <Option value="医药">医药</Option>
                    <Option value="科技">科技</Option>
                    <Option value="消费">消费</Option>
                  </Select>
                </Form.Item>
              </Col>
              
              <Col span={6}>
                <Form.Item
                  name="priority"
                  label="优先级"
                  initialValue={3}
                >
                  <Rate count={5} />
                </Form.Item>
              </Col>
              
              <Col span={6}>
                <Form.Item
                  name="confidence"
                  label="置信度"
                  initialValue={70}
                >
                  <InputNumber 
                    min={0} 
                    max={100} 
                    style={{ width: '100%' }}
                    formatter={(value) => `${value}%`}
                    parser={(value) => value!.replace('%', '')}
                  />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item label="附件">
              <Upload {...uploadProps}>
                <Button icon={<UploadOutlined />}>上传附件</Button>
              </Upload>
            </Form.Item>

            <Divider />

            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => setMemoModalVisible(false)}>
                取消
              </Button>
              <Button onClick={() => handleSaveMemo({ ...memoForm.getFieldsValue(), status: 'draft' })}>
                保存草稿
              </Button>
              <Button 
                type="primary" 
                htmlType="submit"
                loading={saveMemoMutation.isPending}
              >
                发布备忘录
              </Button>
            </Space>
          </Form>
        </Modal>

        {/* 备忘录详情弹窗 */}
        <Modal
          title="备忘录详情"
          open={viewModalVisible}
          onCancel={() => setViewModalVisible(false)}
          width={800}
          footer={[
            <Button key="close" onClick={() => setViewModalVisible(false)}>
              关闭
            </Button>,
            <Button key="edit" icon={<EditOutlined />} onClick={() => selectedMemo && editMemo(selectedMemo)}>
              编辑
            </Button>,
            <Button key="share" type="primary" icon={<ShareAltOutlined />}>
              分享
            </Button>,
          ]}
        >
          {selectedMemo && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Space style={{ marginBottom: 16 }}>
                  <Tag color={getTypeColor(selectedMemo.type)}>
                    {selectedMemo.type}
                  </Tag>
                  <Tag color={getStatusColor(selectedMemo.status)}>
                    {selectedMemo.status}
                  </Tag>
                  <Rate disabled value={selectedMemo.priority} count={5} style={{ fontSize: 12 }} />
                </Space>
                
                <Title level={3}>{selectedMemo.title}</Title>
                
                <Space wrap style={{ marginBottom: 16 }}>
                  {selectedMemo.tags.map(tag => (
                    <Tag key={tag}>{tag}</Tag>
                  ))}
                </Space>
              </div>

              <div>
                <Title level={5}>内容</Title>
                <div style={{ 
                  background: '#fafafa', 
                  padding: 16, 
                  borderRadius: 6,
                  whiteSpace: 'pre-wrap',
                  minHeight: 200
                }}>
                  {selectedMemo.content}
                </div>
              </div>

              <Descriptions column={2} size="small">
                <Descriptions.Item label="创建者">
                  {selectedMemo.created_by}
                </Descriptions.Item>
                <Descriptions.Item label="置信度">
                  {selectedMemo.confidence}%
                </Descriptions.Item>
                <Descriptions.Item label="创建时间">
                  {dayjs(selectedMemo.created_at).format('YYYY-MM-DD HH:mm')}
                </Descriptions.Item>
                <Descriptions.Item label="更新时间">
                  {dayjs(selectedMemo.updated_at).format('YYYY-MM-DD HH:mm')}
                </Descriptions.Item>
                <Descriptions.Item label="阅读量">
                  {selectedMemo.views}
                </Descriptions.Item>
                <Descriptions.Item label="分享量">
                  {selectedMemo.shares}
                </Descriptions.Item>
              </Descriptions>

              {selectedMemo.related_entities.length > 0 && (
                <div>
                  <Title level={5}>关联实体</Title>
                  <Space wrap>
                    {selectedMemo.related_entities.map((entity, index) => (
                      <Tag key={index} color="blue">
                        {entity.name}
                      </Tag>
                    ))}
                  </Space>
                </div>
              )}

              {selectedMemo.attachments.length > 0 && (
                <div>
                  <Title level={5}>附件</Title>
                  <List
                    size="small"
                    dataSource={selectedMemo.attachments}
                    renderItem={(attachment) => (
                      <List.Item
                        actions={[
                          <Button type="link" size="small" icon={<DownloadOutlined />}>
                            下载
                          </Button>
                        ]}
                      >
                        <List.Item.Meta
                          avatar={<FileTextOutlined />}
                          title={attachment.name}
                          description={`${(attachment.size / 1024).toFixed(1)} KB`}
                        />
                      </List.Item>
                    )}
                  />
                </div>
              )}
            </Space>
          )}
        </Modal>

        {/* 模板库抽屉 */}
        <Drawer
          title="备忘录模板库"
          placement="right"
          onClose={() => setTemplateDrawerVisible(false)}
          open={templateDrawerVisible}
          width={500}
        >
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {templates.map(template => (
              <Card
                key={template.id}
                size="small"
                title={template.name}
                extra={
                  <Button 
                    type="primary" 
                    size="small"
                    onClick={() => {
                      memoForm.setFieldsValue({
                        title: template.name,
                        content: template.content,
                        type: 'investment',
                      });
                      setTemplateDrawerVisible(false);
                      setMemoModalVisible(true);
                    }}
                  >
                    使用
                  </Button>
                }
              >
                <Paragraph style={{ margin: 0 }}>
                  {template.description}
                </Paragraph>
              </Card>
            ))}
          </Space>
        </Drawer>
      </div>
    </MainLayout>
  );
}