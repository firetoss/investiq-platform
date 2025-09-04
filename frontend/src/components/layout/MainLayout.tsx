/**
 * InvestIQ Platform - 主布局组件
 * 提供应用的主要导航和布局结构
 */

import React, { useState } from 'react';
import { Layout, Menu, Avatar, Dropdown, Button, Space, Badge } from 'antd';
import { 
  DashboardOutlined,
  BarChartOutlined,
  StockOutlined,
  PieChartOutlined,
  BellOutlined,
  FileTextOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined
} from '@ant-design/icons';
import Link from 'next/link';
import { useRouter } from 'next/router';

const { Header, Sider, Content } = Layout;

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const router = useRouter();

  // 菜单配置
  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: <Link href="/">概览面板</Link>,
    },
    {
      key: '/industries',
      icon: <BarChartOutlined />,
      label: <Link href="/industries">行业评分台</Link>,
    },
    {
      key: '/equities',
      icon: <StockOutlined />,
      label: <Link href="/equities">个股筛选</Link>,
    },
    {
      key: '/portfolios',
      icon: <PieChartOutlined />,
      label: <Link href="/portfolios">组合看板</Link>,
    },
    {
      key: '/alerts',
      icon: <BellOutlined />,
      label: (
        <Link href="/alerts">
          <Space>
            告警中心
            <Badge count={5} size="small" />
          </Space>
        </Link>
      ),
    },
    {
      key: '/memos',
      icon: <FileTextOutlined />,
      label: <Link href="/memos">投资备忘录</Link>,
    },
  ];

  // 用户菜单
  const userMenuItems = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: '用户资料',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
    {
      type: 'divider' as const,
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: '退出登录',
      danger: true,
    },
  ];

  const handleUserMenuClick = ({ key }: { key: string }) => {
    switch (key) {
      case 'logout':
        // 处理退出登录
        console.log('退出登录');
        break;
      case 'profile':
        router.push('/profile');
        break;
      case 'settings':
        router.push('/settings');
        break;
    }
  };

  return (
    <Layout className="page-container">
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        theme="dark"
        width={250}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
        }}
      >
        <div style={{ 
          padding: '16px', 
          textAlign: 'center',
          borderBottom: '1px solid #001529'
        }}>
          <h1 style={{ 
            color: 'white', 
            margin: 0, 
            fontSize: collapsed ? '16px' : '20px',
            fontWeight: 'bold'
          }}>
            {collapsed ? 'IQ' : 'InvestIQ'}
          </h1>
          {!collapsed && (
            <p style={{ 
              color: '#8c8c8c', 
              margin: 0, 
              fontSize: '12px' 
            }}>
              投资决策支持平台
            </p>
          )}
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[router.pathname]}
          items={menuItems}
          style={{ borderRight: 0 }}
        />
      </Sider>
      
      <Layout style={{ marginLeft: collapsed ? 80 : 250 }}>
        <Header
          style={{
            padding: '0 24px',
            background: 'white',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{
              fontSize: '16px',
              width: 48,
              height: 48,
            }}
          />
          
          <Space size="large">
            <Button
              type="text"
              icon={<BellOutlined />}
              size="large"
              style={{ fontSize: '16px' }}
            >
              <Badge count={5} />
            </Button>
            
            <Dropdown
              menu={{
                items: userMenuItems,
                onClick: handleUserMenuClick,
              }}
              placement="bottomRight"
            >
              <Space style={{ cursor: 'pointer' }}>
                <Avatar icon={<UserOutlined />} />
                <span style={{ fontSize: '14px' }}>投资经理</span>
              </Space>
            </Dropdown>
          </Space>
        </Header>
        
        <Content
          style={{
            margin: '24px 24px 0',
            overflow: 'initial',
            minHeight: 'calc(100vh - 112px)',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default MainLayout;