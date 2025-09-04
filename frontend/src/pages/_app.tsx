import type { AppProps } from 'next/app';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { useState } from 'react';
import '@/styles/globals.css';

// Ant Design 主题配置
const theme = {
  token: {
    colorPrimary: '#1890ff',
    colorSuccess: '#52c41a',
    colorWarning: '#faad14',
    colorError: '#f5222d',
    borderRadius: 6,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"',
  },
  components: {
    Layout: {
      headerBg: '#001529',
      siderBg: '#001529',
    },
    Menu: {
      darkItemBg: 'transparent',
      darkSubMenuItemBg: '#000c17',
    },
  },
};

export default function App({ Component, pageProps }: AppProps) {
  // React Query 客户端
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 1000 * 60 * 5, // 5分钟
            cacheTime: 1000 * 60 * 10, // 10分钟
            retry: 2,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ConfigProvider locale={zhCN} theme={theme}>
        <Component {...pageProps} />
      </ConfigProvider>
    </QueryClientProvider>
  );
}