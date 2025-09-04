"""
数据采集服务 - 收集政策、新闻、财报等数据
支持多种数据源的统一采集和处理
"""

import asyncio
import aiohttp
import feedparser
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import re
import json
from pathlib import Path

from pydantic import BaseModel, Field, HttpUrl
import pandas as pd

logger = logging.getLogger(__name__)


class DataSource(BaseModel):
    """数据源配置"""
    name: str
    source_type: str  # rss, api, web, file
    url: Optional[str] = None
    headers: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    enabled: bool = True
    update_interval: int = 3600  # 秒
    last_updated: Optional[datetime] = None


class DataItem(BaseModel):
    """数据项"""
    id: str
    source: str
    data_type: str  # policy, news, financial, market
    title: str
    content: str
    url: Optional[str] = None
    publish_time: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)


class DataCollector:
    """
    数据采集器
    
    功能：
    1. 多源数据采集（RSS、API、网页）
    2. 数据清洗和标准化
    3. 增量更新
    4. 数据质量检查
    """
    
    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据源配置
        self.data_sources = self._setup_default_sources()
        
        # 会话管理
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 数据缓存
        self.data_cache: Dict[str, List[DataItem]] = {}
        
    def _setup_default_sources(self) -> Dict[str, DataSource]:
        """设置默认数据源"""
        sources = {
            # 政策数据源
            "ndrc_news": DataSource(
                name="国家发改委新闻",
                source_type="rss",
                url="https://www.ndrc.gov.cn/xwdt/xwfb/index.xml",
                update_interval=3600
            ),
            "miit_news": DataSource(
                name="工信部新闻",
                source_type="rss", 
                url="https://www.miit.gov.cn/xwdt/gxdt/index.xml",
                update_interval=3600
            ),
            
            # 新闻数据源
            "sina_finance": DataSource(
                name="新浪财经",
                source_type="rss",
                url="https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=1686&k=&num=50&page=1",
                update_interval=1800
            ),
            "eastmoney": DataSource(
                name="东方财富",
                source_type="rss",
                url="http://feed.eastmoney.com/rss/em_f_news.xml",
                update_interval=1800
            ),
            
            # 行业数据源（示例）
            "semiconductor_news": DataSource(
                name="半导体行业新闻",
                source_type="api",
                url="https://api.example.com/semiconductor/news",
                headers={"User-Agent": "InvestIQ-Platform/1.0"},
                update_interval=3600,
                enabled=False  # 默认禁用，需要配置真实API
            )
        }
        
        return sources
    
    async def start_session(self):
        """启动HTTP会话"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": "InvestIQ-Platform/1.0"}
            )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def collect_all_data(self) -> Dict[str, List[DataItem]]:
        """
        采集所有数据源的数据
        
        Returns:
            按数据源分组的数据项
        """
        await self.start_session()
        
        try:
            results = {}
            
            for source_name, source_config in self.data_sources.items():
                if not source_config.enabled:
                    continue
                
                try:
                    logger.info(f"Collecting data from {source_name}")
                    
                    # 检查是否需要更新
                    if self._should_update(source_config):
                        data_items = await self._collect_from_source(source_config)
                        results[source_name] = data_items
                        
                        # 更新缓存
                        self.data_cache[source_name] = data_items
                        source_config.last_updated = datetime.now()
                        
                        logger.info(f"Collected {len(data_items)} items from {source_name}")
                    else:
                        # 使用缓存数据
                        results[source_name] = self.data_cache.get(source_name, [])
                        logger.info(f"Using cached data for {source_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to collect from {source_name}: {e}")
                    results[source_name] = []
            
            return results
            
        finally:
            await self.close_session()
    
    async def collect_from_source(self, source_name: str) -> List[DataItem]:
        """
        从指定数据源采集数据
        
        Args:
            source_name: 数据源名称
            
        Returns:
            数据项列表
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        source_config = self.data_sources[source_name]
        
        if not source_config.enabled:
            logger.warning(f"Data source {source_name} is disabled")
            return []
        
        await self.start_session()
        
        try:
            return await self._collect_from_source(source_config)
        finally:
            await self.close_session()
    
    async def _collect_from_source(self, source: DataSource) -> List[DataItem]:
        """从单个数据源采集数据"""
        if source.source_type == "rss":
            return await self._collect_rss_data(source)
        elif source.source_type == "api":
            return await self._collect_api_data(source)
        elif source.source_type == "web":
            return await self._collect_web_data(source)
        elif source.source_type == "file":
            return await self._collect_file_data(source)
        else:
            logger.error(f"Unsupported source type: {source.source_type}")
            return []
    
    async def _collect_rss_data(self, source: DataSource) -> List[DataItem]:
        """采集RSS数据"""
        try:
            if not source.url:
                return []
            
            # 使用feedparser解析RSS
            # 注意：feedparser是同步的，在生产环境中可能需要异步版本
            feed = feedparser.parse(source.url)
            
            data_items = []
            
            for entry in feed.entries[:50]:  # 限制最多50条
                # 确定数据类型
                data_type = self._determine_data_type(source.name, entry.get('title', ''))
                
                # 解析发布时间
                publish_time = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    publish_time = datetime(*entry.published_parsed[:6])
                
                # 创建数据项
                data_item = DataItem(
                    id=f"{source.name}_{hash(entry.get('link', entry.get('title', '')))}",
                    source=source.name,
                    data_type=data_type,
                    title=entry.get('title', ''),
                    content=self._clean_content(entry.get('summary', entry.get('description', ''))),
                    url=entry.get('link'),
                    publish_time=publish_time,
                    metadata={
                        'author': entry.get('author', ''),
                        'tags': [tag.term for tag in entry.get('tags', [])],
                        'feed_title': feed.feed.get('title', '')
                    }
                )
                
                data_items.append(data_item)
            
            return data_items
            
        except Exception as e:
            logger.error(f"Failed to collect RSS data from {source.url}: {e}")
            return []
    
    async def _collect_api_data(self, source: DataSource) -> List[DataItem]:
        """采集API数据"""
        try:
            if not source.url or not self.session:
                return []
            
            async with self.session.get(
                source.url,
                headers=source.headers,
                params=source.params
            ) as response:
                if response.status != 200:
                    logger.error(f"API request failed: {response.status}")
                    return []
                
                data = await response.json()
                
                # 这里需要根据具体API格式解析数据
                # 以下是示例实现
                data_items = []
                
                if isinstance(data, dict) and 'items' in data:
                    items = data['items']
                elif isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                for item in items[:50]:  # 限制最多50条
                    data_type = self._determine_data_type(source.name, item.get('title', ''))
                    
                    data_item = DataItem(
                        id=f"{source.name}_{item.get('id', hash(str(item)))}",
                        source=source.name,
                        data_type=data_type,
                        title=item.get('title', item.get('headline', '')),
                        content=self._clean_content(item.get('content', item.get('body', ''))),
                        url=item.get('url', item.get('link')),
                        publish_time=self._parse_datetime(item.get('publish_time', item.get('created_at'))),
                        metadata=item
                    )
                    
                    data_items.append(data_item)
                
                return data_items
                
        except Exception as e:
            logger.error(f"Failed to collect API data from {source.url}: {e}")
            return []
    
    async def _collect_web_data(self, source: DataSource) -> List[DataItem]:
        """采集网页数据（简化实现）"""
        # 这里可以实现网页爬虫逻辑
        # 为了简化，暂时返回空列表
        logger.warning(f"Web scraping not implemented for {source.name}")
        return []
    
    async def _collect_file_data(self, source: DataSource) -> List[DataItem]:
        """采集文件数据"""
        try:
            if not source.url:
                return []
            
            file_path = Path(source.url)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return []
            
            data_items = []
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        data_item = DataItem(
                            id=f"{source.name}_{item.get('id', hash(str(item)))}",
                            source=source.name,
                            data_type=item.get('type', 'unknown'),
                            title=item.get('title', ''),
                            content=item.get('content', ''),
                            url=item.get('url'),
                            metadata=item
                        )
                        data_items.append(data_item)
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    data_item = DataItem(
                        id=f"{source.name}_{hash(str(row.to_dict()))}",
                        source=source.name,
                        data_type=row.get('type', 'unknown'),
                        title=row.get('title', ''),
                        content=row.get('content', ''),
                        url=row.get('url'),
                        metadata=row.to_dict()
                    )
                    data_items.append(data_item)
            
            return data_items
            
        except Exception as e:
            logger.error(f"Failed to collect file data from {source.url}: {e}")
            return []
    
    def _should_update(self, source: DataSource) -> bool:
        """检查是否需要更新数据"""
        if not source.last_updated:
            return True
        
        time_since_update = datetime.now() - source.last_updated
        return time_since_update.total_seconds() >= source.update_interval
    
    def _determine_data_type(self, source_name: str, title: str) -> str:
        """根据数据源和标题确定数据类型"""
        title_lower = title.lower()
        
        # 政策相关关键词
        policy_keywords = ['政策', '通知', '意见', '规划', '方案', '办法', '条例', '规定']
        if any(keyword in title_lower for keyword in policy_keywords):
            return 'policy'
        
        # 财务相关关键词
        financial_keywords = ['财报', '业绩', '营收', '利润', '财务', '年报', '季报']
        if any(keyword in title_lower for keyword in financial_keywords):
            return 'financial'
        
        # 市场相关关键词
        market_keywords = ['股价', '涨跌', '交易', '成交', '指数', '市场']
        if any(keyword in title_lower for keyword in market_keywords):
            return 'market'
        
        # 默认为新闻
        return 'news'
    
    def _clean_content(self, content: str) -> str:
        """清洗文本内容"""
        if not content:
            return ""
        
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 移除多余空白
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 限制长度
        if len(content) > 5000:
            content = content[:5000] + "..."
        
        return content
    
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """解析日期时间字符串"""
        if not dt_str:
            return None
        
        try:
            # 尝试多种日期格式
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
            
            logger.warning(f"Failed to parse datetime: {dt_str}")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing datetime {dt_str}: {e}")
            return None
    
    def add_data_source(self, name: str, source: DataSource):
        """添加数据源"""
        self.data_sources[name] = source
        logger.info(f"Added data source: {name}")
    
    def remove_data_source(self, name: str):
        """移除数据源"""
        if name in self.data_sources:
            del self.data_sources[name]
            if name in self.data_cache:
                del self.data_cache[name]
            logger.info(f"Removed data source: {name}")
    
    def get_data_sources(self) -> Dict[str, DataSource]:
        """获取所有数据源配置"""
        return self.data_sources.copy()
    
    def get_cached_data(self, source_name: Optional[str] = None) -> Union[List[DataItem], Dict[str, List[DataItem]]]:
        """获取缓存数据"""
        if source_name:
            return self.data_cache.get(source_name, [])
        return self.data_cache.copy()
    
    async def save_data_to_file(self, data: Dict[str, List[DataItem]], filename: str):
        """保存数据到文件"""
        try:
            file_path = self.data_dir / filename
            
            # 转换为可序列化格式
            serializable_data = {}
            for source, items in data.items():
                serializable_data[source] = [
                    item.dict() for item in items
                ]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save data to file: {e}")
    
    async def load_data_from_file(self, filename: str) -> Dict[str, List[DataItem]]:
        """从文件加载数据"""
        try:
            file_path = self.data_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换回DataItem对象
            result = {}
            for source, items in data.items():
                result[source] = [
                    DataItem(**item) for item in items
                ]
            
            logger.info(f"Data loaded from {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load data from file: {e}")
            return {}


# 全局数据采集器实例
data_collector = DataCollector()
