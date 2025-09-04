"""
数据采集API端点
提供数据源管理和数据采集功能
"""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging

from pydantic import BaseModel, Field

from backend.app.services.data_collector import (
    data_collector,
    DataSource,
    DataItem
)
from backend.app.core.deps import get_current_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


class DataSourceRequest(BaseModel):
    """数据源请求"""
    name: str = Field(..., description="数据源名称")
    source_type: str = Field(..., description="数据源类型", regex="^(rss|api|web|file)$")
    url: Optional[str] = Field(None, description="数据源URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="请求头")
    params: Dict[str, Any] = Field(default_factory=dict, description="请求参数")
    enabled: bool = Field(default=True, description="是否启用")
    update_interval: int = Field(default=3600, description="更新间隔（秒）", ge=60)


class DataCollectionRequest(BaseModel):
    """数据采集请求"""
    source_names: Optional[List[str]] = Field(None, description="指定数据源名称列表，为空则采集所有")
    save_to_file: bool = Field(default=False, description="是否保存到文件")
    filename: Optional[str] = Field(None, description="保存文件名")


@router.get("/sources")
async def get_data_sources(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取所有数据源配置
    
    Args:
        current_user: 当前用户
        
    Returns:
        数据源配置列表
    """
    try:
        logger.info(f"User {current_user.id} requesting data sources")
        
        sources = data_collector.get_data_sources()
        
        # 转换为可序列化格式
        sources_dict = {}
        for name, source in sources.items():
            sources_dict[name] = source.dict()
        
        return {
            "status": "success",
            "sources": sources_dict,
            "total_sources": len(sources_dict),
            "enabled_sources": sum(1 for s in sources.values() if s.enabled)
        }
        
    except Exception as e:
        logger.error(f"Get data sources failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据源失败: {str(e)}")


@router.post("/sources")
async def add_data_source(
    request: DataSourceRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    添加数据源
    
    Args:
        request: 数据源配置
        current_user: 当前用户
        
    Returns:
        添加结果
    """
    try:
        logger.info(f"User {current_user.id} adding data source: {request.name}")
        
        # 检查数据源是否已存在
        existing_sources = data_collector.get_data_sources()
        if request.name in existing_sources:
            raise HTTPException(status_code=400, detail=f"数据源 {request.name} 已存在")
        
        # 创建数据源配置
        source = DataSource(
            name=request.name,
            source_type=request.source_type,
            url=request.url,
            headers=request.headers,
            params=request.params,
            enabled=request.enabled,
            update_interval=request.update_interval
        )
        
        # 添加数据源
        data_collector.add_data_source(request.name, source)
        
        return {
            "status": "success",
            "message": f"数据源 {request.name} 添加成功",
            "source": source.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add data source failed: {e}")
        raise HTTPException(status_code=500, detail=f"添加数据源失败: {str(e)}")


@router.put("/sources/{source_name}")
async def update_data_source(
    source_name: str,
    request: DataSourceRequest,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    更新数据源配置
    
    Args:
        source_name: 数据源名称
        request: 新的数据源配置
        current_user: 当前用户
        
    Returns:
        更新结果
    """
    try:
        logger.info(f"User {current_user.id} updating data source: {source_name}")
        
        # 检查数据源是否存在
        existing_sources = data_collector.get_data_sources()
        if source_name not in existing_sources:
            raise HTTPException(status_code=404, detail=f"数据源 {source_name} 不存在")
        
        # 创建新的数据源配置
        source = DataSource(
            name=request.name,
            source_type=request.source_type,
            url=request.url,
            headers=request.headers,
            params=request.params,
            enabled=request.enabled,
            update_interval=request.update_interval
        )
        
        # 如果名称改变，需要先删除旧的
        if source_name != request.name:
            data_collector.remove_data_source(source_name)
        
        # 添加/更新数据源
        data_collector.add_data_source(request.name, source)
        
        return {
            "status": "success",
            "message": f"数据源 {source_name} 更新成功",
            "source": source.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update data source failed: {e}")
        raise HTTPException(status_code=500, detail=f"更新数据源失败: {str(e)}")


@router.delete("/sources/{source_name}")
async def delete_data_source(
    source_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    删除数据源
    
    Args:
        source_name: 数据源名称
        current_user: 当前用户
        
    Returns:
        删除结果
    """
    try:
        logger.info(f"User {current_user.id} deleting data source: {source_name}")
        
        # 检查数据源是否存在
        existing_sources = data_collector.get_data_sources()
        if source_name not in existing_sources:
            raise HTTPException(status_code=404, detail=f"数据源 {source_name} 不存在")
        
        # 删除数据源
        data_collector.remove_data_source(source_name)
        
        return {
            "status": "success",
            "message": f"数据源 {source_name} 删除成功"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete data source failed: {e}")
        raise HTTPException(status_code=500, detail=f"删除数据源失败: {str(e)}")


@router.post("/collect")
async def collect_data(
    request: DataCollectionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    采集数据
    
    Args:
        request: 数据采集请求
        background_tasks: 后台任务
        current_user: 当前用户
        
    Returns:
        采集结果
    """
    try:
        logger.info(f"User {current_user.id} requesting data collection")
        
        if request.source_names:
            # 采集指定数据源
            results = {}
            for source_name in request.source_names:
                try:
                    data_items = await data_collector.collect_from_source(source_name)
                    results[source_name] = data_items
                    logger.info(f"Collected {len(data_items)} items from {source_name}")
                except Exception as e:
                    logger.error(f"Failed to collect from {source_name}: {e}")
                    results[source_name] = []
        else:
            # 采集所有数据源
            results = await data_collector.collect_all_data()
        
        # 统计结果
        total_items = sum(len(items) for items in results.values())
        
        # 保存到文件（如果需要）
        if request.save_to_file:
            filename = request.filename or f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            background_tasks.add_task(data_collector.save_data_to_file, results, filename)
        
        # 转换为可序列化格式
        serializable_results = {}
        for source, items in results.items():
            serializable_results[source] = [item.dict() for item in items]
        
        return {
            "status": "success",
            "message": f"数据采集完成，共采集 {total_items} 条数据",
            "results": serializable_results,
            "summary": {
                "total_sources": len(results),
                "total_items": total_items,
                "sources_summary": {
                    source: len(items) for source, items in results.items()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"数据采集失败: {str(e)}")


@router.get("/collect/{source_name}")
async def collect_from_source(
    source_name: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    从指定数据源采集数据
    
    Args:
        source_name: 数据源名称
        current_user: 当前用户
        
    Returns:
        采集结果
    """
    try:
        logger.info(f"User {current_user.id} collecting from source: {source_name}")
        
        data_items = await data_collector.collect_from_source(source_name)
        
        # 转换为可序列化格式
        serializable_items = [item.dict() for item in data_items]
        
        return {
            "status": "success",
            "source": source_name,
            "items": serializable_items,
            "total_items": len(data_items)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Collect from source failed: {e}")
        raise HTTPException(status_code=500, detail=f"数据采集失败: {str(e)}")


@router.get("/cache")
async def get_cached_data(
    source_name: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取缓存数据
    
    Args:
        source_name: 数据源名称（可选）
        current_user: 当前用户
        
    Returns:
        缓存数据
    """
    try:
        logger.info(f"User {current_user.id} requesting cached data")
        
        cached_data = data_collector.get_cached_data(source_name)
        
        if source_name:
            # 单个数据源
            serializable_data = [item.dict() for item in cached_data]
            return {
                "status": "success",
                "source": source_name,
                "items": serializable_data,
                "total_items": len(cached_data)
            }
        else:
            # 所有数据源
            serializable_data = {}
            total_items = 0
            
            for source, items in cached_data.items():
                serializable_data[source] = [item.dict() for item in items]
                total_items += len(items)
            
            return {
                "status": "success",
                "data": serializable_data,
                "summary": {
                    "total_sources": len(cached_data),
                    "total_items": total_items,
                    "sources_summary": {
                        source: len(items) for source, items in cached_data.items()
                    }
                }
            }
        
    except Exception as e:
        logger.error(f"Get cached data failed: {e}")
        raise HTTPException(status_code=500, detail=f"获取缓存数据失败: {str(e)}")


@router.post("/save")
async def save_data_to_file(
    filename: str,
    source_names: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    保存数据到文件
    
    Args:
        filename: 文件名
        source_names: 数据源名称列表（可选）
        current_user: 当前用户
        
    Returns:
        保存结果
    """
    try:
        logger.info(f"User {current_user.id} saving data to file: {filename}")
        
        # 获取要保存的数据
        if source_names:
            data_to_save = {}
            for source_name in source_names:
                cached_data = data_collector.get_cached_data(source_name)
                data_to_save[source_name] = cached_data
        else:
            data_to_save = data_collector.get_cached_data()
        
        # 保存到文件
        await data_collector.save_data_to_file(data_to_save, filename)
        
        total_items = sum(len(items) for items in data_to_save.values())
        
        return {
            "status": "success",
            "message": f"数据已保存到文件 {filename}",
            "filename": filename,
            "total_sources": len(data_to_save),
            "total_items": total_items
        }
        
    except Exception as e:
        logger.error(f"Save data to file failed: {e}")
        raise HTTPException(status_code=500, detail=f"保存数据失败: {str(e)}")


@router.post("/load")
async def load_data_from_file(
    filename: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    从文件加载数据
    
    Args:
        filename: 文件名
        current_user: 当前用户
        
    Returns:
        加载结果
    """
    try:
        logger.info(f"User {current_user.id} loading data from file: {filename}")
        
        loaded_data = await data_collector.load_data_from_file(filename)
        
        if not loaded_data:
            raise HTTPException(status_code=404, detail=f"文件 {filename} 不存在或为空")
        
        # 转换为可序列化格式
        serializable_data = {}
        total_items = 0
        
        for source, items in loaded_data.items():
            serializable_data[source] = [item.dict() for item in items]
            total_items += len(items)
        
        return {
            "status": "success",
            "message": f"数据已从文件 {filename} 加载",
            "filename": filename,
            "data": serializable_data,
            "summary": {
                "total_sources": len(loaded_data),
                "total_items": total_items,
                "sources_summary": {
                    source: len(items) for source, items in loaded_data.items()
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load data from file failed: {e}")
        raise HTTPException(status_code=500, detail=f"加载数据失败: {str(e)}")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    数据采集服务健康检查
    
    Returns:
        健康状态信息
    """
    try:
        # 获取数据源状态
        sources = data_collector.get_data_sources()
        cached_data = data_collector.get_cached_data()
        
        # 计算统计信息
        total_sources = len(sources)
        enabled_sources = sum(1 for s in sources.values() if s.enabled)
        total_cached_items = sum(len(items) for items in cached_data.values())
        
        return {
            "status": "healthy",
            "data_sources": {
                "total": total_sources,
                "enabled": enabled_sources,
                "disabled": total_sources - enabled_sources
            },
            "cached_data": {
                "total_items": total_cached_items,
                "sources_with_data": len(cached_data)
            },
            "timestamp": "2025-01-04T16:16:00+08:00"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-04T16:16:00+08:00"
        }
