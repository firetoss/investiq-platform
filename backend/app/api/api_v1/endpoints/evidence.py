"""
InvestIQ Platform - 证据管理API端点
证据收集、管理和审计服务的REST API
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any, BinaryIO
from fastapi import APIRouter, Depends, HTTPException, Header, Query, UploadFile, File, Form
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

from backend.app.core.database import get_db
from backend.app.core.logging import get_logger
from backend.app.core.config import settings
from backend.app.models.evidence import (
    EvidenceItem, DecisionLog, User, AuditSummary, DataLineage,
    EvidenceType, EvidenceSource
)
from backend.app.services.evidence_service import evidence_service
from backend.app.api import API_DESCRIPTIONS

logger = get_logger(__name__)

router = APIRouter()


# Pydantic模型定义
class EvidenceAttachRequest(BaseModel):
    """证据绑定请求模型"""
    entity_type: str = Field(..., description="实体类型")
    entity_id: str = Field(..., description="实体ID")
    evidence_type: str = Field(..., description="证据类型")
    title: str = Field(..., description="证据标题")
    description: Optional[str] = Field(None, description="证据描述")
    source: str = Field(..., description="证据来源")
    source_url: Optional[str] = Field(None, description="来源URL")
    source_name: Optional[str] = Field(None, description="来源名称")
    content: Optional[str] = Field(None, description="证据内容")
    evidence_date: Optional[date] = Field(None, description="证据日期")
    published_at: Optional[datetime] = Field(None, description="发布时间")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    keywords: Optional[List[str]] = Field(None, description="关键词列表")
    extra_data: Optional[Dict[str, Any]] = Field(None, description="扩展数据")

    @validator("entity_type")
    def validate_entity_type(cls, v):
        valid_types = ["industry", "equity", "portfolio"]
        if v not in valid_types:
            raise ValueError(f"实体类型必须为: {valid_types}")
        return v

    @validator("evidence_type")
    def validate_evidence_type(cls, v):
        valid_types = ["policy", "order", "financial", "news", "announcement", "research", "market_data", "other"]
        if v not in valid_types:
            raise ValueError(f"证据类型必须为: {valid_types}")
        return v

    @validator("source")
    def validate_source(cls, v):
        valid_sources = ["official", "authorized", "public"]
        if v not in valid_sources:
            raise ValueError(f"证据来源必须为: {valid_sources}")
        return v


class EvidenceVerifyRequest(BaseModel):
    """证据验证请求模型"""
    verified_by: str = Field(..., description="验证人")
    verification_note: Optional[str] = Field(None, description="验证备注")
    confidence_score: Optional[float] = Field(None, ge=0, le=100, description="置信度评分")
    relevance_score: Optional[float] = Field(None, ge=0, le=100, description="相关性评分")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="质量评分")


class AuditLogRequest(BaseModel):
    """审计日志请求模型"""
    user_id: str = Field(..., description="操作用户")
    action: str = Field(..., description="操作动作")
    resource: Optional[str] = Field(None, description="操作资源")
    resource_id: Optional[str] = Field(None, description="资源ID")
    payload: Optional[Dict[str, Any]] = Field(None, description="操作载荷")
    before_state: Optional[Dict[str, Any]] = Field(None, description="操作前状态")
    after_state: Optional[Dict[str, Any]] = Field(None, description="操作后状态")
    business_context: Optional[Dict[str, Any]] = Field(None, description="业务上下文")
    risk_level: Optional[str] = Field("medium", description="风险级别")


class EvidenceResponse(BaseModel):
    """证据响应模型"""
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    error: Optional[str] = Field(None, description="错误信息")
    request_id: Optional[str] = Field(None, description="请求ID")
    snapshot_ts: str = Field(..., description="快照时间戳")


# API端点定义
@router.post(
    "/attach",
    response_model=EvidenceResponse,
    summary="绑定证据",
    description="为实体绑定证据记录"
)
async def attach_evidence(
    request: EvidenceAttachRequest,
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """绑定证据"""
    try:
        result = await evidence_service.attach_evidence(
            entity_type=request.entity_type,
            entity_id=request.entity_id,
            evidence_type=request.evidence_type,
            title=request.title,
            description=request.description,
            source=request.source,
            source_url=request.source_url,
            source_name=request.source_name,
            content=request.content,
            evidence_date=request.evidence_date,
            published_at=request.published_at,
            tags=request.tags,
            keywords=request.keywords,
            extra_data=request.extra_data,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Attach evidence failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/upload",
    response_model=EvidenceResponse,
    summary="上传证据文件",
    description="上传证据文件到对象存储"
)
async def upload_evidence_file(
    file: UploadFile = File(..., description="证据文件"),
    entity_type: str = Form(..., description="实体类型"),
    entity_id: str = Form(..., description="实体ID"),
    evidence_type: str = Form(..., description="证据类型"),
    title: str = Form(..., description="证据标题"),
    description: Optional[str] = Form(None, description="证据描述"),
    source: str = Form("public", description="证据来源"),
    tags: Optional[str] = Form(None, description="标签列表，逗号分隔"),
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """上传证据文件"""
    try:
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else None
        
        result = await evidence_service.upload_evidence_file(
            file=file,
            entity_type=entity_type,
            entity_id=entity_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            source=source,
            tags=tag_list,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Upload evidence file failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/by-entity",
    response_model=EvidenceResponse,
    summary="按实体获取证据",
    description="根据实体类型和ID获取关联的证据列表"
)
async def get_evidence_by_entity(
    entity_type: str = Query(..., description="实体类型"),
    entity_id: str = Query(..., description="实体ID"),
    db: AsyncSession = Depends(get_db),
    evidence_type: Optional[str] = Query(None, description="证据类型过滤"),
    source: Optional[str] = Query(None, description="来源过滤"),
    verified_only: bool = Query(False, description="仅显示已验证的证据"),
    active_only: bool = Query(True, description="仅显示活跃的证据"),
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小")
):
    """按实体获取证据"""
    try:
        result = await evidence_service.get_evidence_by_entity(
            entity_type=entity_type,
            entity_id=entity_id,
            evidence_type=evidence_type,
            source=source,
            verified_only=verified_only,
            active_only=active_only,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get evidence by entity failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.put(
    "/{evidence_id}/verify",
    response_model=EvidenceResponse,
    summary="验证证据",
    description="对证据进行验证和评分"
)
async def verify_evidence(
    evidence_id: str,
    request: EvidenceVerifyRequest,
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """验证证据"""
    try:
        result = await evidence_service.verify_evidence(
            evidence_id=evidence_id,
            verified_by=request.verified_by,
            verification_note=request.verification_note,
            confidence_score=request.confidence_score,
            relevance_score=request.relevance_score,
            quality_score=request.quality_score,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Verify evidence failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.delete(
    "/{evidence_id}",
    response_model=EvidenceResponse,
    summary="删除证据",
    description="删除证据记录（需要审批）"
)
async def delete_evidence(
    evidence_id: str,
    reason: str = Query(..., description="删除原因"),
    user_id: str = Query(..., description="操作用户"),
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """删除证据"""
    try:
        result = await evidence_service.delete_evidence(
            evidence_id=evidence_id,
            reason=reason,
            user_id=user_id,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Delete evidence failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/audit/log",
    response_model=EvidenceResponse,
    summary="记录审计日志",
    description="记录用户操作的审计日志"
)
async def log_audit_trail(
    request: AuditLogRequest,
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    client_ip: Optional[str] = Header(None, alias="X-Real-IP"),
    user_agent: Optional[str] = Header(None, alias="User-Agent")
):
    """记录审计日志"""
    try:
        result = await evidence_service.log_audit_trail(
            user_id=request.user_id,
            action=request.action,
            resource=request.resource,
            resource_id=request.resource_id,
            payload=request.payload,
            before_state=request.before_state,
            after_state=request.after_state,
            business_context=request.business_context,
            risk_level=request.risk_level,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Log audit trail failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/audit/trail",
    response_model=EvidenceResponse,
    summary="获取审计轨迹",
    description="获取审计日志轨迹"
)
async def get_audit_trail(
    db: AsyncSession = Depends(get_db),
    user_id: Optional[str] = Query(None, description="用户过滤"),
    action: Optional[str] = Query(None, description="操作过滤"),
    resource: Optional[str] = Query(None, description="资源过滤"),
    resource_id: Optional[str] = Query(None, description="资源ID过滤"),
    risk_level: Optional[str] = Query(None, description="风险级别过滤"),
    start_date: Optional[datetime] = Query(None, description="开始时间"),
    end_date: Optional[datetime] = Query(None, description="结束时间"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=200, description="每页大小")
):
    """获取审计轨迹"""
    try:
        result = await evidence_service.get_audit_trail(
            user_id=user_id,
            action=action,
            resource=resource,
            resource_id=resource_id,
            risk_level=risk_level,
            start_date=start_date,
            end_date=end_date,
            page=page,
            page_size=page_size,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get audit trail failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.post(
    "/audit/verify-chain",
    response_model=EvidenceResponse,
    summary="验证审计链完整性",
    description="验证审计链的完整性和一致性"
)
async def verify_audit_chain(
    start_sequence: Optional[int] = Query(None, description="起始序列号"),
    end_sequence: Optional[int] = Query(None, description="结束序列号"),
    db: AsyncSession = Depends(get_db),
    request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """验证审计链完整性"""
    try:
        result = await evidence_service.verify_audit_chain(
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Verify audit chain failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            request_id=request_id,
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/audit/proof/{sequence_number}",
    response_model=EvidenceResponse,
    summary="获取审计证明",
    description="获取特定记录的审计证明材料"
)
async def get_audit_proof(
    sequence_number: int,
    db: AsyncSession = Depends(get_db)
):
    """获取审计证明"""
    try:
        result = await evidence_service.get_audit_proof(
            sequence_number=sequence_number,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get audit proof failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/search",
    response_model=EvidenceResponse,
    summary="搜索证据",
    description="全文搜索证据内容"
)
async def search_evidence(
    query: str = Query(..., description="搜索查询"),
    db: AsyncSession = Depends(get_db),
    entity_type: Optional[str] = Query(None, description="实体类型过滤"),
    evidence_type: Optional[str] = Query(None, description="证据类型过滤"),
    source: Optional[str] = Query(None, description="来源过滤"),
    verified_only: bool = Query(False, description="仅搜索已验证的证据"),
    min_score: float = Query(0.0, description="最小相关性评分"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小")
):
    """搜索证据"""
    try:
        result = await evidence_service.search_evidence(
            query=query,
            entity_type=entity_type,
            evidence_type=evidence_type,
            source=source,
            verified_only=verified_only,
            min_score=min_score,
            page=page,
            page_size=page_size,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Search evidence failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/statistics",
    response_model=EvidenceResponse,
    summary="证据统计",
    description="获取证据管理的统计信息"
)
async def get_evidence_statistics(
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="统计天数")
):
    """证据统计"""
    try:
        result = await evidence_service.get_evidence_statistics(
            days=days,
            db=db
        )
        
        return EvidenceResponse(
            success=True,
            data=result,
            snapshot_ts=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Get evidence statistics failed: {e}", exc_info=True)
        return EvidenceResponse(
            success=False,
            error=str(e),
            snapshot_ts=datetime.utcnow().isoformat()
        )


@router.get(
    "/config/default",
    summary="获取默认配置",
    description="获取证据管理的默认配置"
)
async def get_default_config():
    """获取默认配置"""
    return {
        "evidence_types": {
            "policy": {"name": "政策文件", "description": "政府政策、法规文件"},
            "order": {"name": "订单/招标", "description": "订单信息、招标公告、中标结果"},
            "financial": {"name": "财务数据", "description": "财报、业绩公告、财务指标"},
            "news": {"name": "新闻报道", "description": "新闻报道、媒体文章"},
            "announcement": {"name": "公司公告", "description": "上市公司公告、通知"},
            "research": {"name": "研究报告", "description": "券商研报、行业报告"},
            "market_data": {"name": "市场数据", "description": "价格数据、交易数据"},
            "other": {"name": "其他", "description": "其他类型证据"}
        },
        "evidence_sources": {
            "official": {"name": "官方来源", "weight": 1.0, "description": "政府官网、交易所等"},
            "authorized": {"name": "授权来源", "weight": 0.9, "description": "授权数据提供商"},
            "public": {"name": "公开抓取", "weight": 0.8, "description": "公开网站抓取"}
        },
        "file_types": {
            "supported": ["pdf", "doc", "docx", "xls", "xlsx", "txt", "html", "json", "csv"],
            "max_size_mb": 50,
            "virus_scan": True
        },
        "retention_policy": {
            "hot_days": settings.audit.get("hot_days", 180),
            "cold_days": settings.audit.get("cold_days", 1825),
            "versioning_enabled": True
        },
        "quality_thresholds": {
            "confidence_min": 60.0,
            "relevance_min": 70.0,
            "quality_min": 65.0
        },
        "audit_config": {
            "hash_algorithm": settings.audit.get("hash_algo", "SHA-256"),
            "immutable_chain": True,
            "merkle_tree_enabled": True,
            "daily_summary": True
        }
    }


@router.get(
    "/health",
    summary="证据管理服务健康检查",
    description="检查证据管理服务状态"
)
async def evidence_health_check(
    db: AsyncSession = Depends(get_db)
):
    """证据管理服务健康检查"""
    try:
        # 检查数据库连接
        await db.execute("SELECT 1")
        
        # 检查证据服务状态
        health_status = await evidence_service.health_check(db)
        
        return {
            "status": "healthy",
            "service_status": health_status,
            "features": {
                "evidence_management": True,
                "file_upload": True,
                "audit_trail": True,
                "chain_verification": True,
                "full_text_search": True,
                "data_lineage": True
            },
            "storage": {
                "minio_url": settings.MINIO_URL,
                "bucket": settings.MINIO_BUCKET_NAME,
                "versioning": True,
                "lifecycle_management": True
            },
            "security": {
                "content_hashing": "SHA-256",
                "immutable_audit": True,
                "access_control": "RBAC"
            },
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Evidence health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"证据管理服务健康检查失败: {e}"
        )