"""
InvestIQ Platform - Scoring Service Industry Router
行业评分API路由
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date, datetime

from shared.utils import get_db, create_response, create_error_response, get_request_id
from shared.models import IndustryScoreSnapshot
from ..schemas.industry import (
    IndustryScoreRequest,
    IndustryScoreResponse,
    IndustryScoreListResponse,
    IndustryScoreSnapshotResponse
)
from ..services.industry_service import IndustryService


router = APIRouter()


@router.post("/score", response_model=IndustryScoreResponse)
async def calculate_industry_score(
    request: Request,
    score_request: IndustryScoreRequest,
    db: Session = Depends(get_db)
):
    """计算并保存行业评分"""
    try:
        service = IndustryService(db)
        
        # 计算评分
        snapshot = await service.create_industry_score(
            industry_id=score_request.industry_id,
            P=score_request.P,
            E=score_request.E,
            M=score_request.M,
            R_neg=score_request.R_neg,
            evidences=score_request.evidences or [],
            created_by=score_request.created_by or "system"
        )
        
        request_id = get_request_id(request)
        
        return create_response(
            data=IndustryScoreResponse.from_snapshot(snapshot),
            message="Industry score calculated successfully",
            request_id=request_id
        )
        
    except ValueError as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                message=str(e),
                code=400,
                request_id=request_id
            )
        )
    except Exception as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Internal server error",
                code=500,
                request_id=request_id,
                details=str(e)
            )
        )


@router.get("/score/{industry_id}", response_model=IndustryScoreSnapshotResponse)
async def get_industry_score(
    request: Request,
    industry_id: str,
    as_of: Optional[date] = Query(None, description="评分日期，默认为最新"),
    db: Session = Depends(get_db)
):
    """获取行业评分"""
    try:
        service = IndustryService(db)
        snapshot = await service.get_latest_score(industry_id, as_of)
        
        if not snapshot:
            request_id = get_request_id(request)
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    message=f"Industry score not found for {industry_id}",
                    code=404,
                    request_id=request_id
                )
            )
        
        request_id = get_request_id(request)
        return create_response(
            data=IndustryScoreSnapshotResponse.from_snapshot(snapshot),
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Internal server error",
                code=500,
                request_id=request_id,
                details=str(e)
            )
        )


@router.get("/scores", response_model=IndustryScoreListResponse)
async def list_industry_scores(
    request: Request,
    as_of: Optional[date] = Query(None, description="评分日期，默认为最新"),
    min_score: Optional[float] = Query(None, description="最低评分"),
    max_score: Optional[float] = Query(None, description="最高评分"),
    qualified_only: bool = Query(False, description="仅返回符合入池条件的行业"),
    core_only: bool = Query(False, description="仅返回核心候选行业"),
    skip: int = Query(0, ge=0, description="跳过记录数"),
    limit: int = Query(100, ge=1, le=1000, description="返回记录数"),
    db: Session = Depends(get_db)
):
    """获取行业评分列表"""
    try:
        service = IndustryService(db)
        
        snapshots, total = await service.list_scores(
            as_of=as_of,
            min_score=min_score,
            max_score=max_score,
            qualified_only=qualified_only,
            core_only=core_only,
            skip=skip,
            limit=limit
        )
        
        request_id = get_request_id(request)
        
        return create_response(
            data={
                "items": [IndustryScoreSnapshotResponse.from_snapshot(s) for s in snapshots],
                "total": total,
                "skip": skip,
                "limit": limit,
                "qualified_count": len([s for s in snapshots if s.is_qualified_for_pool()]),
                "core_count": len([s for s in snapshots if s.is_core_candidate()])
            },
            request_id=request_id
        )
        
    except Exception as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Internal server error",
                code=500,
                request_id=request_id,
                details=str(e)
            )
        )


@router.get("/score/{industry_id}/history", response_model=IndustryScoreListResponse)
async def get_industry_score_history(
    request: Request,
    industry_id: str,
    start_date: Optional[date] = Query(None, description="开始日期"),
    end_date: Optional[date] = Query(None, description="结束日期"),
    limit: int = Query(30, ge=1, le=365, description="返回记录数"),
    db: Session = Depends(get_db)
):
    """获取行业评分历史"""
    try:
        service = IndustryService(db)
        
        snapshots = await service.get_score_history(
            industry_id=industry_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        request_id = get_request_id(request)
        
        return create_response(
            data={
                "items": [IndustryScoreSnapshotResponse.from_snapshot(s) for s in snapshots],
                "industry_id": industry_id,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None
                }
            },
            request_id=request_id
        )
        
    except Exception as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Internal server error",
                code=500,
                request_id=request_id,
                details=str(e)
            )
        )


@router.put("/score/{industry_id}", response_model=IndustryScoreResponse)
async def update_industry_score(
    request: Request,
    industry_id: str,
    score_request: IndustryScoreRequest,
    db: Session = Depends(get_db)
):
    """更新行业评分"""
    try:
        if score_request.industry_id != industry_id:
            request_id = get_request_id(request)
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    message="Industry ID in URL and request body must match",
                    code=400,
                    request_id=request_id
                )
            )
        
        service = IndustryService(db)
        
        snapshot = await service.update_industry_score(
            industry_id=industry_id,
            P=score_request.P,
            E=score_request.E,
            M=score_request.M,
            R_neg=score_request.R_neg,
            evidences=score_request.evidences or [],
            created_by=score_request.created_by or "system"
        )
        
        request_id = get_request_id(request)
        
        return create_response(
            data=IndustryScoreResponse.from_snapshot(snapshot),
            message="Industry score updated successfully",
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=400,
            detail=create_error_response(
                message=str(e),
                code=400,
                request_id=request_id
            )
        )
    except Exception as e:
        request_id = get_request_id(request)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Internal server error",
                code=500,
                request_id=request_id,
                details=str(e)
            )
        )