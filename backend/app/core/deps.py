"""
轻量依赖 - 单用户模式的当前用户获取
可选启用 SIMPLE_API_KEY 校验；未设置时直接返回本地 owner 用户。
"""

import os
import uuid
from fastapi import Request, HTTPException, status

from backend.app.models.user import User


_OWNER_USER = User(
    id=uuid.uuid4(),
    username="owner",
    full_name="Local Owner",
    is_active=True,
)


async def get_current_user(request: Request) -> User:
    """
    返回当前用户（单用户模式）。
    若设置 SIMPLE_API_KEY，则要求请求头 X-API-Key 匹配。
    """
    api_key = os.getenv("SIMPLE_API_KEY")
    if api_key:
        provided = request.headers.get("X-API-Key")
        if not provided or provided != api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return _OWNER_USER

