"""FastAPI dependencies."""

from fastapi import Header, HTTPException

from rag_eval.config import get_settings


def verify_optional_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = get_settings().api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")
