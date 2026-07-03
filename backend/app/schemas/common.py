"""Common response schemas used across all modules."""
from pydantic import BaseModel
from typing import Any, List, Optional


class IterationRow(BaseModel):
    """A single iteration step."""
    k: int
    values: dict
    error: Optional[float] = None


class BaseResponse(BaseModel):
    """Unified response schema for all endpoints."""
    success: bool
    message: str
    result: Any = None
    iterations: Optional[List[dict]] = None
    formula: Optional[str] = None
    error: Optional[str] = None