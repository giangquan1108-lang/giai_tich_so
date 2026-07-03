"""Schemas for numerical integration module."""
from pydantic import BaseModel
from typing import Optional, List


class IntegrationRequest(BaseModel):
    """Request schema for numerical integration."""
    function: str
    a: float
    b: float
    n: int = 100
    method: str = "trapezoidal"  # trapezoidal, simpson_13, simpson_38, romberg


class IntegrationResponse(BaseModel):
    """Response schema for numerical integration."""
    success: bool
    message: str
    result: Optional[float] = None
    exact_value: Optional[float] = None
    error: Optional[float] = None
    relative_error: Optional[float] = None
    iterations: List[dict] = []
    formula: str = ""
    plot_data: Optional[dict] = None