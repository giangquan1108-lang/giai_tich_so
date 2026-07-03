"""Schemas for interpolation module."""
from pydantic import BaseModel
from typing import Optional, List


class InterpolationRequest(BaseModel):
    """Request schema for interpolation."""
    x_points: List[float]
    y_points: List[float]
    x_value: Optional[float] = None
    method: str = "lagrange"  # lagrange, newton_forward, newton_backward, divided_differences


class InterpolationResponse(BaseModel):
    """Response schema for interpolation."""
    success: bool
    message: str
    interpolated_value: Optional[float] = None
    polynomial: Optional[str] = None
    divided_diff_table: Optional[List[List[float]]] = None
    iterations: List[dict] = []
    formula: str = ""
    plot_data: Optional[dict] = None