"""Schemas for root finding module."""
from pydantic import BaseModel
from typing import Optional, List


class RootFindingRequest(BaseModel):
    """Request schema for root finding methods."""
    function: str
    a: Optional[float] = None
    b: Optional[float] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    epsilon: float = 1e-6
    max_iterations: int = 100
    method: str = "bisection"  # bisection, newton, secant, fixed_point


class IterationData(BaseModel):
    """Single iteration data."""
    k: int
    x_k: float
    f_x_k: float
    error: Optional[float] = None


class RootFindingResponse(BaseModel):
    """Response schema for root finding."""
    success: bool
    message: str
    root: Optional[float] = None
    f_root: Optional[float] = None
    iterations_count: int = 0
    final_error: float = 0.0
    iterations: List[dict] = []
    formula: str = ""
    convergence_data: Optional[dict] = None
    contraction_info: Optional[str] = None
