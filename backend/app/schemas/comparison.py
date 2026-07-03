"""Schemas for algorithm comparison module."""
from pydantic import BaseModel
from typing import Optional, List


class ComparisonRequest(BaseModel):
    """Request schema for algorithm comparison."""
    # Root finding comparison
    function: Optional[str] = None
    a: Optional[float] = None
    b: Optional[float] = None
    x0: Optional[float] = None
    x1: Optional[float] = None
    epsilon: float = 1e-6
    max_iterations: int = 100
    methods: List[str] = ["bisection", "newton", "secant"]
    task_type: str = "root_finding"  # root_finding, linear_system


class MethodResult(BaseModel):
    """Result for a single method."""
    method: str
    success: bool
    result: Optional[float] = None
    iterations_count: int = 0
    final_error: float = 0.0
    execution_time: float = 0.0
    iterations: List[dict] = []


class ComparisonResponse(BaseModel):
    """Response schema for algorithm comparison."""
    success: bool
    message: str
    results: List[dict] = []
    summary: Optional[dict] = None