"""Schemas for nonlinear system module."""
from pydantic import BaseModel
from typing import Optional, List


class NonlinearSystemRequest(BaseModel):
    """Request schema for nonlinear system solving."""
    functions: List[str]
    variables: List[str]
    initial_guess: List[float]
    epsilon: float = 1e-6
    max_iterations: int = 100
    method: str = "newton"  # newton, fixed_point


class NonlinearSystemResponse(BaseModel):
    """Response schema for nonlinear system solving."""
    success: bool
    message: str
    solution: Optional[List[float]] = None
    jacobian: Optional[List[List[float]]] = None
    iterations_count: int = 0
    final_error: float = 0.0
    iterations: List[dict] = []
    formula: str = ""
    contraction_warning: Optional[str] = None
    stopping_criterion: Optional[str] = None
    jacobian_properties: Optional[List[dict]] = None
