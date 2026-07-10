"""Schemas for matrix inverse module."""
from pydantic import BaseModel
from typing import Optional, List


class MatrixInverseRequest(BaseModel):
    A: List[List[float]]
    method: str = "gauss_jordan"  # gauss_jordan, adjoint, cholesky, bordering, jacobi, gauss_seidel, newton
    epsilon: Optional[float] = None
    max_iterations: Optional[int] = None
    initial_guess: Optional[List[List[float]]] = None


class MatrixInverseResponse(BaseModel):
    success: bool
    message: str
    determinant: Optional[float] = None
    rank: Optional[int] = None
    singular_values: Optional[List[float]] = None
    condition_number: Optional[float] = None
    inverse: Optional[List[List[float]]] = None
    inverse_latex: Optional[str] = None
    verification: Optional[List[List[float]]] = None
    is_accurate: Optional[bool] = None
    steps: Optional[List[dict]] = None
    execution_time: Optional[float] = None
    iterations_count: Optional[int] = None
    final_error: Optional[float] = None
    spectral_radius: Optional[float] = None
    converges: Optional[bool] = None
    iterations: Optional[List[dict]] = None
