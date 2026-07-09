"""Schemas for matrix inverse module."""
from pydantic import BaseModel
from typing import Optional, List


class MatrixInverseRequest(BaseModel):
    A: List[List[float]]
    method: str = "gauss_jordan"  # gauss_jordan, adjoint, lu, cholesky, pseudoinverse_svd


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