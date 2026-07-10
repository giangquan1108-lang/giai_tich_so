"""Schemas for matrix decomposition module."""
from pydantic import BaseModel
from typing import Optional, List


class DecompositionRequest(BaseModel):
    A: List[List[float]]
    method: str = "lu"  # lu, cholesky, qr, svd, schur


class DecompositionResponse(BaseModel):
    success: bool
    message: str
    L: Optional[List[List[float]]] = None
    U: Optional[List[List[float]]] = None
    P: Optional[List[List[float]]] = None
    singular_values: Optional[List[float]] = None
    Vt: Optional[List[List[float]]] = None
    rank: Optional[int] = None
    condition_number: Optional[float] = None
    L_latex: Optional[str] = None
    U_latex: Optional[str] = None
    P_latex: Optional[str] = None
    Vt_latex: Optional[str] = None
    is_accurate: Optional[bool] = None
    verification: Optional[List[List[float]]] = None
    steps: Optional[List[dict]] = None
    execution_time: Optional[float] = None
