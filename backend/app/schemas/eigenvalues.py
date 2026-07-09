"""Schemas for eigenvalues module."""
from pydantic import BaseModel
from typing import Optional, List


class EigenvaluesRequest(BaseModel):
    A: List[List[float]]
    method: str = "characteristic_polynomial"  # characteristic_polynomial, power, inverse_power, rayleigh, qr, jacobi
    epsilon: float = 1e-6
    max_iterations: int = 100
    shift: Optional[float] = None
    initial_guess: Optional[List[float]] = None


class Eigenpair(BaseModel):
    eigenvalue: Optional[float] = None
    eigenvector: Optional[List[float]] = None


class EigenvaluesResponse(BaseModel):
    success: bool
    message: str
    eigenvalues: Optional[List] = None
    eigenpairs: Optional[List[dict]] = None
    dominant_eigenvalue: Optional[float] = None
    dominant_eigenvector: Optional[List[float]] = None
    iterations_count: int = 0
    final_error: float = 0.0
    iterations: List[dict] = []
    execution_time: Optional[float] = None