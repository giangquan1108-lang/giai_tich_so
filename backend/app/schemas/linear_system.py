"""Schemas for unified linear system solver module (direct + iterative)."""
from pydantic import BaseModel
from typing import Optional, List


class LinearSystemRequest(BaseModel):
    """Request schema for solving AX=B with any method.

    A: m×n coefficient matrix
    B: m×p right-hand side matrix (general case). For vector b (m×1), pass [[b1], [b2], ...].
    """
    A: List[List[float]]
    B: List[List[float]]  # m×p matrix
    method: str = "gaussian"  # gaussian, gauss_jordan, lu, cholesky, thomas, jacobi, gauss_seidel, sor
    initial_guess: Optional[List[float]] = None
    epsilon: float = 1e-6
    max_iterations: int = 100
    omega: Optional[float] = 1.0


class LinearSystemResponse(BaseModel):
    """Response schema for linear system solving."""
    success: bool
    message: str
    formula: str = ""
    execution_time: Optional[float] = None

    # Generic solution — X is n×p for AX=B with A(m×n), B(m×p)
    solution: Optional[List[List[float]]] = None

    # Solution classification (direct methods)
    solution_type: Optional[str] = None  # "unique", "inconsistent", "infinite"
    rank_A: Optional[int] = None
    rank_augmented: Optional[int] = None
    analysis: Optional[dict] = None

    # For infinite solutions
    free_variables: Optional[List[str]] = None
    particular_solution: Optional[List[List[float]]] = None  # n×p — one particular solution per B column
    basis_vectors: Optional[List[List[float]]] = None
    general_solution_latex: Optional[str] = None

    # Direct method steps
    steps: Optional[List[dict]] = None
    pivot_info: Optional[List[dict]] = None

    # Iterative method fields
    iterations: List[dict] = []
    iterations_count: int = 0
    final_error: float = 0.0
    convergence_data: Optional[dict] = None

    # Extra metadata
    diagonally_dominant: Optional[bool] = None
    effective_epsilon: Optional[float] = None
    machine_epsilon: Optional[float] = None
    omega: Optional[float] = None

    # Matrix property analysis
    matrix_properties: Optional[dict] = None


class MatrixPropertiesRequest(BaseModel):
    """Request schema for matrix property analysis."""
    A: List[List[float]]


class MatrixPropertiesResponse(BaseModel):
    """Response for matrix property analysis."""
    is_square: bool
    is_symmetric: bool
    is_positive_definite: bool
    is_tridiagonal: bool
    is_diagonally_dominant_strict: bool
    is_diagonally_dominant_weak: bool
    recommendations: List[str] = []


class MatrixInverseRequest(BaseModel):
    """Request schema for matrix inverse computation."""
    A: List[List[float]]
    method: str = "gauss_jordan"  # gauss_jordan, adjoint, lu, cholesky


class MatrixInverseResponse(BaseModel):
    """Response schema for matrix inverse."""
    success: bool
    message: str
    determinant: Optional[float] = None
    rank: Optional[int] = None
    inverse: Optional[List[List[float]]] = None
    inverse_latex: Optional[str] = None
    verification: Optional[List[List[float]]] = None
    is_accurate: Optional[bool] = None
    steps: Optional[List[dict]] = None
    execution_time: Optional[float] = None
