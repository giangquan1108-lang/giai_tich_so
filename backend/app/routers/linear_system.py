"""Unified linear system solver API router — direct + iterative methods."""
from fastapi import APIRouter
from app.schemas.linear_system import (
    LinearSystemRequest, LinearSystemResponse,
    MatrixPropertiesRequest, MatrixPropertiesResponse,
)
from app.algorithms import linear_system

router = APIRouter()

FORMULAS = {
    "gaussian": r"\text{Forward elimination} + \text{Back substitution}",
    "gauss_jordan": r"[A|B] \rightarrow [I|x]",
    "lu": r"A = PLU, \quad LY = PB, \quad UX = Y",
    "cholesky": r"A = LL^T, \quad LY = B, \quad L^TX = Y",
    "thomas": r"\text{Thomas Algorithm (TDMA): } O(n) \text{ cho ma trận ba đường chéo}",
    "jacobi": r"x_i^{k+1} = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j^k\right)",
    "gauss_seidel": r"x_i^{k+1} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij} x_j^{k+1} - \sum_{j>i} a_{ij} x_j^k\right)",
    "sor": r"x_i^{k+1} = (1-\omega)x_i^k + \frac{\omega}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x_j^{new}\right)",
}


@router.post("/solve", response_model=LinearSystemResponse)
async def solve_linear_system(request: LinearSystemRequest):
    """Solve linear system AX = B using any method (direct or iterative)."""
    method = request.method.lower()

    # ---- Validate dimensions ----
    m = len(request.A)
    n = len(request.A[0]) if request.A else 0
    if not request.B or not request.B[0]:
        return LinearSystemResponse(
            success=False,
            message="Ma trận B không được rỗng.",
        )
    p = len(request.B[0])
    if len(request.B) != m:
        return LinearSystemResponse(
            success=False,
            message=f"Kích thước không khớp: rows(B)={len(request.B)} != rows(A)={m}.",
        )

    if method == "gaussian":
        result = linear_system.gaussian_elimination(request.A, request.B)
    elif method == "gauss_jordan":
        result = linear_system.gauss_jordan(request.A, request.B)
    elif method == "lu":
        result = linear_system.lu_decomposition(request.A, request.B)
    elif method == "cholesky":
        result = linear_system.cholesky_decomposition(request.A, request.B)
    elif method == "thomas":
        result = linear_system.thomas_algorithm(request.A, request.B)
    elif method == "jacobi":
        result = linear_system.jacobi(
            request.A, request.B, request.initial_guess,
            request.epsilon, request.max_iterations,
        )
    elif method == "gauss_seidel":
        result = linear_system.gauss_seidel(
            request.A, request.B, request.initial_guess,
            request.epsilon, request.max_iterations,
        )
    elif method == "sor":
        omega = request.omega or 1.0
        result = linear_system.sor(
            request.A, request.B, request.initial_guess,
            request.epsilon, request.max_iterations, omega,
        )
    else:
        return LinearSystemResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: gaussian, gauss_jordan, lu, cholesky, thomas, jacobi, gauss_seidel, sor",
        )

    return LinearSystemResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        formula=FORMULAS.get(method, ""),
        execution_time=result.get("execution_time"),
        solution=result.get("solution"),
        solution_type=result.get("solution_type"),
        rank_A=result.get("rank_A"),
        rank_augmented=result.get("rank_augmented"),
        analysis=result.get("analysis"),
        free_variables=result.get("free_variables"),
        particular_solution=result.get("particular_solution"),
        basis_vectors=result.get("basis_vectors"),
        general_solution_latex=result.get("general_solution_latex"),
        steps=result.get("steps"),
        pivot_info=result.get("pivot_info"),
        iterations=result.get("iterations", []),
        iterations_count=result.get("iterations_count", 0),
        final_error=result.get("final_error", 0.0),
        convergence_data=result.get("convergence_data"),
        diagonally_dominant=result.get("diagonally_dominant"),
        effective_epsilon=result.get("effective_epsilon"),
        machine_epsilon=result.get("machine_epsilon"),
        omega=result.get("omega"),
    )


@router.post("/properties", response_model=MatrixPropertiesResponse)
async def analyze_matrix_properties(request: MatrixPropertiesRequest):
    """Analyze matrix A properties and return recommendations."""
    props = linear_system._detect_matrix_properties(request.A)
    return MatrixPropertiesResponse(**props)