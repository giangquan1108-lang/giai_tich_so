"""Matrix inverse API router."""
from fastapi import APIRouter
from app.schemas.matrix_inverse import MatrixInverseRequest, MatrixInverseResponse
from app.algorithms.matrix_inverse import (
    matrix_inverse_gauss_jordan,
    matrix_inverse_adjoint,
    matrix_inverse_cholesky,
    matrix_inverse_bordering,
    matrix_inverse_jacobi,
    matrix_inverse_gauss_seidel,
    matrix_inverse_newton,
)

router = APIRouter()


@router.post("/", response_model=MatrixInverseResponse)
async def compute_inverse(request: MatrixInverseRequest):
    """Compute matrix inverse using selected method."""
    method = request.method.lower()

    if method == "gauss_jordan":
        result = matrix_inverse_gauss_jordan(request.A)
    elif method == "adjoint":
        result = matrix_inverse_adjoint(request.A)
    elif method == "cholesky":
        result = matrix_inverse_cholesky(request.A)
    elif method == "bordering":
        result = matrix_inverse_bordering(request.A)
    elif method == "jacobi":
        result = matrix_inverse_jacobi(
            request.A,
            epsilon=request.epsilon if request.epsilon is not None else 1e-7,
            max_iterations=request.max_iterations if request.max_iterations is not None else 1000,
            initial_guess=request.initial_guess,
        )
    elif method == "gauss_seidel":
        result = matrix_inverse_gauss_seidel(
            request.A,
            epsilon=request.epsilon if request.epsilon is not None else 1e-7,
            max_iterations=request.max_iterations if request.max_iterations is not None else 1000,
            initial_guess=request.initial_guess,
        )
    elif method == "newton":
        result = matrix_inverse_newton(
            request.A,
            epsilon=request.epsilon if request.epsilon is not None else 1e-7,
            max_iterations=request.max_iterations if request.max_iterations is not None else 100,
            initial_guess=request.initial_guess,
        )
    else:
        return MatrixInverseResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: gauss_jordan, adjoint, cholesky, bordering, jacobi, gauss_seidel, newton",
        )

    return MatrixInverseResponse(**result)
