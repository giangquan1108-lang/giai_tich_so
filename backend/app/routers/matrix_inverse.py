"""Matrix inverse API router."""
from fastapi import APIRouter
from app.schemas.matrix_inverse import MatrixInverseRequest, MatrixInverseResponse
from app.algorithms.matrix_inverse import (
    matrix_inverse_gauss_jordan,
    matrix_inverse_adjoint,
    matrix_inverse_lu,
    matrix_inverse_cholesky,
    pseudoinverse_svd,
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
    elif method == "lu":
        result = matrix_inverse_lu(request.A)
    elif method == "cholesky":
        result = matrix_inverse_cholesky(request.A)
    elif method == "pseudoinverse_svd":
        result = pseudoinverse_svd(request.A)
    else:
        return MatrixInverseResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: gauss_jordan, adjoint, lu, cholesky, pseudoinverse_svd",
        )

    return MatrixInverseResponse(**result)