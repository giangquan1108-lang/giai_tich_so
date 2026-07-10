"""Matrix decomposition API router."""
from fastapi import APIRouter
from app.schemas.matrix_decomposition import DecompositionRequest, DecompositionResponse
from app.algorithms.matrix_decomposition import (
    lu_decomposition, cholesky_decomposition,
    svd_decomposition,
)

router = APIRouter()


@router.post("/", response_model=DecompositionResponse)
async def decompose_matrix(request: DecompositionRequest):
    method = request.method.lower()
    if method == "lu":
        result = lu_decomposition(request.A)
    elif method == "cholesky":
        result = cholesky_decomposition(request.A)
    elif method == "svd":
        result = svd_decomposition(request.A)
    else:
        return DecompositionResponse(success=False, message=f"Unknown method: {method}")
    return DecompositionResponse(**result)