"""Eigenvalues API router."""
from fastapi import APIRouter
from app.schemas.eigenvalues import EigenvaluesRequest, EigenvaluesResponse
from app.algorithms.eigenvalues import (
    characteristic_polynomial,
    power_iteration,
    inverse_power_iteration,
    rayleigh_quotient_iteration,
    qr_algorithm,
    jacobi_method,
)

router = APIRouter()


@router.post("/", response_model=EigenvaluesResponse)
async def compute_eigenvalues(request: EigenvaluesRequest):
    method = request.method.lower()
    if method == "characteristic_polynomial":
        result = characteristic_polynomial(request.A)
    elif method == "power":
        result = power_iteration(request.A, request.epsilon, request.max_iterations, request.initial_guess)
    elif method == "inverse_power":
        result = inverse_power_iteration(request.A, request.shift or 0.0, request.epsilon, request.max_iterations, request.initial_guess)
    elif method == "rayleigh":
        result = rayleigh_quotient_iteration(request.A, request.epsilon, request.max_iterations, request.initial_guess)
    elif method == "qr":
        result = qr_algorithm(request.A, request.max_iterations)
    elif method == "jacobi":
        result = jacobi_method(request.A, request.epsilon, request.max_iterations)
    else:
        return EigenvaluesResponse(success=False, message=f"Unknown method: {method}")
    return EigenvaluesResponse(**result)