"""Nonlinear system API router."""
from fastapi import APIRouter
from app.schemas.nonlinear_system import NonlinearSystemRequest, NonlinearSystemResponse
from app.algorithms import nonlinear_system
from app.utils.math_parser import latex_to_python

router = APIRouter()


@router.post("/", response_model=NonlinearSystemResponse)
async def solve_nonlinear_system(request: NonlinearSystemRequest):
    """Solve a system of nonlinear equations."""
    method = request.method.lower()
    
    # Convert LaTeX functions to Python expressions
    py_functions = []
    for func in request.functions:
        try:
            py_func = latex_to_python(func)
            py_functions.append(py_func)
        except Exception:
            py_functions.append(func)  # fallback: keep original
    
    if method == "newton":
        result = nonlinear_system.newton_multivariable(
            py_functions,
            request.variables,
            request.initial_guess,
            request.epsilon,
            request.max_iterations,
        )
    elif method == "fixed_point":
        result = nonlinear_system.fixed_point_multivariable(
            py_functions,
            request.variables,
            request.initial_guess,
            request.epsilon,
            request.max_iterations,
        )
    else:
        return NonlinearSystemResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: newton, fixed_point",
        )
    
    return NonlinearSystemResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        solution=result.get("solution"),
        jacobian=result.get("jacobian"),
        iterations_count=result.get("iterations_count", 0),
        final_error=result.get("final_error", 0),
        iterations=result.get("iterations", []),
        formula=r"X_{k+1} = X_k - J^{-1}(X_k) \cdot F(X_k)" if method == "newton" else r"X_{k+1} = G(X_k)",
        contraction_warning=result.get("contraction_warning"),
        stopping_criterion=result.get("stopping_criterion"),
        jacobian_properties=result.get("jacobian_properties"),
    )
