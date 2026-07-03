"""Numerical integration API router."""
from fastapi import APIRouter
from app.schemas.integration import IntegrationRequest, IntegrationResponse
from app.algorithms import integration as integ_algo
from app.utils.math_parser import parse_function, latex_to_python

router = APIRouter()


@router.post("/", response_model=IntegrationResponse)
async def solve_integration(request: IntegrationRequest):
    """Perform numerical integration on f(x) from a to b."""
    try:
        func_str = request.function
        try:
            py_func = latex_to_python(func_str)
            f = parse_function(py_func)
        except Exception:
            f = parse_function(func_str)
    except Exception as e:
        return IntegrationResponse(
            success=False,
            message=f"Error parsing function: {str(e)}",
        )
    
    method = request.method.lower()
    
    if method == "trapezoidal":
        result = integ_algo.trapezoidal(f, request.a, request.b, request.n)
    elif method == "simpson_13":
        result = integ_algo.simpson_one_third(f, request.a, request.b, request.n)
    elif method == "simpson_38":
        result = integ_algo.simpson_three_eighth(f, request.a, request.b, request.n)
    elif method == "romberg":
        result = integ_algo.romberg(f, request.a, request.b, min(request.n, 20))
    else:
        return IntegrationResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: trapezoidal, simpson_13, simpson_38, romberg",
        )
    
    relative_error = None
    if result.get("exact_value") and result.get("result"):
        exact = result["exact_value"]
        if abs(exact) > 1e-15:
            relative_error = round(abs(result["result"] - exact) / abs(exact) * 100, 6)
    
    return IntegrationResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        result=result.get("result"),
        exact_value=result.get("exact_value"),
        error=result.get("error"),
        relative_error=relative_error,
        iterations=result.get("iterations", []),
        formula=result.get("formula", ""),
        plot_data=result.get("plot_data"),
    )