"""Root finding API router."""
from fastapi import APIRouter
from app.schemas.root_finding import RootFindingRequest, RootFindingResponse
from app.algorithms import root_finding
from app.utils.math_parser import parse_function, latex_to_python

router = APIRouter()


@router.post("/", response_model=RootFindingResponse)
async def solve_root_finding(request: RootFindingRequest):
    """Solve for roots of f(x) = 0 using various methods."""
    try:
        func_str = request.function
        try:
            py_func = latex_to_python(func_str)
            f = parse_function(py_func)
        except Exception:
            f = parse_function(func_str)
    except Exception as e:
        return RootFindingResponse(
            success=False,
            message=f"Error parsing function: {str(e)}",
        )
    
    method = request.method.lower()
    
    if method == "bisection":
        if request.a is None or request.b is None:
            return RootFindingResponse(
                success=False,
                message="Bisection method requires parameters a and b.",
            )
        result = root_finding.bisection(
            f, request.a, request.b, request.epsilon, request.max_iterations
        )
    
    elif method == "newton":
        if request.x0 is None:
            return RootFindingResponse(
                success=False,
                message="Newton-Raphson method requires initial guess x0.",
            )
        result = root_finding.newton_raphson(
            f, request.x0, request.epsilon, request.max_iterations
        )
    
    elif method == "secant":
        if request.x0 is None or request.x1 is None:
            return RootFindingResponse(
                success=False,
                message="Secant method requires two initial guesses x0 and x1.",
            )
        result = root_finding.secant(
            f, request.x0, request.x1, request.epsilon, request.max_iterations
        )
    
    elif method == "fixed_point":
        if request.x0 is None:
            return RootFindingResponse(
                success=False,
                message="Fixed point iteration requires initial guess x0.",
            )
        result = root_finding.fixed_point_iteration(
            f, request.x0, request.epsilon, request.max_iterations
        )
    
    else:
        return RootFindingResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: bisection, newton, secant, fixed_point",
        )
    
    eff_eps = result.get("effective_epsilon", request.epsilon)
    mach_eps = result.get("machine_epsilon", 0)
    message = result.get("message", "")
    if eff_eps and eff_eps != request.epsilon:
        message += f" (req ε={request.epsilon}, eff ε={eff_eps:.1e}, mach ε={mach_eps:.1e})"
    
    return RootFindingResponse(
        success=result.get("success", False),
        message=message,
        root=result.get("root"),
        f_root=result.get("f_root"),
        iterations_count=result.get("iterations_count", 0),
        final_error=result.get("final_error", 0),
        iterations=result.get("iterations", []),
        formula=_get_formula(method),
        convergence_data=result.get("convergence_data"),
        contraction_info=result.get("contraction_info"),
    )


def _get_formula(method: str) -> str:
    """Get the LaTeX formula for the given method."""
    formulas = {
        "bisection": r"c = \frac{a + b}{2}",
        "newton": r"x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}",
        "secant": r"x_{k+1} = x_k - f(x_k) \cdot \frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}",
        "fixed_point": r"x_{k+1} = g(x_k)",
    }
    return formulas.get(method, "")