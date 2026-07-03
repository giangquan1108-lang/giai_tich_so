"""Algorithm comparison API router."""
import time
from fastapi import APIRouter
from app.schemas.comparison import ComparisonRequest, ComparisonResponse
from app.algorithms import root_finding as rf
from app.utils.math_parser import parse_function

router = APIRouter()


@router.post("/", response_model=ComparisonResponse)
async def compare_algorithms(request: ComparisonRequest):
    """Compare multiple algorithms on the same problem."""
    task_type = request.task_type.lower()
    
    if task_type == "root_finding":
        return _compare_root_finding(request)
    
    return ComparisonResponse(
        success=False,
        message=f"Unknown task type: {task_type}",
    )


def _compare_root_finding(request: ComparisonRequest) -> ComparisonResponse:
    """Compare root finding algorithms."""
    try:
        f = parse_function(request.function)
    except Exception as e:
        return ComparisonResponse(
            success=False,
            message=f"Error parsing function: {str(e)}",
        )
    
    results = []
    
    for method in request.methods:
        method_lower = method.lower()
        start = time.time()
        
        try:
            if method_lower == "bisection" and request.a is not None and request.b is not None:
                result = rf.bisection(f, request.a, request.b, request.epsilon, request.max_iterations)
            elif method_lower == "newton" and request.x0 is not None:
                result = rf.newton_raphson(f, request.x0, request.epsilon, request.max_iterations)
            elif method_lower == "secant" and request.x0 is not None and request.x1 is not None:
                result = rf.secant(f, request.x0, request.x1, request.epsilon, request.max_iterations)
            elif method_lower == "fixed_point" and request.x0 is not None:
                result = rf.fixed_point_iteration(f, request.x0, request.epsilon, request.max_iterations)
            else:
                result = {
                    "success": False,
                    "message": f"Method {method} requires parameters not provided",
                }
            
            results.append({
                "method": method,
                "success": result.get("success", False),
                "root": result.get("root"),
                "iterations_count": result.get("iterations_count", 0),
                "final_error": result.get("final_error", 0),
                "execution_time": result.get("execution_time", 0),
                "iterations": result.get("iterations", []),
            })
        except Exception as e:
            results.append({
                "method": method,
                "success": False,
                "message": str(e),
            })
    
    # Summary
    successful = [r for r in results if r.get("success")]
    summary = {
        "total_methods": len(results),
        "successful_methods": len(successful),
        "fastest": min(successful, key=lambda x: x.get("execution_time", float("inf")))["method"] if successful else None,
        "fewest_iterations": min(successful, key=lambda x: x.get("iterations_count", float("inf")))["method"] if successful else None,
    }
    
    return ComparisonResponse(
        success=True,
        message=f"Compared {len(results)} methods",
        results=results,
        summary=summary,
    )