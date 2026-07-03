"""Interpolation API router."""
from fastapi import APIRouter
from app.schemas.interpolation import InterpolationRequest, InterpolationResponse
from app.algorithms import interpolation as interp_algo

router = APIRouter()


@router.post("/", response_model=InterpolationResponse)
async def solve_interpolation(request: InterpolationRequest):
    """Perform interpolation on given data points."""
    method = request.method.lower()
    
    if len(request.x_points) != len(request.y_points):
        return InterpolationResponse(
            success=False,
            message="x_points and y_points must have the same length.",
        )
    
    if len(request.x_points) < 2:
        return InterpolationResponse(
            success=False,
            message="At least 2 data points are required.",
        )
    
    if method == "lagrange":
        result = interp_algo.lagrange_interpolation(
            request.x_points, request.y_points, request.x_value
        )
    elif method == "newton_forward":
        result = interp_algo.newton_forward_interpolation(
            request.x_points, request.y_points, request.x_value
        )
    elif method == "newton_backward":
        result = interp_algo.newton_backward_interpolation(
            request.x_points, request.y_points, request.x_value
        )
    elif method == "divided_differences":
        result = interp_algo.divided_differences(
            request.x_points, request.y_points, request.x_value
        )
    else:
        return InterpolationResponse(
            success=False,
            message=f"Unknown method: {method}. Choose from: lagrange, newton_forward, newton_backward, divided_differences",
        )
    
    return InterpolationResponse(
        success=result.get("success", False),
        message=result.get("message", ""),
        interpolated_value=result.get("interpolated_value"),
        polynomial=result.get("polynomial"),
        divided_diff_table=result.get("divided_diff_table"),
        iterations=result.get("iterations", []),
        formula=result.get("formula", ""),
    )