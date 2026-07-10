"""Nonlinear system solving algorithms: Newton Multivariable, Fixed Point Multivariable."""
import sys
import numpy as np
from typing import List
import time
from app.utils.math_parser import (
    parse_multivariable_function,
    compute_numerical_jacobian,
)

MACHINE_EPSILON = sys.float_info.epsilon
MIN_EPSILON = 1e-15

def _effective_epsilon(epsilon: float) -> float:
    return max(epsilon, MIN_EPSILON)

def newton_multivariable(
    functions: List[str],
    variables: List[str],
    initial_guess: List[float],
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    jacobian_properties = []  # track det(J) and cond(J) across steps
    n = len(functions)
    x = list(initial_guess)

    parsed_funcs = [parse_multivariable_function(f, variables) for f in functions]

    start_time = time.time()
    error = 0.0
    stopping_criterion = ""

    for k in range(max_iterations):
        # Divergence check: if values explode, bail out
        if any(abs(xi) > 1e10 for xi in x):
            return {
                "success": False,
                "message": "Phương pháp Newton phân kỳ — giá trị vượt quá giới hạn (|x| > 10^10).",
                "stopping_criterion": "divergence_explosion",
                "iterations": iterations,
                "jacobian_properties": jacobian_properties,
                "execution_time": round(time.time() - start_time, 6),
                "iterations_count": k,
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
            }
        try:
            fx = [f(*x) for f in parsed_funcs]
        except (OverflowError, ValueError, FloatingPointError, ZeroDivisionError):
            return {
                "success": False,
                "message": "Phương pháp Newton phân kỳ — tràn số hoặc chia cho 0 khi tính F(x).",
                "stopping_criterion": "math_error",
                "iterations": iterations,
                "jacobian_properties": jacobian_properties,
                "execution_time": round(time.time() - start_time, 6),
                "iterations_count": k,
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
            }
        
        # Check for NaN/Inf in function values
        if any(not np.isfinite(fi) for fi in fx):
            return {
                "success": False,
                "message": "Phương pháp Newton phân kỳ — F(x) chứa NaN hoặc vô cực.",
                "stopping_criterion": "nan_inf",
                "iterations": iterations,
                "jacobian_properties": jacobian_properties,
                "execution_time": round(time.time() - start_time, 6),
                "iterations_count": k,
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
            }
        
        jacobian = compute_numerical_jacobian(functions, variables, x)

        J = np.array(jacobian, dtype=float)
        F = np.array(fx, dtype=float)

        # Compute Jacobian properties
        norm_F_val = float(np.linalg.norm(F))
        try:
            det_J = float(np.linalg.det(J))
        except np.linalg.LinAlgError:
            det_J = 0.0
        try:
            cond_J = float(np.linalg.cond(J))
        except np.linalg.LinAlgError:
            cond_J = float('inf')

        jacobian_properties.append({
            "k": k + 1,
            "det": round(det_J, 8) if abs(det_J) < 1e8 else f"{det_J:.4e}",
            "cond": round(cond_J, 4) if cond_J < 1e8 else f"{cond_J:.4e}",
        })

        # Reasoning logic
        if abs(det_J) < 1e-12:
            reasoning = f"Ma trận Jacobi gần suy biến (det(J) = {det_J:.2e} ≈ 0). Phương pháp Newton có thể thất bại."
        elif cond_J > 1e10:
            reasoning = f"Ma trận Jacobi có điều kiện xấu (cond(J) = {cond_J:.2e}). Hội tụ có thể chậm hoặc không ổn định."
        else:
            reasoning = f"Jacobi khả nghịch (det = {det_J:.4f}). Giải hệ J·dx = -F tìm bước đi."

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return {
                "success": False,
                "message": f"Ma trận Jacobi suy biến tại x = {[round(xi, 7) for xi in x]}. det(J) = {det_J:.2e}. Phương pháp thất bại.",
                "stopping_criterion": "singular_jacobian",
                "iterations": iterations,
                "jacobian_properties": jacobian_properties,
                "execution_time": round(time.time() - start_time, 6),
                "iterations_count": k,
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
            }

        x_new = [x[i] + dx[i] for i in range(n)]
        dx_norm = float(np.linalg.norm(dx))
        error = dx_norm

        # Detailed reasoning for this iteration
        detail_reasoning = (
            f"||F(x)|| = {norm_F_val:.4e}, ||dx|| = {dx_norm:.4e}. "
        )
        if dx_norm < eps_eff:
            detail_reasoning += f"||dx|| < eps = {eps_eff:.1e} → hội tụ."
        elif norm_F_val < eps_eff * n:
            detail_reasoning += f"||F|| < n·eps = {eps_eff*n:.1e} → hội tụ."
        else:
            detail_reasoning += f"Chưa đạt ngưỡng hội tụ → tiếp tục lặp."

        iterations.append({
            "k": k + 1,
            "x": [round(xi, 10) for xi in x],
            "F_x": [round(fi, 10) for fi in fx],
            "jacobian": [[round(jacobian[i][j], 7) for j in range(n)] for i in range(n)],
            "dx": [round(dxi, 10) for dxi in dx],
            "x_new": [round(xi, 10) for xi in x_new],
            "error": round(dx_norm, 10),
            "norm_F": round(norm_F_val, 10),
            "det_jacobian": round(det_J, 8) if abs(det_J) < 1e8 else f"{det_J:.4e}",
            "reasoning": detail_reasoning,
        })

        if dx_norm < eps_eff:
            stopping_criterion = f"||dx|| = {dx_norm:.2e} < eps = {eps_eff:.1e}"
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Hội tụ sau {k+1} vòng lặp: {stopping_criterion}.",
                "solution": [round(xi, 10) for xi in x_new],
                "iterations_count": k + 1,
                "final_error": round(dx_norm, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 7),
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
                "stopping_criterion": stopping_criterion,
                "jacobian_properties": jacobian_properties,
            }
        if norm_F_val < eps_eff * n:
            stopping_criterion = f"||F(x)|| = {norm_F_val:.2e} < n·eps = {eps_eff*n:.1e}"
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Hội tụ sau {k+1} vòng lặp: {stopping_criterion}.",
                "solution": [round(xi, 10) for xi in x_new],
                "iterations_count": k + 1,
                "final_error": round(dx_norm, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 7),
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
                "stopping_criterion": stopping_criterion,
                "jacobian_properties": jacobian_properties,
            }

        x = x_new

    execution_time = time.time() - start_time
    try:
        final_fx = [f(*x) for f in parsed_funcs]
        final_norm = sum(abs(fi) for fi in final_fx)
    except Exception:
        final_norm = float('inf')

    if error < eps_eff * 1000 or final_norm < eps_eff * 1000 * n:
        stopping_criterion = f"||dx|| ≈ {error:.2e} < 1000·eps (hội tụ yếu)"
        return {
            "success": True,
            "message": f"Hội tụ gần đúng sau {max_iterations} vòng lặp (residual ≈ 0).",
            "solution": [round(xi, 10) for xi in x],
            "iterations_count": max_iterations,
            "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 7),
            "effective_epsilon": eps_eff,
            "machine_epsilon": MACHINE_EPSILON,
            "stopping_criterion": stopping_criterion,
            "jacobian_properties": jacobian_properties,
        }

    stopping_criterion = f"Vượt max_iterations = {max_iterations}, ||dx|| = {error:.2e}"
    return {
        "success": False,
        "message": f"Không hội tụ sau {max_iterations} vòng lặp. ||dx|| = {error:.2e} > eps = {eps_eff:.1e}. Thử tăng max_iterations hoặc đổi x₀.",
        "solution": [round(xi, 10) for xi in x],
        "iterations_count": max_iterations,
        "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 7),
        "effective_epsilon": eps_eff,
        "machine_epsilon": MACHINE_EPSILON,
        "stopping_criterion": stopping_criterion,
        "jacobian_properties": jacobian_properties,
    }


def fixed_point_multivariable(
    functions: List[str],
    variables: List[str],
    initial_guess: List[float],
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    n = len(functions)
    x = list(initial_guess)

    parsed_funcs = [parse_multivariable_function(f, variables) for f in functions]

    # Compute Jacobian of G(x) at initial guess to check contraction condition
    contraction_warning = None
    try:
        Jg = compute_numerical_jacobian(functions, variables, x)
        Jg_mat = np.array(Jg, dtype=float)
        singular_values = np.linalg.svd(Jg_mat, compute_uv=False)
        spectral_norm = float(max(singular_values)) if len(singular_values) > 0 else 0
        frobenius_norm = float(np.linalg.norm(Jg_mat, 'fro'))
        if spectral_norm > 1.0:
            contraction_warning = (
                f"Ánh xạ điểm bất động hiện tại có thể không hội tụ. "
                f"||JG||₂ = {spectral_norm:.4f} > 1 (không thỏa điều kiện co). "
                f"Frobenius norm = {frobenius_norm:.4f}"
            )
        elif spectral_norm > 0.9:
            contraction_warning = (
                f"Ánh xạ điểm bất động hội tụ chậm. "
                f"||JG||₂ = {spectral_norm:.4f} (gần 1). "
                f"Frobenius norm = {frobenius_norm:.4f}"
            )
    except Exception:
        pass

    start_time = time.time()
    error = 0.0

    for k in range(max_iterations):
        # Divergence check: if values explode, bail out
        if any(abs(xi) > 1e10 for xi in x):
            return {
                "success": False,
                "message": "Phương pháp điểm bất động phân kỳ — giá trị vượt quá giới hạn (|x| > 10^10).",
                "iterations": iterations,
                "contraction_warning": contraction_warning,
            }
        try:
            x_new = [f(*x) for f in parsed_funcs]
        except (OverflowError, ValueError, FloatingPointError, ZeroDivisionError):
            return {
                "success": False,
                "message": "Phương pháp điểm bất động phân kỳ — tràn số hoặc chia cho 0 khi tính G(x).",
                "iterations": iterations,
                "contraction_warning": contraction_warning,
            }
        
        # Check for NaN/Inf in results
        if any(not np.isfinite(xi) for xi in x_new):
            return {
                "success": False,
                "message": "Phương pháp điểm bất động phân kỳ — kết quả chứa NaN hoặc vô cực.",
                "iterations": iterations,
                "contraction_warning": contraction_warning,
            }
        
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))

        iterations.append({
            "k": k + 1,
            "x": [round(xi, 10) for xi in x],
            "G_x": [round(xi, 10) for xi in x_new],
            "error": round(error, 10),
        })

        if error < eps_eff:
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Fixed point found after {k+1} iterations",
                "solution": [round(xi, 10) for xi in x_new],
                "iterations_count": k + 1,
                "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 7),
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
                "contraction_warning": contraction_warning,
            }

        x = x_new

    execution_time = time.time() - start_time

    if error < eps_eff * 1000:
        return {
            "success": True,
            "message": f"Fixed point found after {max_iterations} iterations (Δ ≈ 0)",
            "solution": [round(xi, 10) for xi in x],
            "iterations_count": max_iterations,
            "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 7),
            "effective_epsilon": eps_eff,
            "machine_epsilon": MACHINE_EPSILON,
            "contraction_warning": contraction_warning,
        }

    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "solution": [round(xi, 10) for xi in x],
        "iterations_count": max_iterations,
        "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 7),
        "effective_epsilon": eps_eff,
        "machine_epsilon": MACHINE_EPSILON,
        "contraction_warning": contraction_warning,
    }