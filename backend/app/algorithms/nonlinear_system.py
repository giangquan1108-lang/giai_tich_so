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
    n = len(functions)
    x = list(initial_guess)

    parsed_funcs = [parse_multivariable_function(f, variables) for f in functions]

    start_time = time.time()
    error = 0.0

    for k in range(max_iterations):
        # Divergence check: if values explode, bail out
        if any(abs(xi) > 1e10 for xi in x):
            return {
                "success": False,
                "message": "Phương pháp phân kỳ — giá trị vượt quá giới hạn (|x| > 10^10).",
                "iterations": iterations,
            }
        try:
            fx = [f(*x) for f in parsed_funcs]
        except (OverflowError, ValueError, FloatingPointError):
            return {
                "success": False,
                "message": "Phương pháp phân kỳ — tràn số khi tính f(x).",
                "iterations": iterations,
            }
        
        # Check for NaN/Inf in function values
        if any(not np.isfinite(fi) for fi in fx):
            return {
                "success": False,
                "message": "Phương pháp phân kỳ — f(x) chứa NaN hoặc vô cực.",
                "iterations": iterations,
            }
        
        jacobian = compute_numerical_jacobian(functions, variables, x)

        J = np.array(jacobian, dtype=float)
        F = np.array(fx, dtype=float)

        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return {
                "success": False,
                "message": "Singular Jacobian matrix. Method fails.",
                "iterations": iterations,
            }

        x_new = [x[i] + dx[i] for i in range(n)]
        error = float(np.linalg.norm(dx))

        iterations.append({
            "k": k + 1,
            "x": [round(xi, 10) for xi in x],
            "F_x": [round(fi, 10) for fi in fx],
            "jacobian": [[round(jacobian[i][j], 6) for j in range(n)] for i in range(n)],
            "dx": [round(dxi, 10) for dxi in dx],
            "x_new": [round(xi, 10) for xi in x_new],
            "error": round(error, 10),
        })

        if error < eps_eff or sum(abs(fi) for fi in fx) < eps_eff * n:
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Solution found after {k+1} iterations",
                "solution": [round(xi, 10) for xi in x_new],
                "iterations_count": k + 1,
                "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 6),
                "effective_epsilon": eps_eff,
                "machine_epsilon": MACHINE_EPSILON,
            }

        x = x_new

    execution_time = time.time() - start_time
    final_norm = sum(abs(f(*x)) for f in parsed_funcs)

    if error < eps_eff * 1000 or final_norm < eps_eff * 1000 * n:
        return {
            "success": True,
            "message": f"Solution found after {max_iterations} iterations (residual ≈ 0)",
            "solution": [round(xi, 10) for xi in x],
            "iterations_count": max_iterations,
            "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 6),
            "effective_epsilon": eps_eff,
            "machine_epsilon": MACHINE_EPSILON,
        }

    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "solution": [round(xi, 10) for xi in x],
        "iterations_count": max_iterations,
        "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 6),
        "effective_epsilon": eps_eff,
        "machine_epsilon": MACHINE_EPSILON,
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
        except (OverflowError, ValueError, FloatingPointError):
            return {
                "success": False,
                "message": "Phương pháp điểm bất động phân kỳ — tràn số khi tính G(x).",
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
                "execution_time": round(execution_time, 6),
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
            "execution_time": round(execution_time, 6),
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
        "execution_time": round(execution_time, 6),
        "effective_epsilon": eps_eff,
        "machine_epsilon": MACHINE_EPSILON,
        "contraction_warning": contraction_warning,
    }