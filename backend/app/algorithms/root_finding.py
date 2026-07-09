"""Root finding algorithms: Bisection, Newton-Raphson, Secant, Fixed Point Iteration."""
import sys
import numpy as np
from typing import Callable, Optional, Tuple, List
import time

# Machine epsilon from sys.float_info
MACHINE_EPSILON = sys.float_info.epsilon  # ~2.22e-16
MIN_EPSILON = 1e-15  # Practical minimum tolerance


def _effective_epsilon(epsilon: float) -> float:
    """Return epsilon clamped to practical minimum."""
    return max(epsilon, MIN_EPSILON)


def _is_converged(fx: float, error: float, epsilon: float) -> bool:
    """Check convergence: BOTH |f(x)| AND step size error must be within tolerance.

    Requires BOTH conditions to avoid false convergence:
    - |f(x)| < epsilon (function value near zero)
    - error < epsilon (step size small)
    
    Falls back to machine epsilon if BOTH are at machine precision.
    """
    eff_eps = max(epsilon, MACHINE_EPSILON * 10)
    if abs(fx) < eff_eps and error < eff_eps:
        return True
    # Machine precision fallback: still converged if both near machine epsilon
    if abs(fx) < MACHINE_EPSILON * 10000 and error < MACHINE_EPSILON * 10000:
        return True
    return False


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    """Bisection Method for finding roots."""
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        # Check if a or b is already a root
        if abs(fa) < eps_eff * 100:
            return {
                "success": True, "message": f"Root found at x = {a:.10f} (boundary)",
                "root": round(a, 10), "f_root": round(fa, 10),
                "iterations_count": 0, "final_error": abs(fa),
                "iterations": [], "execution_time": 0,
                "convergence_data": _get_convergence_data([]),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            }
        if abs(fb) < eps_eff * 100:
            return {
                "success": True, "message": f"Root found at x = {b:.10f} (boundary)",
                "root": round(b, 10), "f_root": round(fb, 10),
                "iterations_count": 0, "final_error": abs(fb),
                "iterations": [], "execution_time": 0,
                "convergence_data": _get_convergence_data([]),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            }
        return {
            "success": False,
            "message": "f(a) and f(b) must have opposite signs",
            "iterations": iterations,
        }
    
    start_time = time.time()
    
    for k in range(max_iterations):
        c = (a + b) / 2.0
        fc = f(c)
        error = (b - a) / 2.0
        
        iterations.append({
            "k": k + 1, "a": round(a, 10), "b": round(b, 10),
            "x_k": round(c, 10), "f_x_k": round(fc, 10),
            "error": round(error, 10),
        })
        
        if _is_converged(fc, error, eps_eff):
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Root found at x = {c:.10f} after {k+1} iterations",
                "root": round(c, 10), "f_root": round(fc, 10),
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 6),
                "convergence_data": _get_convergence_data(iterations),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            }
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    # Final check after max iterations
    execution_time = time.time() - start_time
    fc = f(c)
    final_error = (b - a) / 2.0
    
    # Check if converged despite max iterations
    if abs(fc) < eps_eff * 1000 or final_error < eps_eff * 1000:
        return {
            "success": True,
            "message": f"Root found at x = {c:.10f} after {max_iterations} iterations (reached max, f(x) ≈ 0)",
            "root": round(c, 10), "f_root": round(fc, 10),
            "iterations_count": max_iterations, "final_error": round(final_error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
        }
    
    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "root": round(c, 10), "f_root": round(fc, 10),
        "iterations_count": max_iterations, "final_error": round(final_error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 6),
        "convergence_data": _get_convergence_data(iterations),
        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
    }


def newton_raphson(
    f: Callable[[float], float],
    x0: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
    f_prime: Optional[Callable[[float], float]] = None,
) -> dict:
    """Newton-Raphson Method for finding roots."""
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    
    if f_prime is None:
        def f_prime(x):
            # Use larger h to avoid numerical noise, especially for high-degree polynomials
            h = max(1e-6, 1e-4 * abs(x))
            return (f(x + h) - f(x - h)) / (2 * h)
    
    start_time = time.time()
    error = 0.0
    
    for k in range(max_iterations):
        fx = f(x0)
        fpx = f_prime(x0)
        
        # Check for NaN/Inf
        if np.isnan(fx) or np.isinf(fx) or np.isnan(fpx) or np.isinf(fpx):
            return {
                "success": False,
                "message": f"NaN/Inf encountered at x = {x0:.6f}. Function may be undefined here.",
                "iterations": iterations,
            }
        
        # Small derivative threshold (increased from 1e-15)
        if abs(fpx) < 1e-10:
            return {
                "success": False,
                "message": f"Derivative is near zero (|f'| = {abs(fpx):.2e}) at x = {x0:.6f}. Newton-Raphson method fails.",
                "iterations": iterations,
                "root": round(x0, 10),
                "f_root": round(fx, 10),
            }
        
        dx = fx / fpx
        x1 = x0 - dx
        error = abs(dx)
        
        # Step size limit: if Newton step is unreasonably large relative to current x, abort
        if error > max(abs(x0) * 100, 1e6):
            return {
                "success": False,
                "message": f"Newton step too large (|dx| = {error:.2e}) at x = {x0:.6f}. Method diverging.",
                "iterations": iterations,
                "root": round(x0, 10),
                "f_root": round(fx, 10),
            }
        
        iterations.append({
            "k": k + 1, "x_k": round(x0, 10), "f_x_k": round(fx, 10),
            "f_prime_x_k": round(fpx, 10), "x_next": round(x1, 10),
            "error": round(error, 10),
        })
        
        f_x1 = f(x1)
        if _is_converged(f_x1, error, eps_eff):
            # Post-convergence sanity check: ensure |f(root)| is actually small
            if abs(f_x1) > eps_eff * 1000 and abs(f_x1) > 0.01:
                # Convergence detected but f(x) is still large — false convergence
                msg = (f"False convergence detected: |f({x1:.6f})| = {abs(f_x1):.2e} >> epsilon = {eps_eff:.1e}. "
                       f"Newton-Raphson may be stuck at a local extremum or flat region.")
                return {
                    "success": False,
                    "message": msg,
                    "root": round(x1, 10), "f_root": round(f_x1, 10),
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations,
                    "execution_time": round(time.time() - start_time, 6),
                    "convergence_data": _get_convergence_data(iterations),
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
                }
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Root found at x = {x1:.10f} after {k+1} iterations",
                "root": round(x1, 10), "f_root": round(f_x1, 10),
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 6),
                "convergence_data": _get_convergence_data(iterations),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            }
        
        x0 = x1
    
    execution_time = time.time() - start_time
    final_fx = f(x0)
    
    # Check if converged despite max iterations
    if abs(final_fx) < eps_eff * 1000 or error < eps_eff * 1000:
        return {
            "success": True,
            "message": f"Root found at x = {x0:.10f} after {max_iterations} iterations (f(x) ≈ 0)",
            "root": round(x0, 10), "f_root": round(final_fx, 10),
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
        }
    
    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "root": round(x0, 10), "f_root": round(final_fx, 10),
        "iterations_count": max_iterations, "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 6),
        "convergence_data": _get_convergence_data(iterations),
        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
    }


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    """Secant Method for finding roots."""
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    
    start_time = time.time()
    f_x0 = f(x0)
    f_x1 = f(x1)
    error = 0.0
    
    for k in range(max_iterations):
        if abs(f_x1 - f_x0) < 1e-15:
            return {
                "success": False,
                "message": "Division by zero in Secant method",
                "iterations": iterations,
            }
        
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x2 - x1)
        f_x2 = f(x2)
        
        iterations.append({
            "k": k + 1, "x_prev": round(x0, 10), "x_k": round(x1, 10),
            "f_x_k": round(f_x1, 10), "x_next": round(x2, 10),
            "error": round(error, 10),
        })
        
        if _is_converged(f_x2, error, eps_eff):
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Root found at x = {x2:.10f} after {k+1} iterations",
                "root": round(x2, 10), "f_root": round(f_x2, 10),
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 6),
                "convergence_data": _get_convergence_data(iterations),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            }
        
        x0, x1 = x1, x2
        f_x0, f_x1 = f_x1, f_x2
    
    execution_time = time.time() - start_time
    
    if abs(f_x1) < eps_eff * 1000 or error < eps_eff * 1000:
        return {
            "success": True,
            "message": f"Root found at x = {x1:.10f} after {max_iterations} iterations (f(x) ≈ 0)",
            "root": round(x1, 10), "f_root": round(f_x1, 10),
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
        }
    
    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "root": round(x1, 10), "f_root": round(f_x1, 10),
        "iterations_count": max_iterations, "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 6),
        "convergence_data": _get_convergence_data(iterations),
        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
    }


def fixed_point_iteration(
    f: Callable[[float], float],
    x0: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> dict:
    """Fixed Point Iteration method for x = g(x)."""
    eps_eff = _effective_epsilon(epsilon)
    iterations = []
    
    # Check contraction condition |g'(x₀)|
    contraction_info = None
    try:
        h = 1e-6
        g_prime = (f(x0 + h) - f(x0 - h)) / (2 * h)
        if abs(g_prime) >= 1:
            contraction_info = f"|g'(x₀)| = {abs(g_prime):.4f} ≥ 1 — g(x) có thể không thỏa điều kiện hội tụ của phương pháp Lặp Đơn."
        else:
            contraction_info = f"|g'(x₀)| = {abs(g_prime):.4f} < 1 — điều kiện hội tụ có khả năng được thỏa mãn."
    except Exception:
        pass
    
    start_time = time.time()
    error = 0.0
    
    for k in range(max_iterations):
        # Divergence check
        if abs(x0) > 1e10:
            return {
                "success": False,
                "message": "Phương pháp lặp đơn phân kỳ — giá trị vượt quá giới hạn (|x| > 10^10).",
                "iterations": iterations,
                "contraction_info": contraction_info,
            }
        try:
            x1 = f(x0)
        except (OverflowError, ValueError, FloatingPointError):
            return {
                "success": False,
                "message": "Phương pháp lặp đơn phân kỳ — tràn số khi tính g(x).",
                "iterations": iterations,
                "contraction_info": contraction_info,
            }
        
        if isinstance(x1, complex):
            return {
                "success": False,
                "message": f"Phương pháp lặp đơn sinh giá trị phức tại vòng lặp {k+1}: g({x0:.6f}) = {x1}. Dùng _real_cbrt hoặc _real_root để tránh số phức.",
                "iterations": iterations,
                "contraction_info": contraction_info,
            }
        
        if not np.isfinite(x1):
            return {
                "success": False,
                "message": "Phương pháp lặp đơn phân kỳ — kết quả chứa NaN hoặc vô cực.",
                "iterations": iterations,
                "contraction_info": contraction_info,
            }
        
        error = abs(x1 - x0)
        
        iterations.append({
            "k": k + 1, "x_k": round(x0, 10),
            "g_x_k": round(x1, 10), "error": round(error, 10),
        })
        
        if error < eps_eff:
            execution_time = time.time() - start_time
            return {
                "success": True,
                "message": f"Fixed point found at x = {x1:.10f} after {k+1} iterations",
                "root": round(x1, 10), "f_root": round(error, 10),
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations,
                "execution_time": round(execution_time, 6),
                "convergence_data": _get_convergence_data(iterations),
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
                "contraction_info": contraction_info,
            }
        
        x0 = x1
    
    execution_time = time.time() - start_time
    
    if error < eps_eff * 1000:
        return {
            "success": True,
            "message": f"Fixed point found at x = {x0:.10f} after {max_iterations} iterations (Δ ≈ 0)",
            "root": round(x0, 10), "f_root": 0.0,
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations,
            "execution_time": round(execution_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
            "contraction_info": contraction_info,
        }
    
    return {
        "success": False,
        "message": f"Did not converge after {max_iterations} iterations",
        "root": round(x0, 10), "f_root": 0.0,
        "iterations_count": max_iterations, "final_error": round(error, 10),
        "iterations": iterations,
        "execution_time": round(execution_time, 6),
        "convergence_data": _get_convergence_data(iterations),
        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON,
        "contraction_info": contraction_info,
    }


def _get_convergence_data(iterations: List[dict]) -> dict:
    """Extract convergence data for plotting."""
    if not iterations:
        return {"iterations": [], "errors": [], "x_values": []}
    k_values = [it["k"] for it in iterations]
    error_values = [it.get("error", 0) for it in iterations]
    x_values = [it.get("x_k", 0) for it in iterations]
    return {
        "iterations": k_values,
        "errors": error_values,
        "x_values": x_values,
    }