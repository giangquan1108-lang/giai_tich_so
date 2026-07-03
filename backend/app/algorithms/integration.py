"""Numerical integration algorithms: Trapezoidal, Simpson 1/3, Simpson 3/8, Romberg."""
import numpy as np
from typing import Callable, List
import time


def trapezoidal(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> dict:
    """Trapezoidal Rule for numerical integration.
    
    Formula: h/2 * [f(a) + 2*sum(f(x_i)) + f(b)]
    where h = (b-a)/n
    """
    iterations = []
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    f_values = [f(x) for x in x_points]
    
    start_time = time.time()
    
    # Compute sum
    total = f_values[0] + f_values[n]
    for i in range(1, n):
        total += 2 * f_values[i]
    
    result = (h / 2) * total
    
    # Build iteration data for visualization (grouped)
    group_size = max(1, n // 20)
    for i in range(0, n + 1, group_size):
        end = min(i + group_size, n + 1)
        group_sum = sum(f_values[j] for j in range(i, end))
        iterations.append({
            "range": f"[{round(x_points[i], 6)}, {round(x_points[min(end-1, n)], 6)}]",
            "sum_f": round(group_sum, 10),
            "h": round(h, 10),
        })
    
    # Exact computation using more points for reference
    exact_result = None
    try:
        from scipy import integrate
        exact_result, _ = integrate.quad(f, a, b)
        exact_result = round(exact_result, 10)
    except Exception:
        pass
    
    error = None
    if exact_result is not None:
        error = round(abs(result - exact_result), 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Trapezoidal rule with n={n} subintervals",
        "result": round(result, 10),
        "exact_value": exact_result,
        "error": error,
        "iterations": iterations,
        "formula": r"T_n = \frac{h}{2}\left[f(a) + 2\sum_{i=1}^{n-1}f(x_i) + f(b)\right]",
        "plot_data": {
            "x": [round(x, 6) for x in x_points],
            "y": [round(y, 6) for y in f_values],
            "a": a,
            "b": b,
        },
        "execution_time": round(execution_time, 6),
    }


def simpson_one_third(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> dict:
    """Simpson's 1/3 Rule for numerical integration.
    
    Formula: h/3 * [f(x0) + 4*sum(odd) + 2*sum(even) + f(xn)]
    Requires n to be even.
    """
    if n % 2 != 0:
        n += 1
    
    iterations = []
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    f_values = [f(x) for x in x_points]
    
    start_time = time.time()
    
    # Simpson's 1/3 rule
    total = f_values[0] + f_values[n]
    for i in range(1, n, 2):
        total += 4 * f_values[i]
    for i in range(2, n, 2):
        total += 2 * f_values[i]
    
    result = (h / 3) * total
    
    # Build iteration data
    odd_sum = sum(4 * f_values[i] for i in range(1, n, 2))
    even_sum = sum(2 * f_values[i] for i in range(2, n, 2))
    iterations.append({
        "description": "Component sums",
        "f(a)": round(f_values[0], 10),
        "f(b)": round(f_values[n], 10),
        "4*odd_sum": round(odd_sum, 10),
        "2*even_sum": round(even_sum, 10),
        "total": round(total, 10),
    })
    
    exact_result = None
    try:
        from scipy import integrate
        exact_result, _ = integrate.quad(f, a, b)
        exact_result = round(exact_result, 10)
    except Exception:
        pass
    
    error = None
    if exact_result is not None:
        error = round(abs(result - exact_result), 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Simpson's 1/3 rule with n={n} subintervals",
        "result": round(result, 10),
        "exact_value": exact_result,
        "error": error,
        "iterations": iterations,
        "formula": r"S_n = \frac{h}{3}\left[f(x_0) + 4\sum_{i=1,3,...}^{n-1}f(x_i) + 2\sum_{i=2,4,...}^{n-2}f(x_i) + f(x_n)\right]",
        "plot_data": {
            "x": [round(x, 6) for x in x_points],
            "y": [round(y, 6) for y in f_values],
            "a": a,
            "b": b,
        },
        "execution_time": round(execution_time, 6),
    }


def simpson_three_eighth(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 100,
) -> dict:
    """Simpson's 3/8 Rule for numerical integration.
    
    Formula: 3h/8 * [f(x0) + 3*f(x1) + 3*f(x2) + 2*f(x3) + ...]
    Requires n to be a multiple of 3.
    """
    while n % 3 != 0:
        n += 1
    
    iterations = []
    h = (b - a) / n
    x_points = [a + i * h for i in range(n + 1)]
    f_values = [f(x) for x in x_points]
    
    start_time = time.time()
    
    # Simpson's 3/8 rule
    total = f_values[0] + f_values[n]
    for i in range(1, n):
        if i % 3 == 0:
            total += 2 * f_values[i]
        else:
            total += 3 * f_values[i]
    
    result = (3 * h / 8) * total
    
    iterations.append({
        "description": "Component sums",
        "f(a)": round(f_values[0], 10),
        "f(b)": round(f_values[n], 10),
        "total": round(total, 10),
    })
    
    exact_result = None
    try:
        from scipy import integrate
        exact_result, _ = integrate.quad(f, a, b)
        exact_result = round(exact_result, 10)
    except Exception:
        pass
    
    error = None
    if exact_result is not None:
        error = round(abs(result - exact_result), 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Simpson's 3/8 rule with n={n} subintervals",
        "result": round(result, 10),
        "exact_value": exact_result,
        "error": error,
        "iterations": iterations,
        "formula": r"S_n^{3/8} = \frac{3h}{8}\left[f(x_0) + 3f(x_1) + 3f(x_2) + 2f(x_3) + ...\right]",
        "plot_data": {
            "x": [round(x, 6) for x in x_points],
            "y": [round(y, 6) for y in f_values],
            "a": a,
            "b": b,
        },
        "execution_time": round(execution_time, 6),
    }


def romberg(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 10,
) -> dict:
    """Romberg Integration.
    
    Uses Richardson extrapolation on trapezoidal rule approximations.
    R(i,j) = R(i,j-1) + (R(i,j-1) - R(i-1,j-1)) / (4^j - 1)
    """
    iterations = []
    
    start_time = time.time()
    
    # Romberg table
    R = [[0.0] * (n + 1) for _ in range(n + 1)]
    
    # First column: trapezoidal rule with increasing subdivisions
    h = b - a
    R[0][0] = h / 2 * (f(a) + f(b))
    
    for i in range(1, n + 1):
        h /= 2
        # Sum new function values
        s = 0.0
        num_new = 2 ** (i - 1)
        for k in range(num_new):
            x = a + (2 * k + 1) * h
            s += f(x)
        R[i][0] = R[i - 1][0] / 2 + s * h
        
        # Richardson extrapolation
        for j in range(1, i + 1):
            R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (4 ** j - 1)
        
        iterations.append({
            "level": i,
            "h": round(h, 10),
            "R_i0": round(R[i][0], 10),
            "extrapolations": [round(R[i][j], 10) for j in range(1, i + 1)],
        })
    
    result = R[n][n]
    
    exact_result = None
    try:
        from scipy import integrate
        exact_result, _ = integrate.quad(f, a, b)
        exact_result = round(exact_result, 10)
    except Exception:
        pass
    
    error = None
    if exact_result is not None:
        error = round(abs(result - exact_result), 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Romberg integration with {n} levels of refinement",
        "result": round(result, 10),
        "exact_value": exact_result,
        "error": error,
        "iterations": iterations,
        "formula": r"R_{i,j} = R_{i,j-1} + \frac{R_{i,j-1} - R_{i-1,j-1}}{4^j - 1}",
        "romberg_table": [[round(R[i][j], 10) if j <= i else None for j in range(n + 1)] for i in range(n + 1)],
        "execution_time": round(execution_time, 6),
    }