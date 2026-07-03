"""Interpolation algorithms: Lagrange, Newton Forward, Newton Backward, Divided Differences."""
import numpy as np
from typing import List
import time


def lagrange_interpolation(
    x_points: List[float],
    y_points: List[float],
    x_value: float = None,
) -> dict:
    """Lagrange Interpolation.
    
    L(x) = sum_{i=0}^{n} y_i * l_i(x)
    where l_i(x) = product_{j!=i} (x - x_j) / (x_i - x_j)
    """
    n = len(x_points)
    iterations = []
    
    start_time = time.time()
    
    # Build Lagrange basis polynomials info
    for i in range(n):
        terms = []
        for j in range(n):
            if j != i:
                terms.append(f"(x - {x_points[j]})/({x_points[i]} - {x_points[j]})")
        iterations.append({
            "basis_index": i,
            "l_i": f"l_{i}(x) = " + " * ".join(terms),
            "y_i": y_points[i],
        })
    
    # Build polynomial string
    poly_parts = []
    for i in range(n):
        numerator_terms = []
        denominator = 1.0
        for j in range(n):
            if j != i:
                numerator_terms.append(f"(x - {x_points[j]})")
                denominator *= (x_points[i] - x_points[j])
        coeff = y_points[i] / denominator
        num_str = " * ".join(numerator_terms) if numerator_terms else "1"
        poly_parts.append(f"({round(coeff, 10)}) * {num_str}")
    
    polynomial = " + ".join(poly_parts)
    
    # Evaluate at x_value
    interpolated_value = None
    if x_value is not None:
        interpolated_value = 0.0
        for i in range(n):
            li = 1.0
            for j in range(n):
                if j != i:
                    li *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
            interpolated_value += y_points[i] * li
        interpolated_value = round(interpolated_value, 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Lagrange interpolation polynomial of degree {n-1} computed",
        "interpolated_value": interpolated_value,
        "polynomial": polynomial,
        "iterations": iterations,
        "formula": r"L(x) = \sum_{i=0}^{" + str(n-1) + r"} y_i \cdot l_i(x)",
        "execution_time": round(execution_time, 6),
    }


def newton_forward_interpolation(
    x_points: List[float],
    y_points: List[float],
    x_value: float = None,
) -> dict:
    """Newton Forward Difference Interpolation.
    
    P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1) + ...
    """
    n = len(x_points)
    start_time = time.time()
    
    # Build divided difference table
    dd_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd_table[i][0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    iterations = []
    for j in range(n):
        col = [round(dd_table[i][j], 10) for i in range(n - j)]
        iterations.append({
            "order": j,
            "differences": col,
        })
    
    # Build polynomial string
    poly_parts = []
    for j in range(n):
        coeff = dd_table[0][j]
        terms = [f"(x - {x_points[i]})" for i in range(j)]
        term_str = " * ".join(terms) if terms else "1"
        poly_parts.append(f"({round(coeff, 10)}) * {term_str}")
    
    polynomial = " + ".join(poly_parts)
    
    # Evaluate at x_value
    interpolated_value = None
    if x_value is not None:
        interpolated_value = dd_table[0][0]
        product_term = 1.0
        for j in range(1, n):
            product_term *= (x_value - x_points[j - 1])
            interpolated_value += dd_table[0][j] * product_term
        interpolated_value = round(interpolated_value, 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Newton forward interpolation polynomial of degree {n-1} computed",
        "interpolated_value": interpolated_value,
        "polynomial": polynomial,
        "divided_diff_table": [[round(dd_table[i][j], 10) for j in range(n - i)] for i in range(n)],
        "iterations": iterations,
        "formula": r"P(x) = f[x_0] + \sum_{j=1}^{n} f[x_0,...,x_j] \prod_{i=0}^{j-1}(x - x_i)",
        "execution_time": round(execution_time, 6),
    }


def newton_backward_interpolation(
    x_points: List[float],
    y_points: List[float],
    x_value: float = None,
) -> dict:
    """Newton Backward Difference Interpolation.
    
    Uses backward differences for better accuracy near the end of the data.
    """
    n = len(x_points)
    start_time = time.time()
    
    # Build divided difference table (same as forward)
    dd_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd_table[i][0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    iterations = []
    for j in range(n):
        col = [round(dd_table[i][j], 10) for i in range(n - j)]
        iterations.append({
            "order": j,
            "differences": col,
        })
    
    # Build polynomial using backward differences
    poly_parts = []
    for j in range(n):
        coeff = dd_table[n - 1 - j][j]
        terms = [f"(x - {x_points[n - 1 - i]})" for i in range(j)]
        term_str = " * ".join(terms) if terms else "1"
        poly_parts.append(f"({round(coeff, 10)}) * {term_str}")
    
    polynomial = " + ".join(poly_parts)
    
    # Evaluate at x_value
    interpolated_value = None
    if x_value is not None:
        interpolated_value = dd_table[n - 1][0]
        product_term = 1.0
        for j in range(1, n):
            product_term *= (x_value - x_points[n - j])
            interpolated_value += dd_table[n - 1 - j][j] * product_term
        interpolated_value = round(interpolated_value, 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Newton backward interpolation polynomial of degree {n-1} computed",
        "interpolated_value": interpolated_value,
        "polynomial": polynomial,
        "divided_diff_table": [[round(dd_table[i][j], 10) for j in range(n - i)] for i in range(n)],
        "iterations": iterations,
        "formula": r"P(x) = f[x_n] + \sum_{j=1}^{n} \nabla^j f[x_n] \prod_{i=0}^{j-1}(x - x_{n-i})",
        "execution_time": round(execution_time, 6),
    }


def divided_differences(
    x_points: List[float],
    y_points: List[float],
    x_value: float = None,
) -> dict:
    """Divided Differences method (same table as Newton, shows the full process)."""
    n = len(x_points)
    start_time = time.time()
    
    # Build divided difference table
    dd_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        dd_table[i][0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_points[i + j] - x_points[i])
    
    iterations = []
    for j in range(n):
        col = [round(dd_table[i][j], 10) for i in range(n - j)]
        iterations.append({
            "order": j,
            "differences": col,
        })
    
    # Polynomial
    poly_parts = []
    for j in range(n):
        coeff = dd_table[0][j]
        terms = [f"(x - {x_points[i]})" for i in range(j)]
        term_str = " * ".join(terms) if terms else "1"
        poly_parts.append(f"({round(coeff, 10)}) * {term_str}")
    
    polynomial = " + ".join(poly_parts)
    
    interpolated_value = None
    if x_value is not None:
        interpolated_value = dd_table[0][0]
        product_term = 1.0
        for j in range(1, n):
            product_term *= (x_value - x_points[j - 1])
            interpolated_value += dd_table[0][j] * product_term
        interpolated_value = round(interpolated_value, 10)
    
    execution_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Divided differences table computed for {n} points",
        "interpolated_value": interpolated_value,
        "polynomial": polynomial,
        "divided_diff_table": [[round(dd_table[i][j], 10) for j in range(n - i)] for i in range(n)],
        "iterations": iterations,
        "formula": r"f[x_0, x_1, ..., x_n] = \frac{f[x_1,...,x_n] - f[x_0,...,x_{n-1}]}{x_n - x_0}",
        "execution_time": round(execution_time, 6),
    }