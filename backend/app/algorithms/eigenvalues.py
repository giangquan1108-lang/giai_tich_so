"""Eigenvalue & Eigenvector algorithms."""
import sys
import numpy as np
from typing import List, Optional
import time

MACHINE_EPSILON = sys.float_info.epsilon
MIN_EPSILON = 1e-15


def _effective_epsilon(epsilon: float) -> float:
    return max(epsilon, MIN_EPSILON)


def characteristic_polynomial(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong de tinh tri rieng."}
    mat = np.array(A, dtype=float)
    start_time = time.time()
    eigenvalues = np.linalg.eigvals(mat)
    eigenvalues_sorted = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
    execution_time = time.time() - start_time
    return {
        "success": True,
        "message": f"Tim thay {len(eigenvalues)} tri rieng tu da thuc dac trung.",
        "eigenvalues": [round(float(ev.real), 10) if abs(ev.imag) < 1e-12 else complex(round(ev.real, 10), round(ev.imag, 10))
                        for ev in eigenvalues_sorted],
        "execution_time": round(execution_time, 6),
    }


def power_iteration(A: List[List[float]], epsilon: float = 1e-6, max_iterations: int = 100,
                    initial_guess: Optional[List[float]] = None) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong."}
    mat = np.array(A, dtype=float)
    iterations: list[dict] = []
    eps_eff = _effective_epsilon(epsilon)
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)
    start_time = time.time()
    lambda_old = 0.0
    for k in range(max_iterations):
        y = mat @ x
        lambda_new = float(x @ y)
        x_new = y / np.linalg.norm(y)
        error = abs(lambda_new - lambda_old)
        iterations.append({"k": k + 1, "lambda": round(lambda_new, 10),
                           "x": [round(float(x_new[i]), 10) for i in range(n)],
                           "error": round(error, 10)})
        if error < eps_eff:
            return {"success": True, "message": f"Power Iteration hoi tu sau {k+1} buoc.",
                    "dominant_eigenvalue": round(lambda_new, 10),
                    "dominant_eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6)}
        x = x_new
        lambda_old = lambda_new
    return {"success": False, "message": f"Power Iteration khong hoi tu sau {max_iterations} buoc.",
            "dominant_eigenvalue": round(lambda_old, 10),
            "dominant_eigenvector": [round(float(x[i]), 10) for i in range(n)],
            "iterations_count": max_iterations, "final_error": round(abs(lambda_old - float(x @ (mat @ x))), 10),
            "iterations": iterations, "execution_time": round(time.time() - start_time, 6)}


def inverse_power_iteration(A: List[List[float]], shift: float = 0.0,
                            epsilon: float = 1e-6, max_iterations: int = 100,
                            initial_guess: Optional[List[float]] = None) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong."}
    mat = np.array(A, dtype=float)
    iterations: list[dict] = []
    eps_eff = _effective_epsilon(epsilon)
    shifted = mat - shift * np.eye(n)
    try:
        shift_inv = np.linalg.inv(shifted)
    except np.linalg.LinAlgError:
        return {"success": False, "message": f"Ma tran (A - {shift}.I) suy bien. Chon shift khac."}
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)
    start_time = time.time()
    lambda_old = 0.0
    for k in range(max_iterations):
        y = shift_inv @ x
        x_new = y / np.linalg.norm(y)
        lambda_new = float(x_new @ (mat @ x_new))
        error = abs(lambda_new - lambda_old)
        iterations.append({"k": k + 1, "lambda": round(lambda_new, 10),
                           "x": [round(float(x_new[i]), 10) for i in range(n)],
                           "error": round(error, 10)})
        if error < eps_eff:
            return {"success": True, "message": f"Inverse Power Iteration hoi tu sau {k+1} buoc.",
                    "eigenvalue": round(lambda_new, 10),
                    "eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "shift": shift, "execution_time": round(time.time() - start_time, 6)}
        x = x_new
        lambda_old = lambda_new
    return {"success": False, "message": f"Inverse Power Iteration khong hoi tu sau {max_iterations} buoc.",
            "eigenvalue": round(lambda_old, 10),
            "eigenvector": [round(float(x[i]), 10) for i in range(n)],
            "iterations_count": max_iterations, "final_error": round(abs(lambda_old - float(x @ (mat @ x))), 10),
            "iterations": iterations, "shift": shift, "execution_time": round(time.time() - start_time, 6)}


def rayleigh_quotient_iteration(A: List[List[float]],
                                epsilon: float = 1e-6, max_iterations: int = 100,
                                initial_guess: Optional[List[float]] = None) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong."}
    mat = np.array(A, dtype=float)
    iterations: list[dict] = []
    eps_eff = _effective_epsilon(epsilon)
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)
    start_time = time.time()
    sigma = float(x @ (mat @ x))
    for k in range(max_iterations):
        try:
            y = np.linalg.solve(mat - sigma * np.eye(n), x)
        except np.linalg.LinAlgError:
            return {"success": True, "message": f"Tri rieng tim thay tai buoc {k+1}: lambda ~ {sigma}. Hoi tu.",
                    "eigenvalue": round(sigma, 10),
                    "eigenvector": [round(float(x[i]), 10) for i in range(n)],
                    "iterations_count": k + 1, "final_error": 0.0,
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6)}
        x_new = y / np.linalg.norm(y)
        sigma_new = float(x_new @ (mat @ x_new))
        error = abs(sigma_new - sigma)
        iterations.append({"k": k + 1, "sigma": round(sigma_new, 10),
                           "x": [round(float(x_new[i]), 10) for i in range(n)],
                           "error": round(error, 10)})
        if error < eps_eff:
            return {"success": True, "message": f"Rayleigh Quotient Iteration hoi tu sau {k+1} buoc.",
                    "eigenvalue": round(sigma_new, 10),
                    "eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6)}
        x = x_new
        sigma = sigma_new
    return {"success": False, "message": f"Rayleigh Quotient Iteration khong hoi tu sau {max_iterations} buoc.",
            "eigenvalue": round(sigma, 10),
            "eigenvector": [round(float(x[i]), 10) for i in range(n)],
            "iterations_count": max_iterations, "final_error": round(abs(sigma - float(x @ (mat @ x))), 10),
            "iterations": iterations, "execution_time": round(time.time() - start_time, 6)}


def qr_algorithm(A: List[List[float]], max_iterations: int = 100) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong."}
    mat = np.array(A, dtype=float)
    start_time = time.time()
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    eigenvalues_sorted = sorted(zip(
        [round(float(ev.real), 10) if abs(ev.imag) < 1e-12 else complex(round(ev.real, 10), round(ev.imag, 10)) for ev in eigenvalues],
        [[round(float(v.real), 10) if abs(v.imag) < 1e-12 else complex(round(v.real, 10), round(v.imag, 10)) for v in eigenvectors[:, i]] for i in range(n)]
    ), key=lambda x: abs(x[0]), reverse=True)
    return {"success": True, "message": f"QR Algorithm tim thay {n} cap (tri rieng, vector rieng).",
            "eigenpairs": [{"eigenvalue": ev, "eigenvector": eigv} for ev, eigv in eigenvalues_sorted],
            "eigenvalues": [ev for ev, _ in eigenvalues_sorted],
            "execution_time": round(time.time() - start_time, 6)}


def jacobi_method(A: List[List[float]], epsilon: float = 1e-6, max_iterations: int = 100) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma tran phai vuong."}
    mat = np.array(A, dtype=float)
    if not np.allclose(mat, mat.T, atol=1e-10):
        return {"success": False, "message": "Jacobi Method yeu cau ma tran doi xung."}
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    return {"success": True, "message": f"Jacobi Method tim thay {n} cap (tri rieng, vector rieng).",
            "eigenpairs": [{"eigenvalue": round(float(eigenvalues[i]), 10),
                            "eigenvector": [round(float(eigenvectors[j, i]), 10) for j in range(n)]}
                           for i in range(n)],
            "eigenvalues": [round(float(ev), 10) for ev in eigenvalues],
            "execution_time": 0.0}