"""Eigenvalue & Eigenvector algorithms with detailed step-by-step solutions."""
import sys
import numpy as np
from typing import List, Optional
import time

MACHINE_EPSILON = sys.float_info.epsilon
MIN_EPSILON = 1e-15


def _effective_epsilon(epsilon: float) -> float:
    return max(epsilon, MIN_EPSILON)


def _matrix_to_latex(mat: np.ndarray, precision: int = 7) -> str:
    """Convert a numpy matrix to LaTeX bmatrix string."""
    m, n = mat.shape
    if m == 0 or n == 0:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = []
    for i in range(m):
        cells = " & ".join(
            f"{float(mat[i, j]):.{precision}f}" if abs(float(mat[i, j])) >= 1e-12 else "0"
            for j in range(n)
        )
        rows.append(cells)
    return r"\begin{bmatrix}" + " \\\\ ".join(rows) + r"\end{bmatrix}"


def _round_list(lst: List[float], precision: int = 10) -> List[float]:
    return [round(v, precision) for v in lst]


# =============================================================================
# 1. Characteristic Polynomial
# =============================================================================

def characteristic_polynomial(A: List[List[float]]) -> dict:
    """Compute all eigenvalues via the characteristic polynomial.

    Theory: p(λ) = det(A - λI) = 0. Solve for λ.
    Uses numpy.linalg.eigvals for computation, suitable for reference/verification.

    Application condition: A must be square.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông để tính trị riêng."}
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    step_counter = 0

    # Step 0: Intro
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Đa thức đặc trưng."
            f" Điều kiện áp dụng: A vuông ({n}×{n})."
            f" Nguyên lý: det(A − λI) = 0 → giải tìm λ."
        ),
    })
    step_counter += 1

    # Step 1: Form A - λI symbolically
    lam_sym = "λ"
    symbolic_rows = []
    for i in range(n):
        cells = []
        for j in range(n):
            val = mat[i, j]
            if i == j:
                if abs(val) < 1e-15:
                    cells.append(f"−{lam_sym}")
                elif val > 0:
                    cells.append(f"{val:.4f} − {lam_sym}")
                else:
                    cells.append(f"{val:.4f} − {lam_sym}")
            else:
                cells.append(f"{val:.4f}")
        symbolic_rows.append(" & ".join(cells))

    step_desc = f"Bước {step_counter}: Lập ma trận A − λI = [{'] ['.join(symbolic_rows)}]."
    steps.append({
        "step": step_counter,
        "description": step_desc,
        "matrix": [[round(float(mat[i, j]), 7) for j in range(n)] for i in range(n)],
    })
    step_counter += 1

    # Step 2: Compute characteristic polynomial coefficients via Faddeev–LeVerrier
    try:
        coeffs = np.poly(mat)
        # np.poly returns [1, -trace, ..., (-1)^n·det]
        poly_str = f"p({lam_sym}) = {lam_sym}^{n}"
        for k in range(1, n + 1):
            c = coeffs[k]
            if abs(c) < 1e-15:
                continue
            power = n - k
            if power == 0:
                term = f"{c:+.7f}"
            elif power == 1:
                term = f"{c:+.7f}·{lam_sym}"
            else:
                term = f"{c:+.7f}·{lam_sym}^{power}"
            poly_str += term
        poly_str += " = 0"

        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Khai triển det(A − λI) → đa thức đặc trưng bậc {n}: {poly_str}."
            ),
        })
    except Exception:
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Khai triển det(A − λI) → đa thức đặc trưng bậc {n}."
                f" Hệ số tính bằng thuật toán Faddeev–LeVerrier."
            ),
        })
    step_counter += 1

    # Step 3: Solve polynomial
    start_time = time.time()
    eigenvalues = np.linalg.eigvals(mat)
    eigenvalues_sorted = sorted(eigenvalues, key=lambda x: abs(x), reverse=True)
    execution_time = time.time() - start_time

    ev_list = []
    for ev in eigenvalues_sorted:
        if abs(ev.imag) < 1e-12:
            ev_list.append(round(float(ev.real), 10))
        else:
            ev_list.append(complex(round(ev.real, 10), round(ev.imag, 10)))

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Giải p(λ) = 0 → {len(ev_list)} trị riêng: "
            + ", ".join(f"λ{i+1} = {v}" for i, v in enumerate(ev_list))
            + "."
        ),
    })
    step_counter += 1

    # Step 4: Check accuracy
    verify_info = ""
    try:
        trace_A = float(np.trace(mat))
        trace_ev = sum(float(ev.real) if isinstance(ev, complex) else float(ev) for ev in ev_list)
        det_A = float(np.linalg.det(mat))
        det_ev = 1.0
        for ev in ev_list:
            det_ev *= float(ev.real) if isinstance(ev, complex) else float(ev)
        verify_info = (
            f" Kiểm tra: Σλᵢ = {trace_ev:.7f} (≈ trace(A) = {trace_A:.7f}),"
            f" Πλᵢ = {det_ev:.7f} (≈ det(A) = {det_A:.7f})."
        )
    except Exception:
        pass

    steps.append({
        "step": step_counter,
        "description": f"Bước {step_counter}: Hoàn tất.{verify_info}",
    })

    return {
        "success": True,
        "message": f"Tìm thấy {len(eigenvalues)} trị riêng từ đa thức đặc trưng.",
        "eigenvalues": ev_list,
        "steps": steps,
        "execution_time": round(execution_time, 7),
    }


# =============================================================================
# 2. Power Iteration
# =============================================================================

def power_iteration(A: List[List[float]], epsilon: float = 1e-6, max_iterations: int = 100,
                    initial_guess: Optional[List[float]] = None) -> dict:
    """Find the dominant eigenvalue & eigenvector via Power Iteration.

    Theory: v^{(k+1)} = A·v^{(k)} / ||A·v^{(k)}||, λ^{(k)} = v^{(k)}^T·A·v^{(k)}.
    Converges to the eigenvalue with largest absolute value (and its eigenvector).

    Application condition: A square. Converges if dominant eigenvalue is unique in magnitude.
    Does NOT converge if |λ1| = |λ2| for two distinct eigenvalues.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông."}
    mat = np.array(A, dtype=float)
    eps_eff = _effective_epsilon(epsilon)

    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory & conditions
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Power Iteration (lũy thừa) — tìm trị riêng trội nhất λ_max."
            f" Điều kiện áp dụng: A vuông {n}×{n}, |λ_max| > |λ_khác| (trị riêng trội duy nhất về độ lớn)."
            f" ε = {eps_eff}, max_iter = {max_iterations}."
            f" Công thức: v^(k+1) = A·v^(k) / ||A·v^(k)||; λ^(k) = v^(k)^T·A·v^(k)."
        ),
    })
    step_counter += 1

    # Step 1: Initial guess
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Khởi tạo v⁽⁰⁾ = [{', '.join(str(round(float(x[i]), 7)) for i in range(n))}]^T"
            f" (đã chuẩn hóa ||v||₂ = 1)."
        ),
    })
    step_counter += 1

    start_time = time.time()
    iterations: list[dict] = []
    lambda_old = 0.0

    for k in range(max_iterations):
        y = mat @ x
        lambda_new = float(x @ y)
        x_new = y / np.linalg.norm(y)
        error = abs(lambda_new - lambda_old)

        iter_entry = {
            "k": k + 1,
            "lambda": round(lambda_new, 10),
            "x": [round(float(x_new[i]), 10) for i in range(n)],
            "error": round(error, 10),
        }
        iterations.append(iter_entry)

        # Show first 4 iterations + convergence step in steps
        if k < 4:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Vòng {k+1}: v^({k+1}) = A·v^({k}) = [{', '.join(str(round(float(y[i]), 7)) for i in range(n))}]^T,"
                    f" chuẩn hóa → λ^({k+1}) = {round(lambda_new, 7)},"
                    f" ε = |λ_new − λ_old| = {round(error, 10)}."
                ),
                "iteration": iter_entry,
            })
            step_counter += 1

        if error < eps_eff:
            if k >= 4:
                steps.append({
                    "step": step_counter,
                    "description": (
                        f"Vòng {k+1}: λ^({k+1}) = {round(lambda_new, 7)},"
                        f" ε = {round(error, 10)} < {eps_eff} → HỘI TỤ."
                    ),
                    "iteration": iter_entry,
                })
                step_counter += 1

            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Hội tụ sau {k+1} vòng lặp."
                    f" λ_max ≈ {round(lambda_new, 10)},"
                    f" v_max = [{', '.join(str(round(float(x_new[i]), 10)) for i in range(n))}]^T."
                    f" Sai số |λ^(k+1) − λ^(k)| = {round(error, 10)} < ε = {eps_eff}."
                ),
            })
            return {
                "success": True, "message": f"Power Iteration hội tụ sau {k+1} bước.",
                "dominant_eigenvalue": round(lambda_new, 10),
                "dominant_eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations, "steps": steps,
                "execution_time": round(time.time() - start_time, 7),
            }
        x = x_new
        lambda_old = lambda_new

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Không hội tụ sau {max_iterations} vòng."
            f" λ cuối ≈ {round(lambda_old, 10)}."
            f" Sai số cuối = {round(abs(lambda_old - float(x @ (mat @ x))), 10)} ≥ ε."
            f" Có thể do |λ₁| ≈ |λ₂| hoặc cần tăng max_iter."
        ),
    })
    return {
        "success": False, "message": f"Power Iteration không hội tụ sau {max_iterations} bước.",
        "dominant_eigenvalue": round(lambda_old, 10),
        "dominant_eigenvector": [round(float(x[i]), 10) for i in range(n)],
        "iterations_count": max_iterations, "final_error": round(abs(lambda_old - float(x @ (mat @ x))), 10),
        "iterations": iterations, "steps": steps,
        "execution_time": round(time.time() - start_time, 7),
    }


# =============================================================================
# 3. Inverse Power Iteration
# =============================================================================

def inverse_power_iteration(A: List[List[float]], shift: float = 0.0,
                            epsilon: float = 1e-6, max_iterations: int = 100,
                            initial_guess: Optional[List[float]] = None) -> dict:
    """Find eigenvalue closest to 'shift' via Inverse Power Iteration.

    Theory: Apply Power Iteration to (A - σI)^{-1}.
    Converges to eigenvalue λ closest to σ, with convergence rate |λ - σ| / |λ_next - σ|.

    Application condition: A square, (A - σI) invertible.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông."}
    mat = np.array(A, dtype=float)
    eps_eff = _effective_epsilon(epsilon)

    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Inverse Power Iteration (lũy thừa nghịch đảo)."
            f" Điều kiện áp dụng: A vuông {n}×{n}, (A − σI) khả nghịch."
            f" σ = {shift}, ε = {eps_eff}, max_iter = {max_iterations}."
            f" Nguyên lý: Áp dụng Power Iteration cho (A − σI)⁻¹ → tìm λ gần σ nhất."
            f" v^(k+1) = (A − σI)⁻¹·v^(k) / ||(A − σI)⁻¹·v^(k)||."
        ),
    })
    step_counter += 1

    # Step 1: Form (A - σI) and compute its inverse
    shifted = mat - shift * np.eye(n)
    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Lập ma trận B = A − {shift}·I:"
        ),
        "matrix": [[round(float(shifted[i, j]), 7) for j in range(n)] for i in range(n)],
    })
    step_counter += 1

    try:
        shift_inv = np.linalg.inv(shifted)
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Tính B⁻¹ = (A − {shift}·I)⁻¹ thành công."
                f" (Kiểm tra: B khả nghịch → phương pháp áp dụng được.)"
            ),
            "matrix": [[round(float(shift_inv[i, j]), 7) for j in range(n)] for i in range(n)],
        })
        step_counter += 1
    except np.linalg.LinAlgError:
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: (A − {shift}·I) suy biến, không khả nghịch."
                f" Chọn σ khác hoặc σ đã trùng với trị riêng."
            ),
        })
        return {"success": False, "message": f"Ma trận (A - {shift}·I) suy biến. Chọn σ khác."}

    # Step 2: Initial guess
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Khởi tạo v⁽⁰⁾ = [{', '.join(str(round(float(x[i]), 7)) for i in range(n))}]^T"
            f" (đã chuẩn hóa ||v||₂ = 1). Áp dụng lặp lũy thừa cho B⁻¹."
        ),
    })
    step_counter += 1

    start_time = time.time()
    iterations: list[dict] = []
    lambda_old = 0.0

    for k in range(max_iterations):
        y = shift_inv @ x
        x_new = y / np.linalg.norm(y)
        lambda_new = float(x_new @ (mat @ x_new))
        error = abs(lambda_new - lambda_old)

        iter_entry = {
            "k": k + 1, "lambda": round(lambda_new, 10),
            "x": [round(float(x_new[i]), 10) for i in range(n)],
            "error": round(error, 10),
        }
        iterations.append(iter_entry)

        if k < 4:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Vòng {k+1}: w = B⁻¹·v^({k}),"
                    f" v^({k+1}) = w / ||w||,"
                    f" λ^({k+1}) = v^(k+1)^T·A·v^(k+1) = {round(lambda_new, 7)},"
                    f" ε = {round(error, 10)}."
                ),
                "iteration": iter_entry,
            })
            step_counter += 1

        if error < eps_eff:
            if k >= 4:
                steps.append({
                    "step": step_counter,
                    "description": (
                        f"Vòng {k+1}: λ = {round(lambda_new, 7)}, ε = {round(error, 10)} < {eps_eff} → HỘI TỤ."
                    ),
                    "iteration": iter_entry,
                })
                step_counter += 1

            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Hội tụ sau {k+1} vòng."
                    f" λ ≈ {round(lambda_new, 10)} (trị riêng gần σ = {shift} nhất)."
                    f" Sai số = {round(error, 10)} < ε."
                ),
            })
            return {
                "success": True, "message": f"Inverse Power Iteration hội tụ sau {k+1} bước.",
                "eigenvalue": round(lambda_new, 10),
                "eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations, "shift": shift, "steps": steps,
                "execution_time": round(time.time() - start_time, 7),
            }
        x = x_new
        lambda_old = lambda_new

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Không hội tụ sau {max_iterations} vòng."
            f" λ cuối ≈ {round(lambda_old, 10)}."
            f" Có thể cần tăng max_iter hoặc chọn σ khác."
        ),
    })
    return {
        "success": False, "message": f"Inverse Power Iteration không hội tụ sau {max_iterations} bước.",
        "eigenvalue": round(lambda_old, 10),
        "eigenvector": [round(float(x[i]), 10) for i in range(n)],
        "iterations_count": max_iterations, "final_error": round(abs(lambda_old - float(x @ (mat @ x))), 10),
        "iterations": iterations, "shift": shift, "steps": steps,
        "execution_time": round(time.time() - start_time, 7),
    }


# =============================================================================
# 4. Rayleigh Quotient Iteration
# =============================================================================

def rayleigh_quotient_iteration(A: List[List[float]],
                                epsilon: float = 1e-6, max_iterations: int = 100,
                                initial_guess: Optional[List[float]] = None) -> dict:
    """Find an eigenvalue-eigenvector pair via Rayleigh Quotient Iteration.

    Theory: At each step, compute Rayleigh quotient σ = v^T·A·v / v^T·v,
    then solve (A - σI)·y = v for improved eigenvector.
    Cubic convergence for symmetric matrices.

    Application condition: A square. Converges to SOME eigenvalue.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông."}
    mat = np.array(A, dtype=float)
    eps_eff = _effective_epsilon(epsilon)

    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Rayleigh Quotient Iteration."
            f" Điều kiện áp dụng: A vuông {n}×{n}. ε = {eps_eff}, max_iter = {max_iterations}."
            f" Nguyên lý: σ^(k) = v^(k)^T·A·v^(k) / ||v^(k)||² (Rayleigh quotient),"
            f" giải (A − σ^(k)·I)·y = v^(k) → v^(k+1) = y / ||y||."
            f" Hội tụ bậc 3 với ma trận đối xứng, bậc 2 với ma trận không đối xứng."
        ),
    })
    step_counter += 1

    # Step 1: Initial guess
    x = np.array(initial_guess, dtype=float) if (initial_guess and len(initial_guess) == n) else np.ones(n, dtype=float)
    x = x / np.linalg.norm(x)
    sigma = float(x @ (mat @ x))

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Khởi tạo v⁽⁰⁾ = [{', '.join(str(round(float(x[i]), 7)) for i in range(n))}]^T"
            f" (đã chuẩn hóa)."
            f" Tính Rayleigh quotient: σ⁽⁰⁾ = v⁽⁰⁾^T·A·v⁽⁰⁾ = {round(sigma, 7)}."
        ),
    })
    step_counter += 1

    start_time = time.time()
    iterations: list[dict] = []

    for k in range(max_iterations):
        try:
            y = np.linalg.solve(mat - sigma * np.eye(n), x)
        except np.linalg.LinAlgError:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: (A − {round(sigma, 10)}·I) suy biến"
                    f" → tìm thấy trị riêng chính xác λ ≈ {round(sigma, 10)}."
                    f" v = [{', '.join(str(round(float(x[i]), 10)) for i in range(n))}]^T."
                ),
            })
            return {
                "success": True, "message": f"Trị riêng tìm thấy tại bước {k+1}: λ ≈ {sigma}. Hội tụ.",
                "eigenvalue": round(sigma, 10),
                "eigenvector": [round(float(x[i]), 10) for i in range(n)],
                "iterations_count": k + 1, "final_error": 0.0,
                "iterations": iterations, "steps": steps,
                "execution_time": round(time.time() - start_time, 7),
            }

        x_new = y / np.linalg.norm(y)
        sigma_new = float(x_new @ (mat @ x_new))
        error = abs(sigma_new - sigma)

        iter_entry = {
            "k": k + 1, "sigma": round(sigma_new, 10),
            "x": [round(float(x_new[i]), 10) for i in range(n)],
            "error": round(error, 10),
        }
        iterations.append(iter_entry)

        if k < 4 or error < eps_eff:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Vòng {k+1}: σ = {round(sigma, 7)}, giải (A − σI)·y = v^(k)"
                    f" → v^(k+1) = [{', '.join(str(round(float(x_new[i]), 7)) for i in range(n))}]^T,"
                    f" σ_new = {round(sigma_new, 7)}, ε = {round(error, 10)}"
                    + (" → HỘI TỤ." if error < eps_eff else ".")
                ),
                "iteration": iter_entry,
            })
            step_counter += 1

        if error < eps_eff:
            return {
                "success": True, "message": f"Rayleigh Quotient Iteration hội tụ sau {k+1} bước.",
                "eigenvalue": round(sigma_new, 10),
                "eigenvector": [round(float(x_new[i]), 10) for i in range(n)],
                "iterations_count": k + 1, "final_error": round(error, 10),
                "iterations": iterations, "steps": steps,
                "execution_time": round(time.time() - start_time, 7),
            }
        x = x_new
        sigma = sigma_new

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Không hội tụ sau {max_iterations} vòng."
            f" σ cuối ≈ {round(sigma, 10)}."
            f" Sai số cuối = {round(abs(sigma - float(x @ (mat @ x))), 10)}."
        ),
    })
    return {
        "success": False, "message": f"Rayleigh Quotient Iteration không hội tụ sau {max_iterations} bước.",
        "eigenvalue": round(sigma, 10),
        "eigenvector": [round(float(x[i]), 10) for i in range(n)],
        "iterations_count": max_iterations, "final_error": round(abs(sigma - float(x @ (mat @ x))), 10),
        "iterations": iterations, "steps": steps,
        "execution_time": round(time.time() - start_time, 7),
    }


# =============================================================================
# 5. Jacobi Method (Classical — Givens rotations)
# =============================================================================

def jacobi_method(A: List[List[float]], epsilon: float = 1e-6, max_iterations: int = 100) -> dict:
    """Find all eigenvalues & eigenvectors for SYMMETRIC matrices via Jacobi method.

    Theory: Iteratively apply Givens rotations to zero out off-diagonal elements.
    Repeated sweeps over all (i,j) with i < j, zeroing the largest |a_ij|.
    Converges to diagonal matrix with eigenvalues on diagonal.

    Application condition: A must be SQUARE and SYMMETRIC (A = A^T).
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông."}
    mat = np.array(A, dtype=float)
    eps_eff = _effective_epsilon(epsilon)

    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Jacobi (cổ điển) — dùng phép quay Givens cho ma trận ĐỐI XỨNG."
            f" Điều kiện áp dụng: A vuông {n}×{n} và A = A^T (đối xứng)."
            f" ε = {eps_eff}, max_iter = {max_iterations}."
            f" Nguyên lý: Tìm a_pq có |a_pq| lớn nhất ngoài đường chéo,"
            f" tính góc θ = ½·arctan(2a_pq / (a_pp − a_qq)),"
            f" dựng ma trận quay J (Givens), cập nhật A' = J^T·A·J."
            f" Lặp đến khi mọi |a_ij| < ε (i≠j)."
        ),
    })
    step_counter += 1

    # Step 1: Check symmetry
    if not np.allclose(mat, mat.T, atol=1e-10):
        steps.append({
            "step": step_counter,
            "description": "Bước 1: Kiểm tra đối xứng: A ≠ A^T → phương pháp Jacobi không áp dụng được.",
        })
        return {"success": False,
                "message": "Phương pháp Jacobi yêu cầu ma trận đối xứng. Ma trận A không đối xứng.",
                "steps": steps}

    steps.append({
        "step": step_counter,
        "description": "Bước 1: Kiểm tra đối xứng: A = A^T → thỏa mãn. Bắt đầu quá trình Jacobi.",
        "matrix": [[round(float(mat[i, j]), 7) for j in range(n)] for i in range(n)],
    })
    step_counter += 1

    start_time = time.time()

    A_k = mat.copy()
    V_k = np.eye(n)  # Eigenvector accumulator

    global_iter = 0
    sweep = 0
    sweep_steps_shown = 0  # Limit detailed sweep steps to avoid overflow

    while global_iter < max_iterations:
        sweep += 1

        # Find largest off-diagonal element (p, q) with p < q
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A_k[i, j]) > max_val:
                    max_val = abs(A_k[i, j])
                    p, q = i, j

        if max_val < eps_eff:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Sweep {sweep} — |a_{{{p+1},{q+1}}}| = {round(max_val, 10)} < ε"
                    f" → TẤT CẢ PHẦN TỬ NGOÀI ĐƯỜNG CHÉO < ε. HỘI TỤ."
                ),
            })
            step_counter += 1
            break

        # Show sweep header (limit to avoid excessive steps)
        if sweep_steps_shown < 6:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Sweep {sweep} — |a_{{{p+1},{q+1}}}| = {round(max_val, 7)}"
                    f" là phần tử ngoài đường chéo lớn nhất (≥ ε = {eps_eff})."
                    f" Thực hiện phép quay Givens khử a_{{{p+1},{q+1}}}."
                ),
            })
            step_counter += 1

        # Compute rotation angle
        if abs(A_k[p, p] - A_k[q, q]) < 1e-15:
            theta = np.pi / 4  # 45 degrees
        else:
            theta = 0.5 * np.arctan2(2.0 * A_k[p, q], A_k[p, p] - A_k[q, q])

        c = np.cos(theta)
        s = np.sin(theta)

        if sweep_steps_shown < 6:
            steps.append({
                "step": step_counter,
                "description": (
                    f"  S({sweep}): θ = ½·arctan(2·{round(float(A_k[p, q]), 7)}"
                    f" / ({round(float(A_k[p, p]), 7)} − {round(float(A_k[q, q]), 7)}))"
                    f" = {round(float(theta), 7)} rad = {round(float(np.degrees(theta)), 4)}°."
                    f" cos(θ) = {round(float(c), 7)}, sin(θ) = {round(float(s), 7)}."
                ),
            })
            step_counter += 1

        # Apply rotation: A' = J^T @ A @ J
        # In-place update (only rows/cols p, q change)
        # Row updates
        for j in range(n):
            if j not in (p, q):
                a_pj = c * A_k[p, j] + s * A_k[q, j]
                a_qj = -s * A_k[p, j] + c * A_k[q, j]
                A_k[p, j] = a_pj
                A_k[q, j] = a_qj
                A_k[j, p] = a_pj  # Symmetry
                A_k[j, q] = a_qj  # Symmetry

        a_pp = c * c * A_k[p, p] + 2 * s * c * A_k[p, q] + s * s * A_k[q, q]
        a_qq = s * s * A_k[p, p] - 2 * s * c * A_k[p, q] + c * c * A_k[q, q]
        a_pq = (c * c - s * s) * A_k[p, q] + s * c * (A_k[q, q] - A_k[p, p])

        A_k[p, p] = a_pp
        A_k[q, q] = a_qq
        A_k[p, q] = a_pq
        A_k[q, p] = a_pq  # Should be ~0

        # Update eigenvector accumulator: V = V @ J
        for i in range(n):
            v_ip = c * V_k[i, p] + s * V_k[i, q]
            v_iq = -s * V_k[i, p] + c * V_k[i, q]
            V_k[i, p] = v_ip
            V_k[i, q] = v_iq

        if sweep_steps_shown < 6:
            steps.append({
                "step": step_counter,
                "description": (
                    f"  Sau phép quay: a_{{{p+1},{q+1}}} ≈ {round(float(A_k[p, q]), 10)}"
                    f" (bị khử → ≈ 0). Tiếp tục sweep tiếp theo."
                ),
            })
            step_counter += 1

        sweep_steps_shown += 1
        global_iter += 1

    execution_time = time.time() - start_time

    if global_iter >= max_iterations:
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Đạt max_iter = {max_iterations}."
                f" Phần tử ngoài đường chéo lớn nhất còn = {round(max_val, 10)}."
            ),
        })
        step_counter += 1

    # Extract eigenvalues (diagonal of A_k) and eigenvectors (columns of V_k)
    eigenvalues = np.diag(A_k)
    eigenpairs = []
    for i in range(n):
        eigenpairs.append({
            "eigenvalue": round(float(eigenvalues[i]), 10),
            "eigenvector": [round(float(V_k[j, i]), 10) for j in range(n)],
        })

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Hoàn tất sau {sweep} sweep ({global_iter} phép quay Givens)."
            f" Trị riêng (đường chéo): "
            + ", ".join(f"λ{i+1} = {round(float(eigenvalues[i]), 10)}" for i in range(n))
            + "."
            f" Vector riêng tương ứng trong ma trận V (tích các phép quay)."
        ),
    })

    return {
        "success": True,
        "message": f"Phương pháp Jacobi tìm thấy {n} cặp (trị riêng, vector riêng) sau {sweep} sweep.",
        "eigenpairs": eigenpairs,
        "eigenvalues": [round(float(ev), 10) for ev in eigenvalues],
        "steps": steps,
        "execution_time": round(execution_time, 7),
    }


# =============================================================================
# 6. Danielewski Method — Frobenius companion form
# =============================================================================

def danielewski_method(A: List[List[float]]) -> dict:
    """Find all eigenvalues & eigenvectors via Danielewski's method.

    Theory (Danielewski / Frobenius companion method):
    1. Transform A to Frobenius (companion) form P via similarity transformations:
       P = S^{-1} · A · S
    2. P has the form:
       [0   1   0  ... 0  ]
       [0   0   1  ... 0  ]
       [...             1  ]
       [p0  p1  p2 ... p_{n-1}]
    3. The last row of P gives the characteristic polynomial:
       λ^n - p_{n-1}·λ^{n-1} - ... - p_1·λ - p_0 = 0
    4. Solve the polynomial for eigenvalues.
    5. For each eigenvalue λ_i, find eigenvector by solving (A - λ_i·I)·v = 0.

    Application condition: A must be square. The method works for any square matrix
    (though numerically sensitive for large/high-condition matrices).
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": "Ma trận phải vuông để áp dụng Danielewski."}

    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phương pháp Danielewski (dạng Frobenius / companion)."
            f" Điều kiện áp dụng: A vuông {n}×{n}."
            f" Nguyên lý: Biến đổi đồng dạng A → P (dạng companion),"
            f" hàng cuối của P chứa hệ số đa thức đặc trưng."
            f" Giải đa thức → trị riêng → vector riêng qua nullspace của (A − λI)."
        ),
    })
    step_counter += 1

    # Step 1: Input matrix
    M = mat.copy()
    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Ma trận đầu vào A ({n}×{n})."
            f" Bắt đầu quá trình khử Danielewski — đưa về dạng Frobenius."
        ),
        "matrix": [[round(float(M[i, j]), 7) for j in range(n)] for i in range(n)],
    })
    step_counter += 1

    start_time = time.time()

    # Build Frobenius form via similarity transformations
    # Algorithm: for k = n-2, n-3, ..., 0:
    #   If M[k+1, k] != 1, normalize row k+1
    #   For i != k+1, eliminate M[i, k] using row operations + corresponding column operations
    total_ops = 0

    for k in range(n - 2, -1, -1):
        pivot_row = k + 1
        pivot_val = M[pivot_row, k]

        if abs(pivot_val) < 1e-15:
            # Try to find a non-zero pivot above
            found = False
            for r in range(k, -1, -1):
                if abs(M[r, k]) > 1e-15:
                    # Swap rows r and pivot_row
                    M[[r, pivot_row]] = M[[pivot_row, r]]
                    # Swap columns too (similarity transformation)
                    M[:, [r, pivot_row]] = M[:, [pivot_row, r]]
                    pivot_val = M[pivot_row, k]
                    found = True
                    steps.append({
                        "step": step_counter,
                        "description": (
                            f"Bước {step_counter}: Khử cột {k+1}: hoán vị đồng dạng hàng/cột"
                            f" {r+1} ↔ {pivot_row+1} để có pivot ≠ 0."
                        ),
                    })
                    step_counter += 1
                    break
            if not found:
                # Cannot proceed — fall back
                break

        # Normalize pivot row
        if abs(pivot_val - 1.0) > 1e-15:
            factor = 1.0 / pivot_val
            M[pivot_row] *= factor
            # Compensate with column operation (similarity)
            M[:, pivot_row] *= pivot_val
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Chuẩn hóa hàng {pivot_row+1}"
                    f" (chia cho {round(float(pivot_val), 7)})"
                    f" và cột {pivot_row+1} (nhân với {round(float(pivot_val), 7)})"
                    f" → bảo toàn đồng dạng."
                ),
            })
            step_counter += 1

        # Eliminate other entries in column k
        for i in range(n):
            if i == pivot_row:
                continue
            if abs(M[i, k]) < 1e-15:
                continue

            factor = M[i, k]
            total_ops += 1

            # Row operation: R_i = R_i - factor * R_{pivot_row}
            for j in range(n):
                M[i, j] -= factor * M[pivot_row, j]

            # Column operation (similarity): C_{pivot_row} = C_{pivot_row} + factor * C_i
            for j in range(n):
                M[j, pivot_row] += factor * M[j, i]

            # Only log detailed steps for small matrices or first few ops
            if total_ops <= 8 or n <= 3:
                steps.append({
                    "step": step_counter,
                    "description": (
                        f"Bước {step_counter}: Khử M[{i+1},{k+1}] = {round(float(factor), 7)}"
                        f" bằng R_{i+1} ← R_{i+1} − {round(float(factor), 7)}·R_{pivot_row+1}"
                        f" và C_{pivot_row+1} ← C_{pivot_row+1} + {round(float(factor), 7)}·C_{i+1}."
                    ),
                })
                step_counter += 1

    if total_ops > 8 and n > 3:
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Tổng cộng {total_ops} phép khử đồng dạng đã thực hiện"
                f" (chỉ hiển thị các phép đầu tiên)."
            ),
        })
        step_counter += 1

    # Extract characteristic polynomial from last row
    poly_coeffs = [float(M[n - 1, j]) for j in range(n)]

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Dạng Frobenius (companion) hoàn tất."
            f" Hàng cuối P[{n}, :] = [{', '.join(f'{c:.7f}' for c in poly_coeffs)}]."
            f" Đa thức đặc trưng: p(λ) = λ^{n}"
            + "".join(f" − {poly_coeffs[n-1-j]:.7f}·λ^{n-1-j}" for j in range(n))
            + " = 0."
        ),
        "matrix": [[round(float(M[i, j]), 7) for j in range(n)] for i in range(n)],
    })
    step_counter += 1

    # Solve the polynomial
    eigenvalues_list = []
    try:
        # Build polynomial: λ^n - p_{n-1}λ^{n-1} - ... - p_0 = 0
        coeffs = [1.0]
        for j in range(n - 1, -1, -1):
            coeffs.append(-float(M[n - 1, j]))
        eigenvalues = np.roots(coeffs)
        for ev in eigenvalues:
            if abs(ev.imag) < 1e-12:
                eigenvalues_list.append(round(float(ev.real), 10))
            else:
                eigenvalues_list.append(complex(round(ev.real, 10), round(ev.imag, 10)))
    except Exception:
        eigenvalues_list_raw = np.linalg.eigvals(mat)
        for ev in eigenvalues_list_raw:
            if abs(ev.imag) < 1e-12:
                eigenvalues_list.append(round(float(ev.real), 10))
            else:
                eigenvalues_list.append(complex(round(ev.real, 10), round(ev.imag, 10)))

    eigenvalues_sorted = sorted(eigenvalues_list, key=lambda x: abs(x), reverse=True)

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Giải p(λ) = 0 → {len(eigenvalues_sorted)} trị riêng: "
            + ", ".join(f"λ{i+1} = {v}" for i, v in enumerate(eigenvalues_sorted))
            + "."
        ),
    })
    step_counter += 1

    # Find eigenvectors for each eigenvalue
    eigenpairs = []
    for ev in eigenvalues_sorted:
        ev_real = ev.real if isinstance(ev, complex) else ev
        try:
            shifted = mat - ev_real * np.eye(n)
            _, _, vh = np.linalg.svd(shifted)
            eigvec = vh[-1]
            eigvec = eigvec / np.linalg.norm(eigvec)
            eigenpairs.append({
                "eigenvalue": ev,
                "eigenvector": [round(float(eigvec[i]), 10) for i in range(n)],
            })
        except Exception:
            eigenpairs.append({
                "eigenvalue": ev,
                "eigenvector": [1.0 if i == 0 else 0.0 for i in range(n)],
            })

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Tìm vector riêng cho mỗi λ bằng nullspace của (A − λI) qua SVD."
            f" Với mỗi λᵢ, giải (A − λᵢI)·v = 0 → vᵢ là vector cuối của V^T trong SVD."
            f" Hoàn tất: {len(eigenpairs)} cặp (λ, v)."
        ),
    })

    execution_time = time.time() - start_time

    return {
        "success": True,
        "message": f"Phương pháp Danielewski tìm thấy {len(eigenpairs)} cặp (trị riêng, vector riêng).",
        "eigenpairs": eigenpairs,
        "eigenvalues": eigenvalues_sorted,
        "steps": steps,
        "execution_time": round(execution_time, 7),
    }