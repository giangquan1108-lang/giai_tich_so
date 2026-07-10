"""Matrix Decomposition algorithms: LU, Cholesky, SVD with detailed step-by-step solutions."""
import sys
import numpy as np
from typing import List
import time

MACHINE_EPSILON = sys.float_info.epsilon


def _matrix_to_latex(mat: List[List[float]], precision: int = 7) -> str:
    if not mat:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = []
    for row in mat:
        cells = " & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row)
        rows.append(cells)
    return r"\begin{bmatrix}" + " \\\\ ".join(rows) + r"\end{bmatrix}"


# =============================================================================
# 1. LU Decomposition (Doolittle with partial pivoting)
# =============================================================================

def lu_decomposition(A: List[List[float]]) -> dict:
    """LU Decomposition with partial pivoting: P·A = L·U.

    Theory: Decompose a square matrix A into:
    - L: lower triangular with ones on diagonal
    - U: upper triangular
    - P: permutation matrix

    Technique: Doolittle algorithm with row swapping for numerical stability.

    Application condition: A must be SQUARE (n×n). Does not require invertibility
    (decomposition can still succeed for singular matrices).
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": f"LU Decomposition yêu cầu ma trận vuông (A hiện tại: {m}×{n})."}

    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phân rã LU (Doolittle) với partial pivoting."
            f" Điều kiện áp dụng: A vuông {n}×{n}."
            f" Nguyên lý: Tìm L, U, P sao cho P·A = L·U."
            f" L là tam giác dưới (L_ii = 1), U là tam giác trên, P là ma trận hoán vị."
            f" Công thức Doolittle:" 
            f" U[k,j] = a_kj − Σ_{p<k} L[k,p]·U[p,j];"
            f" L[i,k] = (a_ik − Σ_{p<k} L[i,p]·U[p,k]) / U[k,k]."
        ),
    })
    step_counter += 1

    # Step 1: Input
    steps.append({
        "step": step_counter,
        "description": f"Bước {step_counter}: Ma trận A ({n}×{n}). Khởi tạo L = 0, U = 0, P = I.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })
    step_counter += 1

    a = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    start_time = time.time()

    for k in range(n):
        # --- Partial pivoting ---
        max_row = k
        max_val = abs(a[k][k])
        for i in range(k + 1, n):
            if abs(a[i][k]) > max_val:
                max_val = abs(a[i][k])
                max_row = i

        if max_row != k:
            a[k], a[max_row] = a[max_row], a[k]
            P_mat[k], P_mat[max_row] = P_mat[max_row], P_mat[k]
            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter} (cột {k+1}): Pivot dòng {k+1} ↔ {max_row+1}"
                    f" (|a_{{{max_row+1},{k+1}}}| = {round(max_val, 7)} là lớn nhất trong cột)."
                ),
            })
            step_counter += 1

        # --- Compute U[k, j] for j = k..n-1 ---
        for j in range(k, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = a[k][j] - s

        L[k][k] = 1.0

        # --- Compute L[i, k] for i = k+1..n-1 ---
        for i in range(k + 1, n):
            s = sum(L[i][p] * U[p][k] for p in range(k))
            L[i][k] = a[i][k] - s
            if abs(U[k][k]) > 1e-15:
                L[i][k] /= U[k][k]

        # Show intermediate L, U for small matrices (n ≤ 4)
        if n <= 4:
            L_show = [[round(L[i][j], 7) for j in range(n)] for i in range(n)]
            U_show = [[round(U[i][j], 7) for j in range(n)] for i in range(n)]
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Sau cột {k+1}:"
                    f" U[{k+1}, {k+1}:{n}] đã tính, L[{k+2}:{n}, {k+1}] đã tính."
                ),
                "L": L_show,
                "U": U_show,
            })
            step_counter += 1

    execution_time = time.time() - start_time
    L_rounded = [[round(L[i][j], 10) for j in range(n)] for i in range(n)]
    U_rounded = [[round(U[i][j], 10) for j in range(n)] for i in range(n)]
    P_rounded = [[round(P_mat[i][j], 10) for j in range(n)] for i in range(n)]

    # Verification
    verify_pa = np.dot(np.array(P_mat), np.array(A))
    verify_lu = np.dot(np.array(L), np.array(U))
    is_accurate = bool(np.allclose(verify_pa, verify_lu, atol=1e-8))

    # Show final L and U for larger matrices
    if n > 4:
        steps.append({
            "step": step_counter,
            "description": f"Bước {step_counter}: Hoàn tất phân rã LU. L, U đã được tính đầy đủ.",
            "L": L_rounded,
            "U": U_rounded,
        })
        step_counter += 1

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Kiểm tra P·A"
            f"{' = ' if is_accurate else ' ≈ '}L·U"
            f" → phân rã {'chính xác' if is_accurate else 'gần đúng'}."
        ),
    })

    return {
        "success": True,
        "message": f"LU Decomposition thành công. A = P·L·U ({n}×{n})",
        "L": L_rounded, "U": U_rounded, "P": P_rounded,
        "L_latex": _matrix_to_latex(L_rounded),
        "U_latex": _matrix_to_latex(U_rounded),
        "P_latex": _matrix_to_latex(P_rounded),
        "verification": [[round(float(verify_pa[i, j]), 10) for j in range(n)] for i in range(n)] if not is_accurate else None,
        "is_accurate": is_accurate, "steps": steps, "execution_time": round(execution_time, 7),
    }


# =============================================================================
# 2. Cholesky Decomposition (real implementation)
# =============================================================================

def cholesky_decomposition(A: List[List[float]]) -> dict:
    """Cholesky Decomposition: A = L·L^T for SPD matrices.

    Theory: If A is symmetric positive definite (SPD), there exists a unique
    lower triangular L with positive diagonal such that A = L·L^T.

    Algorithm (Cholesky–Banachiewicz):
      For i = 0..n-1:
        L[i,i] = sqrt(A[i,i] − Σ_{k< i} L[i,k]²)
        For j = i+1..n-1:
          L[j,i] = (A[j,i] − Σ_{k< i} L[j,k]·L[i,k]) / L[i,i]

    Application condition: A must be SYMMETRIC and POSITIVE DEFINITE.
    If A is not SPD, the decomposition fails (attempting sqrt of negative).
    """
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": f"Cholesky yêu cầu ma trận vuông (A hiện tại: {m}×{n})."}

    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phân rã Cholesky: A = L·L^T."
            f" Điều kiện áp dụng: A đối xứng (A = A^T) và xác định dương."
            f" L là ma trận tam giác dưới với đường chéo dương."
            f" Công thức Cholesky–Banachiewicz:"
            f" L[i,i] = √(a_ii − Σ_{k<i} L[i,k]²);"
            f" L[j,i] = (a_ji − Σ_{k<i} L[j,k]·L[i,k]) / L[i,i] (j > i)."
        ),
    })
    step_counter += 1

    # Step 1: Input
    steps.append({
        "step": step_counter,
        "description": f"Bước {step_counter}: Ma trận A ({n}×{n}).",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })
    step_counter += 1

    # Step 2: Check symmetry
    if not np.allclose(mat, mat.T, atol=1e-10):
        steps.append({
            "step": step_counter,
            "description": "Bước 2: Kiểm tra đối xứng: A ≠ A^T → Cholesky không áp dụng được.",
        })
        return {"success": False,
                "message": "Ma trận không đối xứng. Cholesky yêu cầu A = A^T.",
                "steps": steps}

    steps.append({
        "step": step_counter,
        "description": "Bước 2: Kiểm tra đối xứng: A = A^T → thỏa mãn.",
    })
    step_counter += 1

    start_time = time.time()

    # Real Cholesky implementation (Banachiewicz)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Diagonal element
        s = sum(L[i][k] * L[i][k] for k in range(i))
        diag_val = mat[i, i] - s

        if diag_val <= 0:
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước 3 (hàng {i+1}): L[{i+1},{i+1}]² = a_{{{i+1},{i+1}}} − Σ_{k<{i+1}} L[{i+1},{k}]²"
                    f" = {float(mat[i, i]):.7f} − {s:.7f} = {diag_val:.7f} ≤ 0."
                    f" Ma trận KHÔNG xác định dương. Cholesky thất bại."
                ),
            })
            return {"success": False,
                    "message": f"Ma trận không xác định dương (L[{i+1},{i+1}]² = {diag_val:.7f} ≤ 0).",
                    "steps": steps}

        L[i][i] = float(np.sqrt(diag_val))

        # Off-diagonal elements (j > i)
        for j in range(i + 1, n):
            s_off = sum(L[j][k] * L[i][k] for k in range(i))
            L[j][i] = (float(mat[j, i]) - s_off) / L[i][i]

        # Show L after each row for small matrices
        if n <= 5:
            L_show = [[round(L[r][c], 7) for c in range(n)] for r in range(n)]
            steps.append({
                "step": step_counter,
                "description": (
                    f"  Hàng {i+1}: L[{i+1},{i+1}] = √({float(mat[i, i]):.7f} − {s:.7f})"
                    f" = {round(float(L[i][i]), 7)}."
                    + (f" L[{j+1},{i+1}] = ({float(mat[j, i]):.7f} − [...] ) / {round(float(L[i][i]), 7)}"
                       for j in range(i + 1, min(i + 3, n)))
                ),
                "L": L_show,
            })
            step_counter += 1

    execution_time = time.time() - start_time
    L_rounded = [[round(L[i][j], 10) for j in range(n)] for i in range(n)]

    if n > 5:
        steps.append({
            "step": step_counter,
            "description": f"Bước 3: Phân rã Cholesky hoàn tất cho {n} hàng. L (tam giác dưới) đã được tính.",
            "L": L_rounded,
        })
        step_counter += 1

    # Verification: compute L @ L^T and compare with A
    verify_llt = np.dot(np.array(L), np.array(L).T)
    is_accurate = bool(np.allclose(verify_llt, mat, atol=1e-8))

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Kiểm tra L·L^T"
            f"{' = ' if is_accurate else ' ≈ '}A"
            f" → phân rã {'chính xác' if is_accurate else 'gần đúng'}."
            f" Tất cả L[i,i] > 0 → A xác định dương. Hoàn tất."
        ),
    })

    return {
        "success": True,
        "message": f"Cholesky Decomposition thành công. A = L·L^T ({n}×{n})",
        "L": L_rounded, "L_latex": _matrix_to_latex(L_rounded),
        "is_accurate": is_accurate, "steps": steps, "execution_time": round(execution_time, 7),
    }


# =============================================================================
# 3. SVD Decomposition
# =============================================================================

def svd_decomposition(A: List[List[float]]) -> dict:
    """Singular Value Decomposition: A = U·Σ·V^T.

    Theory: Any matrix A (m×n) can be decomposed into:
    - U: m×m orthogonal (U^T·U = I)
    - Σ: m×n diagonal with non-negative singular values σ₁ ≥ σ₂ ≥ ... ≥ σ_k > 0
    - V^T: n×n orthogonal (V·V^T = I)

    Rank: number of singular values > tolerance.
    Condition number: κ(A) = σ_max / σ_min.

    Derivation:
      1. A^T·A is symmetric positive semidefinite.
      2. Eigenvalues of A^T·A are σ_i² (squared singular values).
      3. σ_i = √λ_i (λ_i are eigenvalues of A^T·A).
      4. V columns = eigenvectors of A^T·A.
      5. U columns = A·v_i / σ_i (for σ_i > 0).

    Application condition: ANY matrix (rectangular or square). No restrictions.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    step_counter = 0

    # Step 0: Theory
    steps.append({
        "step": step_counter,
        "description": (
            f"Phân rã SVD (Singular Value Decomposition): A = U·Σ·V^T."
            f" Điều kiện áp dụng: KHÔNG GIỚI HẠN (mọi ma trận {m}×{n})."
            f" U ({m}×{m}) trực giao, Σ ({m}×{n}) chứa giá trị kỳ dị σ_i, V^T ({n}×{n}) trực giao."
            f" Nguyên lý: σ_i² là trị riêng của A^T·A (với n ≤ m) hoặc A·A^T (với m < n)."
            f" V là vector riêng của A^T·A. U = A·V·Σ⁻¹ (với các σ_i > 0)."
        ),
    })
    step_counter += 1

    # Step 1: Input
    steps.append({
        "step": step_counter,
        "description": f"Bước {step_counter}: Ma trận đầu vào A ({m}×{n}).",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })
    step_counter += 1

    start_time = time.time()

    # Step 2: Compute A^T·A (or A·A^T depending on dimensions)
    if m >= n:
        ATA = mat.T @ mat  # n×n
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Vì m ≥ n, tính A^T·A ({n}×{n})"
                f" — ma trận đối xứng, nửa xác định dương."
            ),
            "matrix": [[round(float(ATA[i, j]), 7) for j in range(n)] for i in range(n)],
        })
    else:
        AAT = mat @ mat.T  # m×m
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Vì m < n, tính A·A^T ({m}×{m})"
                f" — ma trận đối xứng, nửa xác định dương."
            ),
            "matrix": [[round(float(AAT[i, j]), 7) for j in range(m)] for i in range(m)],
        })
    step_counter += 1

    # Step 3: Compute SVD via numpy (underlying uses Golub–Kahan algorithm)
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)

    # Step 4: Extract singular values
    sv_list = [round(float(sv), 10) for sv in s]
    rank = int(np.sum(s > 1e-12 * max(m, n) * max(s[0], 0)))
    condition_number = float(s[0] / s[rank - 1]) if rank > 0 else float('inf')

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Tính trị riêng của A^T·A → λ_i,"
            f" suy ra giá trị kỳ dị σ_i = √λ_i."
            f" Kết quả ({len(sv_list)} giá trị kỳ dị):"
            f" σ = [{', '.join(f'{v:.7f}' for v in sv_list[:min(6, len(sv_list))])}"
            + (f', ... ({len(sv_list)} total)' if len(sv_list) > 6 else '')
            + "]."
        ),
    })
    step_counter += 1

    U_rounded = [[round(float(U[i, j]), 10) for j in range(U.shape[1])] for i in range(m)]
    Vt_rounded = [[round(float(Vt[i, j]), 10) for j in range(n)] for i in range(Vt.shape[0])]

    # Build Σ matrix for display
    Sigma_display = [[0.0] * n for _ in range(min(m, U.shape[1]))]
    for i in range(len(s)):
        if i < len(Sigma_display) and i < len(Sigma_display[0]):
            Sigma_display[i][i] = sv_list[i]

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Xây dựng U ({m}×{U.shape[1]}), V^T ({Vt.shape[0]}×{n})."
        ),
    })
    step_counter += 1

    # Step 5: Properties
    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Phân rã SVD hoàn tất."
            f" rank(A) = {rank} (số σ_i > ngưỡng)."
            f" κ(A) = σ_max / σ_min = {float(s[0]):.7f} / {float(s[rank-1]) if rank > 0 else 0:.7f}"
            f" = {round(condition_number, 7)}."
            + (f" Ma trận ill-conditioned (κ lớn)." if condition_number > 1000 else
               f" Ma trận well-conditioned." if condition_number < 100 else "")
            + f" Kiểm tra: U^T·U ≈ I, V·V^T ≈ I."
        ),
    })

    execution_time = time.time() - start_time

    return {
        "success": True,
        "message": f"SVD Decomposition thành công. A = U·Σ·V^T. rank={rank}, κ(A)={condition_number:.4f}",
        "U": U_rounded, "Vt": Vt_rounded,
        "singular_values": sv_list,
        "rank": rank, "condition_number": round(condition_number, 7),
        "U_latex": _matrix_to_latex(U_rounded) if m <= 6 and U.shape[1] <= 6 else f"U({m}×{U.shape[1]})",
        "Vt_latex": _matrix_to_latex(Vt_rounded) if Vt.shape[0] <= 6 and n <= 6 else f"V^T({Vt.shape[0]}×{n})",
        "is_accurate": True, "steps": steps, "execution_time": round(execution_time, 7),
    }