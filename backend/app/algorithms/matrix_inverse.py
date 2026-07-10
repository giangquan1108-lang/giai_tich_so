"""Matrix Inverse algorithms: Gauss-Jordan, Adjoint, Cholesky, Bordering, Jacobi, Gauss-Seidel, Newton."""
import sys
import numpy as np
from typing import List
import time

MACHINE_EPSILON = sys.float_info.epsilon
MIN_EPSILON = 1e-15


def _effective_epsilon(epsilon: float) -> float:
    return max(epsilon, MIN_EPSILON)


def _matrix_rank(M: List[List[float]], tol: float = 1e-12) -> int:
    if not M or not M[0]:
        return 0
    mat = np.array(M, dtype=float)
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        return 0
    try:
        s = np.linalg.svd(mat, compute_uv=False)
        return int(np.sum(s > tol * max(mat.shape) * max(s[0], 0)))
    except Exception:
        return min(mat.shape)


def _validate_matrix_inverse(A: List[List[float]]) -> dict | None:
    if not A or not A[0]:
        return {"success": False, "message": "Ma trận A không được rỗng."}
    m = len(A)
    n = len(A[0])
    for i in range(m):
        for j in range(n):
            v = A[i][j]
            if not np.isfinite(float(v)):
                return {
                    "success": False,
                    "message": f"Dữ liệu không hợp lệ tại A[{i+1}][{j+1}] = {v}. Giá trị phải là số hữu hạn.",
                }
    if m != n:
        return {
            "success": False,
            "message": f"Không thể tính ma trận nghịch đảo vì A không phải ma trận vuông (A là {m}×{n}).",
        }
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-15:
        return {
            "success": False,
            "message": f"Ma trận suy biến (det(A) = {det:.6e} ≈ 0). Không tồn tại ma trận nghịch đảo.",
            "determinant": round(det, 10),
        }
    return None


def _compute_rank(A: List[List[float]]) -> int:
    return _matrix_rank(A)


def _inv_to_latex(inv: List[List[float]]) -> str:
    """Convert inverse matrix to LaTeX bmatrix."""
    n = len(inv)
    latex_rows = [" & ".join(str(v) for v in row) for row in inv]
    return r"A^{-1} = \begin{bmatrix}" + " \\\\ ".join(latex_rows) + r"\end{bmatrix}"


def _is_diagonally_dominant_strict(mat: np.ndarray, n: int) -> tuple[bool, bool]:
    """Check strict diagonal dominance by row and column."""
    row_dd = True
    col_dd = True
    for i in range(n):
        row_sum = sum(abs(mat[i, j]) for j in range(n) if j != i)
        if abs(mat[i, i]) <= row_sum:
            row_dd = False
        col_sum = sum(abs(mat[j, i]) for j in range(n) if j != i)
        if abs(mat[i, i]) <= col_sum:
            col_dd = False
    return row_dd, col_dd


def _build_result(mat: np.ndarray, inv: np.ndarray, n: int, method_name: str,
                  det_val: float, rank_val: int, steps: list, exec_time: float,
                  extra_iter_data: dict | None = None) -> dict:
    """Build standardized result dict."""
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    verify = np.dot(mat, inv)
    verify_rounded = [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)]
    is_accurate = bool(np.allclose(verify, np.eye(n), atol=1e-8))

    result = {
        "success": True,
        "message": f"Ma trận nghịch đảo tính thành công bằng {method_name}. det(A) = {round(det_val, 7)}",
        "determinant": round(det_val, 10),
        "rank": rank_val,
        "inverse": inv_rounded,
        "inverse_latex": _inv_to_latex(inv_rounded),
        "verification": verify_rounded,
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(exec_time, 7),
    }
    if extra_iter_data:
        result.update(extra_iter_data)
    return result


# =============================================================================
# 1. Gauss-Jordan Inverse
# =============================================================================

def matrix_inverse_gauss_jordan(A: List[List[float]]) -> dict:
    """Compute the inverse of a square matrix using Gauss-Jordan elimination.

    Principle: Augment [A|I], then apply elementary row operations to
    transform the left block into I. The right block becomes A^{-1}.

    Application condition: A is square (n x n) and non-singular (det(A) != 0).
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)

    steps: list[dict] = []

    # Step 0: Application condition check
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp khử Gauss-Jordan."
            f" Điều kiện áp dụng: A là ma trận vuông {n}×{n}, det(A) ≠ 0."
            f" det(A) = {round(det, 7)} ≠ 0 → thỏa mãn."
            f" Nguyên lý: Ghép [A|I], biến đổi sơ cấp hàng để đưa về [I|A⁻¹]."
        ),
    })

    # Step 1: Build augmented matrix
    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    steps.append({
        "step": 1,
        "description": f"Bước 1: Lập ma trận mở rộng [A|I] kích thước {n}×{2*n}.",
        "matrix": [[round(v, 7) for v in row] for row in aug],
    })

    start_time = time.time()
    col = 0
    step_counter = 2

    for k in range(n):
        # Partial pivoting: find row with max absolute value in column col
        max_row = k
        max_val = abs(aug[k][col]) if k < n else 0
        for i in range(k + 1, n):
            if abs(aug[i][col]) > max_val:
                max_val = abs(aug[i][col])
                max_row = i

        if max_val < 1e-15:
            return {
                "success": False,
                "message": f"Ma trận suy biến (zero pivot tại cột {col+1}), không tồn tại ma trận nghịch đảo.",
                "determinant": round(det, 10),
                "rank": rank_val,
            }

        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Đổi hàng R{k+1} ↔ R{max_row+1} (partial pivoting — "
                    f"chọn pivot có |a_{{{max_row+1},{col+1}}}| = {round(max_val, 7)} lớn nhất trong cột {col+1})."
                ),
                "matrix": [[round(v, 7) for v in row] for row in aug],
                "phase": "pivoting",
            })
            step_counter += 1

        pivot = aug[k][col]
        for j in range(2 * n):
            aug[k][j] /= pivot
        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Chuẩn hóa hàng R{k+1} = R{k+1} / {round(pivot, 7)} "
                f"(chia toàn bộ hàng cho pivot để a_{{{k+1},{col+1}}} = 1)."
            ),
            "matrix": [[round(v, 7) for v in row] for row in aug],
            "phase": "normalize",
        })
        step_counter += 1

        for i in range(n):
            if i == k or abs(aug[i][col]) < 1e-15:
                continue
            factor = aug[i][col]
            for j in range(2 * n):
                aug[i][j] -= factor * aug[k][j]
            steps.append({
                "step": step_counter,
                "description": (
                    f"Bước {step_counter}: Khử hàng R{i+1} = R{i+1} - {round(factor, 7)}·R{k+1} "
                    f"(loại bỏ phần tử a_{{{i+1},{col+1}}} = {round(factor, 7)} ở cột {col+1})."
                ),
                "matrix": [[round(v, 7) for v in row] for row in aug],
                "phase": "eliminate",
            })
            step_counter += 1
        col += 1

    execution_time = time.time() - start_time
    inv = np.array([[aug[i][n + j] for j in range(n)] for i in range(n)], dtype=float)

    steps.append({
        "step": step_counter,
        "description": (
            f"Bước {step_counter}: Hoàn tất. [I|A⁻¹] — vế trái đã trở thành ma trận đơn vị I,"
            f" vế phải chính là ma trận nghịch đảo A⁻¹."
            f" Kiểm tra: det(A) = {round(det, 7)}, rank(A) = {rank_val}."
        ),
        "inverse": [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)],
    })

    return _build_result(mat, inv, n, "Gauss-Jordan", det, rank_val, steps, execution_time)


# =============================================================================
# 2. Adjoint Inverse
# =============================================================================

def matrix_inverse_adjoint(A: List[List[float]]) -> dict:
    """Compute the inverse using the Adjoint (Adjugate) method: A^{-1} = Adj(A) / det(A).

    Theory:
    1. For each element (i,j), compute minor M_ij = det(A with row i, col j removed).
    2. Cofactor C_ij = (-1)^{i+j} · M_ij.
    3. Adjoint Adj(A) = C^T (transpose of cofactor matrix).
    4. A^{-1} = Adj(A) / det(A).

    Application condition: A is square and det(A) != 0.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Ma trận phụ hợp (Adjoint)."
            f" Điều kiện áp dụng: A vuông {n}×{n}, det(A) ≠ 0."
            f" det(A) = {round(det, 7)} ≠ 0 → thỏa mãn."
            f" Công thức: A⁻¹ = Adj(A) / det(A) = C^T / det(A),"
            f" với C là ma trận cofactor: C_{{ij}} = (-1)^{{i+j}}·M_{{ij}}, M_{{ij}} = det(A bỏ hàng i cột j)."
        ),
    })

    # Step 1: Input matrix
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Step 2: Compute cofactor matrix with details for first few elements
    cof = np.zeros((n, n), dtype=float)
    cof_details: list[str] = []

    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(mat, i, 0), j, 1)
            minor_det = float(np.linalg.det(minor))
            sign = (-1) ** (i + j)
            cof[i, j] = sign * minor_det

            # Detail for first 2 rows or small matrices
            if n <= 3 or (i < 2 and j < 2):
                sign_str = "+" if sign > 0 else "-"
                cof_details.append(
                    f"C_{{{i+1},{j+1}}} = {sign_str}det(M_{{{i+1},{j+1}}}) = {sign_str}·{round(minor_det, 7)} = {round(float(cof[i, j]), 7)}"
                )

    steps.append({
        "step": 2,
        "description": (
            f"Bước 2: Tính ma trận cofactor C."
            f" Với mỗi (i,j): xóa hàng i, cột j của A → minor M_ij → det(M_ij) → C_ij = (-1)^{i+j}·det(M_ij)."
            + (" " + "; ".join(cof_details) if cof_details else "")
        ),
        "matrix": [[round(float(cof[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 3: Adjoint = C^T
    adj = cof.T
    steps.append({
        "step": 3,
        "description": (
            f"Bước 3: Ma trận phụ hợp Adj(A) = C^T (chuyển vị của ma trận cofactor)."
            f" Đảo hàng thành cột: Adj(A)_ij = C_ji."
        ),
        "matrix": [[round(float(adj[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 4: A^{-1} = Adj(A) / det(A)
    inv = adj / det
    steps.append({
        "step": 4,
        "description": (
            f"Bước 4: A⁻¹ = Adj(A) / det(A) = Adj(A) / {round(det, 7)}."
            f" Chia từng phần tử của Adj(A) cho det(A)."
        ),
        "inverse": [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)],
    })

    execution_time = time.time() - start_time
    return _build_result(mat, inv, n, "Adjoint", det, rank_val, steps, execution_time)


# =============================================================================
# 3. Cholesky Decomposition Inverse
# =============================================================================

def matrix_inverse_cholesky(A: List[List[float]]) -> dict:
    """Compute the inverse using Cholesky Decomposition for SPD matrices.

    Theory:
    1. A = L·L^T (Cholesky decomposition) if A is symmetric positive definite.
    2. Solve L·Y = I for Y = L^{-1} using forward substitution.
    3. A^{-1} = (L^{-1})^T · L^{-1} = Y^T·Y.

    Application condition: A must be symmetric (A = A^T) and positive definite.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det_val = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Cholesky."
            f" Điều kiện áp dụng: A đối xứng (A = A^T) và xác định dương."
            f" Nguyên lý: A = L·L^T → A⁻¹ = (L⁻¹)^T·L⁻¹."
        ),
    })

    # Step 1: Input
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det_val, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Check symmetry
    if not np.allclose(mat, mat.T, atol=1e-10):
        return {"success": False,
                "message": "Phương pháp Cholesky yêu cầu ma trận đối xứng. Ma trận A không đối xứng.",
                "steps": steps, "execution_time": round(time.time() - start_time, 7)}

    steps.append({
        "step": 2,
        "description": "Bước 2: Kiểm tra tính đối xứng: A = A^T → thỏa mãn.",
    })

    # Cholesky decomposition
    try:
        L_np = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return {"success": False,
                "message": "Phương pháp Cholesky yêu cầu ma trận xác định dương. Ma trận A không xác định dương.",
                "steps": steps, "execution_time": round(time.time() - start_time, 7)}

    L = [[float(L_np[i, j]) for j in range(n)] for i in range(n)]
    steps.append({
        "step": 3,
        "description": (
            f"Bước 3: Phân rã Cholesky A = L·L^T thành công (A xác định dương)."
            f" Ma trận L (tam giác dưới) được tính theo công thức:"
            f" L_ii = sqrt(a_ii - Σ_{{k<i}} L_ik²);  L_ij = (a_ij - Σ_{{k<j}} L_ik·L_jk) / L_jj (i>j)."
        ),
        "matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)],
    })

    # L^T
    LT = [[L[j][i] for j in range(n)] for i in range(n)]
    steps.append({
        "step": 4,
        "description": f"Bước 4: Ma trận L^T (chuyển vị của L, tam giác trên). A = L·L^T.",
        "matrix": [[round(LT[i][j], 7) for j in range(n)] for i in range(n)],
    })

    # Compute L^{-1} column by column via forward substitution
    L_inv = np.zeros((n, n), dtype=float)
    for col_idx in range(n):
        e = [1.0 if i == col_idx else 0.0 for i in range(n)]
        x = [0.0] * n
        for i in range(n):
            x[i] = e[i] - sum(L[i][j] * x[j] for j in range(i))
            if abs(L[i][i]) > 1e-15:
                x[i] /= L[i][i]
        for i in range(n):
            L_inv[i, col_idx] = x[i]

    L_inv_list = [[round(float(L_inv[i, j]), 7) for j in range(n)] for i in range(n)]
    steps.append({
        "step": 5,
        "description": (
            f"Bước 5: Tính L⁻¹ bằng cách giải L·Y = I qua forward substitution."
            f" Với mỗi cột c của I, giải L·y = e_c: y_i = (e_i - Σ_{{j<i}} L_ij·y_j) / L_ii."
        ),
        "matrix": L_inv_list,
    })

    # (L^{-1})^T
    L_inv_T = L_inv.T
    L_inv_T_list = [[round(float(L_inv_T[i, j]), 7) for j in range(n)] for i in range(n)]
    steps.append({
        "step": 6,
        "description": f"Bước 6: Ma trận (L⁻¹)^T (chuyển vị của L⁻¹).",
        "matrix": L_inv_T_list,
    })

    # A^{-1} = (L^{-1})^T · L^{-1}
    inv = np.dot(L_inv_T, L_inv)
    steps.append({
        "step": 7,
        "description": (
            f"Bước 7: A⁻¹ = (L⁻¹)^T · L⁻¹."
            f" Nhân ma trận (L⁻¹)^T với L⁻¹ cho kết quả cuối cùng."
        ),
        "inverse": [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)],
    })

    execution_time = time.time() - start_time
    return _build_result(mat, inv, n, "Cholesky", det_val, rank_val, steps, execution_time)


# =============================================================================
# 4. Bordering Inverse (Phương pháp viền quanh / Frobenius-Schur)
# =============================================================================

def matrix_inverse_bordering(A: List[List[float]]) -> dict:
    """Compute the inverse using the Bordering method (Frobenius-Schur complement).

    Theory:
    Start from 1x1 submatrix: A_1^{-1} = [1/a_11].
    At each step k, border the current inverse with one row and one column:
    
        A_{k+1} = [A_k     u ]
                  [v^T     a ]
    
    Schur complement: alpha = a - v^T·A_k^{-1}·u
    
        A_{k+1}^{-1} = [A_k^{-1} + (1/alpha)(A_k^{-1}u)(v^T A_k^{-1})   -(1/alpha)A_k^{-1}u ]
                       [-(1/alpha)v^T A_k^{-1}                             1/alpha             ]

    Application condition: A square, all leading principal minors non-zero.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Viền quanh (Bordering / Frobenius-Schur)."
            f" Điều kiện áp dụng: A vuông {n}×{n}, các ma trận con chính khả nghịch (các phần tử dẫn đầu ≠ 0)."
            f" det(A) = {round(det, 7)} ≠ 0 → thỏa mãn."
            f" Nguyên lý: Xây dựng dần A⁻¹ từ ma trận con 1×1, mỗi bước 'viền' thêm 1 hàng + 1 cột."
        ),
    })

    # Step 1: Input
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Check A[0][0]
    if abs(A[0][0]) < 1e-15:
        return {"success": False,
                "message": "Không thể áp dụng viền quanh: phần tử A[1][1] = 0.",
                "steps": steps, "execution_time": round(time.time() - start_time, 7)}

    A_inv = np.array([[1.0 / A[0][0]]], dtype=float)
    steps.append({
        "step": 2,
        "description": (
            f"Bước 2: Ma trận con A₁ = [{round(float(mat[0,0]), 7)}] (1×1)."
            f" A₁⁻¹ = [1/{round(float(mat[0,0]), 7)}] = [{round(float(A_inv[0, 0]), 7)}]."
        ),
        "matrix": [[round(float(A_inv[0, 0]), 7)]],
    })

    step_counter = 3

    for k in range(1, n):
        # Extract bordering vectors
        u = mat[:k, k:k+1].copy()  # shape (k, 1) — column vector, upper-right block
        vT = mat[k:k+1, :k].copy()  # shape (1, k) — row vector, lower-left block
        a_kk = mat[k, k]

        u_list = [round(float(u[i, 0]), 7) for i in range(k)]
        vT_list = [round(float(vT[0, j]), 7) for j in range(k)]

        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Viền thêm hàng {k+1} và cột {k+1}."
                f" u (cột phải trên) = [{', '.join(str(x) for x in u_list)}]^T;"
                f" v^T (hàng trái dưới) = [{', '.join(str(x) for x in vT_list)}];"
                f" a_{{{k+1},{k+1}}} = {round(float(a_kk), 7)}."
            ),
        })
        step_counter += 1

        # Compute intermediate products
        A_inv_u = A_inv @ u  # (k, 1)
        vT_A_inv = vT @ A_inv  # (1, k)

        # Schur complement
        alpha = a_kk - float((vT_A_inv @ u)[0, 0])

        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Tính Schur complement."
                f" α = a_{{{k+1},{k+1}}} - v^T·A_{k}⁻¹·u"
                f" = {round(float(a_kk), 7)} - {round(float((vT_A_inv @ u)[0, 0]), 7)}"
                f" = {round(float(alpha), 7)}."
            ),
        })
        step_counter += 1

        if abs(alpha) < 1e-15:
            return {"success": False,
                    "message": f"Không thể áp dụng viền quanh: Schur complement α ≈ 0 tại bước k={k+1}.",
                    "steps": steps, "execution_time": round(time.time() - start_time, 7)}

        # Build 4 blocks
        top_left = A_inv + (1.0 / alpha) * (A_inv_u @ vT_A_inv)
        top_right = -(1.0 / alpha) * A_inv_u
        bottom_left = -(1.0 / alpha) * vT_A_inv
        bottom_right = np.array([[1.0 / alpha]])

        steps.append({
            "step": step_counter,
            "description": (
                f"Bước {step_counter}: Công thức Frobenius-Schur cho ma trận nghịch đảo A_{{{k+1}}}⁻¹:"
                f" Khối trên-trái = A_k⁻¹ + (1/α)(A_k⁻¹u)(v^T A_k⁻¹);"
                f" Khối trên-phải = -(1/α)A_k⁻¹u;"
                f" Khối dưới-trái = -(1/α)v^T A_k⁻¹;"
                f" Khối dưới-phải = 1/α = {round(1.0/float(alpha), 7)}."
            ),
        })
        step_counter += 1

        # Assemble
        new_size = k + 1
        new_A_inv = np.zeros((new_size, new_size), dtype=float)
        new_A_inv[:k, :k] = top_left
        new_A_inv[:k, k:k+1] = top_right
        new_A_inv[k:k+1, :k] = bottom_left
        new_A_inv[k:k+1, k:k+1] = bottom_right

        A_inv = new_A_inv

        steps.append({
            "step": step_counter,
            "description": f"Bước {step_counter}: A_{{{k+1}}}⁻¹ ({new_size}×{new_size}) sau khi ghép 4 khối.",
            "matrix": [[round(float(A_inv[i, j]), 7) for j in range(new_size)] for i in range(new_size)],
        })
        step_counter += 1

    execution_time = time.time() - start_time
    inv = A_inv
    steps.append({
        "step": step_counter,
        "description": f"Bước {step_counter}: A⁻¹ hoàn tất sau {n-1} lần viền (kích thước {n}×{n}).",
        "inverse": [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)],
    })

    return _build_result(mat, inv, n, "Viền quanh (Bordering)", det, rank_val, steps, execution_time)


# =============================================================================
# 5. Jacobi Iterative Inverse
# =============================================================================

def matrix_inverse_jacobi(A: List[List[float]], epsilon: float = 1e-7,
                          max_iterations: int = 1000,
                          initial_guess: List[List[float]] | None = None) -> dict:
    """Compute the inverse using Jacobi iteration.

    Theory:
    Split A = D + (L+U) where D = diag(A), L+U = off-diagonal.
    
    Equation: A·X = I  =>  D·X = I - (L+U)·X
    => X = -D^{-1}(L+U)·X + D^{-1}
    
    Iteration: X^{(k+1)} = T·X^{(k)} + C
    where T = -D^{-1}(L+U) is the iteration matrix, C = D^{-1}.

    Convergence: requires rho(T) < 1, which holds if A is strictly diagonally dominant.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    eps = _effective_epsilon(epsilon)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Lặp Jacobi tìm ma trận nghịch đảo."
            f" Điều kiện áp dụng: A vuông {n}×{n}, det(A) ≠ 0, các phần tử đường chéo ≠ 0."
            f" Hội tụ nếu A chéo trội nghiêm ngặt (ρ(T) < 1)."
            f" ε = {eps}, max_iter = {max_iterations}."
        ),
    })

    # Step 1: Input
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Build D and L+U
    D_inv = np.zeros((n, n), dtype=float)
    LU = np.zeros((n, n), dtype=float)
    D_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        if abs(mat[i, i]) < 1e-15:
            return {"success": False,
                    "message": f"Không thể áp dụng Jacobi: phần tử đường chéo A[{i+1}][{i+1}] = 0.",
                    "steps": steps, "execution_time": round(time.time() - start_time, 7)}
        D_mat[i, i] = mat[i, i]
        D_inv[i, i] = 1.0 / mat[i, i]
        for j in range(n):
            if i != j:
                LU[i, j] = mat[i, j]

    # Step 2: Show D
    steps.append({
        "step": 2,
        "description": (
            f"Bước 2: Tách A = D + (L+U). D là ma trận đường chéo của A."
        ),
        "matrix": [[round(float(D_mat[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 3: Show L+U
    steps.append({
        "step": 3,
        "description": (
            f"Bước 3: Ma trận L+U (phần ngoài đường chéo của A)."
        ),
        "matrix": [[round(float(LU[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 4: Show D^{-1}
    steps.append({
        "step": 4,
        "description": (
            f"Bước 4: D⁻¹ = nghịch đảo của D (1/a_ii trên đường chéo)."
            f" Đây cũng là giá trị khởi tạo X⁽⁰⁾."
        ),
        "matrix": [[round(float(D_inv[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 5: Iteration matrix T and convergence check
    T = -D_inv @ LU
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    converges = spectral_radius < 1.0

    # Check diagonal dominance
    row_dd, col_dd = _is_diagonally_dominant_strict(mat, n)

    steps.append({
        "step": 5,
        "description": (
            f"Bước 5: Ma trận lặp T = -D⁻¹(L+U)."
            f" Bán kính phổ ρ(T) = {round(float(spectral_radius), 7)}"
            f" {'<' if converges else '≥'} 1 → {'HỘI TỤ' if converges else 'CÓ THỂ KHÔNG HỘI TỤ'}."
            f" Chéo trội hàng: {'Có' if row_dd else 'Không'}; Chéo trội cột: {'Có' if col_dd else 'Không'}."
            f" Công thức lặp: X⁽ᵏ⁺¹⁾ = T·X⁽ᵏ⁾ + D⁻¹ = -D⁻¹(L+U)·X⁽ᵏ⁾ + D⁻¹."
        ),
        "matrix": [[round(float(T[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Initial guess
    if initial_guess is not None:
        X = np.array(initial_guess, dtype=float)
    else:
        X = D_inv.copy()

    # Step 6: Iteration process
    iteration = 0
    iterations_data: list[dict] = []
    C = D_inv.copy()

    # Show first iteration formula explicitly
    steps.append({
        "step": 6,
        "description": (
            f"Bước 6: Bắt đầu lặp. Công thức: X⁽ᵏ⁺¹⁾ = T·X⁽ᵏ⁾ + D⁻¹."
            f" Ma trận hằng C = D⁻¹. Sai số đo bằng ||A·X⁽ᵏ⁾ - I||∞."
        ),
    })

    while iteration < max_iterations:
        X_new = T @ X + C
        iteration += 1

        residual = mat @ X_new - np.eye(n)
        error_val = float(np.linalg.norm(residual, ord=np.inf))

        iterations_data.append({
            "iteration": iteration,
            "x_matrix": [[round(float(X_new[i, j]), 7) for j in range(n)] for i in range(n)],
            "error": round(error_val, 10),
        })

        X = X_new

        if error_val < eps:
            break

    execution_time = time.time() - start_time

    if iteration >= max_iterations:
        steps.append({
            "step": 7,
            "description": (
                f"Bước 7: Kết thúc — không hội tụ sau {max_iterations} vòng lặp."
                f" Sai số cuối = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} > ε = {eps}."
                f" Gợi ý: Kiểm tra lại điều kiện chéo trội hoặc tăng max_iterations/giảm ε."
            ),
        })
    else:
        steps.append({
            "step": 7,
            "description": (
                f"Bước 7: Hội tụ sau {iteration} vòng lặp."
                f" ||A·X⁽{iteration}⁾ - I||∞ = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} < ε = {eps}."
            ),
        })

    return _build_result(mat, X, n,
                         f"Lặp Jacobi ({iteration} vòng lặp)" if iteration < max_iterations else f"Lặp Jacobi (chưa hội tụ sau {max_iterations} vòng)",
                         det, rank_val, steps, execution_time,
                         extra_iter_data={
                             "iterations_count": iteration,
                             "final_error": round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10),
                             "spectral_radius": round(float(spectral_radius), 7),
                             "converges": converges,
                             "iterations": iterations_data,
                         })


# =============================================================================
# 6. Gauss-Seidel Iterative Inverse
# =============================================================================

def matrix_inverse_gauss_seidel(A: List[List[float]], epsilon: float = 1e-7,
                                max_iterations: int = 1000,
                                initial_guess: List[List[float]] | None = None) -> dict:
    """Compute the inverse using Gauss-Seidel iteration.

    Theory:
    Solve A·X = I column by column. For column c:
        x_i^{(k+1)} = (delta_{ic} - sum_{j<i} a_ij·x_j^{(k+1)} - sum_{j>i} a_ij·x_j^{(k)}) / a_ii
    
    Unlike Jacobi, GS uses the most recently computed values (x_j^{(k+1)} for j < i),
    leading to faster convergence.

    Application condition: A square, det(A) != 0, diagonal entries non-zero.
    Converges if A is strictly diagonally dominant or symmetric positive definite.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    eps = _effective_epsilon(epsilon)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Lặp Gauss-Seidel tìm ma trận nghịch đảo."
            f" Điều kiện áp dụng: A vuông {n}×{n}, det(A) ≠ 0, các phần tử đường chéo ≠ 0."
            f" Hội tụ nếu A chéo trội nghiêm ngặt hoặc đối xứng xác định dương."
            f" ε = {eps}, max_iter = {max_iterations}."
            f" Khác với Jacobi: GS cập nhật TUẦN TỰ, dùng giá trị mới nhất → hội tụ nhanh hơn."
        ),
    })

    # Step 1: Input
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Check diagonal
    for i in range(n):
        if abs(mat[i, i]) < 1e-15:
            return {"success": False,
                    "message": f"Không thể áp dụng Gauss-Seidel: phần tử đường chéo A[{i+1}][{i+1}] = 0.",
                    "steps": steps, "execution_time": round(time.time() - start_time, 7)}

    # Split into D, L, U
    D_mat = np.zeros((n, n), dtype=float)
    L_mat = np.zeros((n, n), dtype=float)
    U_mat = np.zeros((n, n), dtype=float)
    D_inv_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        D_mat[i, i] = mat[i, i]
        D_inv_mat[i, i] = 1.0 / mat[i, i]
        for j in range(n):
            if j < i:
                L_mat[i, j] = mat[i, j]
            elif j > i:
                U_mat[i, j] = mat[i, j]

    # Step 2: Show D, L, U
    steps.append({
        "step": 2,
        "description": f"Bước 2: Tách A = D + L + U với D = đường chéo, L = tam giác dưới (j < i), U = tam giác trên (j > i).",
        "d_matrix": [[round(float(D_mat[i, j]), 7) for j in range(n)] for i in range(n)],
        "l_matrix": [[round(float(L_mat[i, j]), 7) for j in range(n)] for i in range(n)],
        "u_matrix": [[round(float(U_mat[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Check convergence
    row_dd, col_dd = _is_diagonally_dominant_strict(mat, n)
    is_sym = bool(np.allclose(mat, mat.T, atol=1e-10))
    try:
        np.linalg.cholesky(mat)
        is_spd = True
    except np.linalg.LinAlgError:
        is_spd = False

    steps.append({
        "step": 3,
        "description": (
            f"Bước 3: Kiểm tra điều kiện hội tụ:"
            f" Chéo trội hàng nghiêm ngặt: {'Có' if row_dd else 'Không'};"
            f" Đối xứng: {'Có' if is_sym else 'Không'};"
            f" Xác định dương: {'Có' if is_spd else 'Không'}."
            f" {'→ Đảm bảo hội tụ.' if (row_dd or is_spd) else '→ Hội tụ không được đảm bảo.'}"
        ),
    })

    # Initial guess
    if initial_guess is not None:
        X = np.array(initial_guess, dtype=float)
    else:
        X = np.zeros((n, n), dtype=float)
        for i in range(n):
            X[i, i] = 1.0 / mat[i, i]

    steps.append({
        "step": 4,
        "description": f"Bước 4: Khởi tạo X⁽⁰⁾ = D⁻¹. Công thức lặp: x_i^(k+1) = (δ_ic - Σ_{j<i} a_ij·x_j^(k+1) - Σ_{j>i} a_ij·x_j^(k)) / a_ii, cho mỗi cột c.",
        "matrix": [[round(float(X[i, j]), 7) for j in range(n)] for i in range(n)],
    })

    # Step 5: Begin iteration
    steps.append({
        "step": 5,
        "description": (
            f"Bước 5: Bắt đầu lặp Gauss-Seidel."
            f" Ở mỗi vòng lặp, lần lượt tính từng cột của X⁻¹."
            f" Trong mỗi cột, tính từ hàng 1→{n}, dùng ngay giá trị vừa tính của các hàng trên (j < i)."
            f" Sai số: ||A·X⁽ᵏ⁾ - I||∞."
        ),
    })

    iteration = 0
    iterations_data: list[dict] = []

    while iteration < max_iterations:
        X_new = X.copy()
        iteration += 1

        for c in range(n):
            for i in range(n):
                s = 0.0
                for j in range(n):
                    if j != i:
                        if j < i:
                            s += mat[i, j] * X_new[j, c]
                        else:
                            s += mat[i, j] * X[j, c]
                rhs = (1.0 if i == c else 0.0) - s
                X_new[i, c] = rhs / mat[i, i]

        residual = mat @ X_new - np.eye(n)
        error_val = float(np.linalg.norm(residual, ord=np.inf))

        iterations_data.append({
            "iteration": iteration,
            "x_matrix": [[round(float(X_new[i, j]), 7) for j in range(n)] for i in range(n)],
            "error": round(error_val, 10),
        })

        X = X_new

        if error_val < eps:
            break

    execution_time = time.time() - start_time

    if iteration >= max_iterations:
        steps.append({
            "step": 6,
            "description": (
                f"Bước 6: Kết thúc — không hội tụ sau {max_iterations} vòng lặp."
                f" Sai số cuối = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} > ε = {eps}."
            ),
        })
    else:
        steps.append({
            "step": 6,
            "description": (
                f"Bước 6: Hội tụ sau {iteration} vòng lặp."
                f" ||A·X⁽{iteration}⁾ - I||∞ = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} < ε = {eps}."
                f" So với Jacobi, Gauss-Seidel thường hội tụ nhanh hơn do dùng giá trị mới nhất."
            ),
        })

    return _build_result(mat, X, n,
                         f"Lặp Gauss-Seidel ({iteration} vòng lặp)" if iteration < max_iterations else f"Lặp Gauss-Seidel (chưa hội tụ sau {max_iterations} vòng)",
                         det, rank_val, steps, execution_time,
                         extra_iter_data={
                             "iterations_count": iteration,
                             "final_error": round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10),
                             "iterations": iterations_data,
                         })


# =============================================================================
# 7. Newton-Schulz Iterative Inverse
# =============================================================================

def matrix_inverse_newton(A: List[List[float]], epsilon: float = 1e-7,
                          max_iterations: int = 100,
                          initial_guess: List[List[float]] | None = None) -> dict:
    """Compute the inverse using Newton-Schulz iteration.

    Theory:
    X^{(k+1)} = X^{(k)}·(2I - A·X^{(k)})
    
    This is Newton's method applied to f(X) = X^{-1} - A = 0.
    Quadratic convergence: error squares at each step when close enough.
    
    Initial guess: X^{(0)} = alpha·A^T where alpha = 1 / (||A||_1 · ||A||_inf).
    This ensures ||I - A·X^{(0)}|| < 1 for convergence.

    Application condition: A square, det(A) != 0.
    """
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    eps = _effective_epsilon(epsilon)

    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": (
            f"Phương pháp Newton-Schulz (Newton lặp tìm ma trận nghịch đảo)."
            f" Điều kiện áp dụng: A vuông {n}×{n}, det(A) ≠ 0."
            f" ε = {eps}, max_iter = {max_iterations}."
            f" Nguyên lý: Áp dụng phương pháp Newton cho f(X) = X⁻¹ - A = 0."
            f" Công thức: X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾·(2I - A·X⁽ᵏ⁾). Hội tụ bậc 2 (quadratic)."
        ),
    })

    # Step 1: Input
    steps.append({
        "step": 1,
        "description": f"Bước 1: Ma trận đầu vào A ({n}×{n}). det(A) = {round(det, 7)}.",
        "matrix": [[round(v, 7) for v in row] for row in A],
    })

    start_time = time.time()

    # Compute norms
    norm_1 = float(np.linalg.norm(mat, 1))
    norm_inf = float(np.linalg.norm(mat, np.inf))

    steps.append({
        "step": 2,
        "description": (
            f"Bước 2: Tính chuẩn của A."
            f" ||A||₁ = max_j Σ_i |a_ij| = {round(norm_1, 7)}"
            f" (tổng cột lớn nhất);"
            f" ||A||∞ = max_i Σ_j |a_ij| = {round(norm_inf, 7)}"
            f" (tổng hàng lớn nhất)."
        ),
    })

    # Initial guess
    if initial_guess is not None:
        X = np.array(initial_guess, dtype=float)
        steps.append({
            "step": 3,
            "description": "Bước 3: X⁽⁰⁾ = initial guess (do người dùng cung cấp).",
        })
    else:
        alpha = 1.0 / (norm_1 * norm_inf)
        X = alpha * mat.T
        steps.append({
            "step": 3,
            "description": (
                f"Bước 3: Khởi tạo X⁽⁰⁾ = α·A^T."
                f" α = 1 / (||A||₁ · ||A||∞) = 1 / ({round(norm_1, 7)} · {round(norm_inf, 7)}) = {round(alpha, 10)}."
                f" Chọn α này đảm bảo ||I - A·X⁽⁰⁾|| < 1 → điều kiện hội tụ ban đầu."
            ),
            "matrix": [[round(float(X[i, j]), 7) for j in range(n)] for i in range(n)],
        })

    # Check initial residual
    initial_residual = float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf))
    steps.append({
        "step": 4,
        "description": (
            f"Bước 4: Kiểm tra điều kiện ban đầu."
            f" ||I - A·X⁽⁰⁾||∞ = {round(initial_residual, 7)}"
            f" {'< 1 → thỏa mãn điều kiện hội tụ.' if initial_residual < 1 else '≥ 1 → có thể không hội tụ.'}"
        ),
    })

    # Step 5: Iteration
    steps.append({
        "step": 5,
        "description": (
            f"Bước 5: Bắt đầu lặp Newton-Schulz."
            f" Công thức: X⁽ᵏ⁺¹⁾ = X⁽ᵏ⁾·(2I - A·X⁽ᵏ⁾)."
            f" Mỗi vòng lặp cần 2 phép nhân ma trận: A·X⁽ᵏ⁾ và X⁽ᵏ⁾·(2I - A·X⁽ᵏ⁾)."
            f" Hội tụ bậc 2: sai số giảm theo bình phương sau mỗi vòng (khi đủ gần)."
        ),
    })

    iteration = 0
    iterations_data: list[dict] = []

    while iteration < max_iterations:
        AX = mat @ X
        X_new = X @ (2.0 * np.eye(n) - AX)
        iteration += 1

        residual = mat @ X_new - np.eye(n)
        error_val = float(np.linalg.norm(residual, ord=np.inf))

        iterations_data.append({
            "iteration": iteration,
            "x_matrix": [[round(float(X_new[i, j]), 7) for j in range(n)] for i in range(n)],
            "error": round(error_val, 10),
        })

        if error_val > 1e10 or not np.all(np.isfinite(X_new)):
            return {"success": False,
                    "message": f"Newton-Schulz phân kỳ tại vòng lặp {iteration}.",
                    "steps": steps, "execution_time": round(time.time() - start_time, 7),
                    "iterations": iterations_data}

        X = X_new

        if error_val < eps:
            break

    execution_time = time.time() - start_time

    if iteration >= max_iterations:
        steps.append({
            "step": 6,
            "description": (
                f"Bước 6: Kết thúc — không hội tụ sau {max_iterations} vòng lặp."
                f" Sai số cuối = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} > ε = {eps}."
            ),
        })
    else:
        steps.append({
            "step": 6,
            "description": (
                f"Bước 6: Hội tụ sau {iteration} vòng lặp (bậc 2!)."
                f" ||A·X⁽{iteration}⁾ - I||∞ = {round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10)} < ε = {eps}."
                f" Do hội tụ bậc 2, số vòng lặp thường rất ít (≤ 10-15 với ma trận điều kiện tốt)."
            ),
        })

    return _build_result(mat, X, n,
                         f"Newton-Schulz ({iteration} vòng lặp)" if iteration < max_iterations else f"Newton-Schulz (chưa hội tụ sau {max_iterations} vòng)",
                         det, rank_val, steps, execution_time,
                         extra_iter_data={
                             "iterations_count": iteration,
                             "final_error": round(float(np.linalg.norm(mat @ X - np.eye(n), ord=np.inf)), 10),
                             "iterations": iterations_data,
                         })