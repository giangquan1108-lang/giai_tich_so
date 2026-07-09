"""Matrix Inverse algorithms: Gauss-Jordan, Adjoint, LU, Cholesky, Pseudoinverse SVD."""
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


# =============================================================================
# 1. Gauss-Jordan Inverse
# =============================================================================

def matrix_inverse_gauss_jordan(A: List[List[float]]) -> dict:
    """Compute the inverse of a square matrix using Gauss-Jordan elimination."""
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)

    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    steps: list[dict] = []
    steps.append({
        "step": 0,
        "description": f"Augmented matrix [A|I] ({n}×{2*n})",
        "matrix": [[round(v, 6) for v in row] for row in aug],
    })

    start_time = time.time()
    col = 0
    for k in range(n):
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
            steps.append({"step": len(steps), "description": f"Swap R{k+1} <-> R{max_row+1}",
                          "matrix": [[round(v, 6) for v in row] for row in aug]})

        pivot = aug[k][col]
        for j in range(2 * n):
            aug[k][j] /= pivot
        steps.append({"step": len(steps), "description": f"R{k+1} = R{k+1} / ({round(pivot, 6)})",
                      "matrix": [[round(v, 6) for v in row] for row in aug]})

        for i in range(n):
            if i == k or abs(aug[i][col]) < 1e-15:
                continue
            factor = aug[i][col]
            for j in range(2 * n):
                aug[i][j] -= factor * aug[k][j]
            steps.append({"step": len(steps),
                          "description": f"R{i+1} = R{i+1} - ({round(factor, 6)}) * R{k+1}",
                          "matrix": [[round(v, 6) for v in row] for row in aug]})
        col += 1

    execution_time = time.time() - start_time
    inv = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    inv_rounded = [[round(v, 10) for v in row] for row in inv]

    verify = np.dot(mat, np.array(inv, dtype=float))
    verify_rounded = [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)]
    is_accurate = bool(np.allclose(verify, np.eye(n), atol=1e-8))

    steps.append({"step": len(steps),
                  "description": f"[I|A⁻¹] — Inverse computed. det(A) = {round(det, 6)}, rank(A) = {rank_val}",
                  "inverse": inv_rounded})

    latex_rows = [" & ".join(str(v) for v in row) for row in inv_rounded]
    inv_latex = r"A^{-1} = \begin{bmatrix}" + " \\\\ ".join(latex_rows) + r"\end{bmatrix}"

    return {
        "success": True,
        "message": f"Ma trận nghịch đảo tính thành công bằng Gauss-Jordan. det(A) = {round(det, 6)}",
        "determinant": round(det, 10),
        "rank": rank_val,
        "inverse": inv_rounded,
        "inverse_latex": inv_latex,
        "verification": verify_rounded,
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(execution_time, 6),
    }


# =============================================================================
# 2. Adjoint Inverse
# =============================================================================

def matrix_inverse_adjoint(A: List[List[float]]) -> dict:
    """Compute the inverse using the Adjoint (Adjugate) method: A^{-1} = Adj(A) / det(A)."""
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Input matrix A ({n}×{n}), det(A) = {round(det, 6)}",
                  "matrix": [[round(v, 6) for v in row] for row in A]})

    start_time = time.time()
    cof = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(mat, i, 0), j, 1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

    steps.append({"step": 1, "description": "Ma trận cofactor C (C_{ij} = (-1)^{i+j} · det(M_{ij}))",
                  "matrix": [[round(float(cof[i, j]), 6) for j in range(n)] for i in range(n)]})

    adj = cof.T
    steps.append({"step": 2, "description": "Ma trận phụ hợp Adj(A) = C^T",
                  "matrix": [[round(float(adj[i, j]), 6) for j in range(n)] for i in range(n)]})

    inv = adj / det
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 3, "description": f"A⁻¹ = Adj(A) / det(A) = Adj(A) / {round(det, 6)}",
                  "inverse": inv_rounded})

    execution_time = time.time() - start_time
    verify = np.dot(mat, inv)
    verify_rounded = [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)]
    is_accurate = bool(np.allclose(verify, np.eye(n), atol=1e-8))

    latex_rows = [" & ".join(str(v) for v in row) for row in inv_rounded]
    inv_latex = r"A^{-1} = \begin{bmatrix}" + " \\\\ ".join(latex_rows) + r"\end{bmatrix}"

    return {
        "success": True,
        "message": f"Ma trận nghịch đảo tính thành công bằng Adjoint. det(A) = {round(det, 6)}",
        "determinant": round(det, 10),
        "rank": rank_val,
        "inverse": inv_rounded,
        "inverse_latex": inv_latex,
        "verification": verify_rounded,
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(execution_time, 6),
    }


# =============================================================================
# 3. LU Decomposition Inverse
# =============================================================================

def _lu_factorize(A: List[List[float]], n: int, steps: list):
    a = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):
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
            steps.append({"step": len(steps), "description": f"Swap rows {k+1} and {max_row+1}",
                          "matrix": [row[:] for row in a]})
        for j in range(k, n):
            U[k][j] = a[k][j] - sum(L[k][p] * U[p][j] for p in range(k))
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = a[i][k] - sum(L[i][p] * U[p][k] for p in range(k))
            if abs(U[k][k]) > 1e-15:
                L[i][k] /= U[k][k]
    return L, U, P_mat


def _solve_lu_single(L: List[List[float]], U: List[List[float]],
                     P_mat: List[List[float]], b_col: List[float], n: int) -> List[float]:
    Pb = [sum(P_mat[i][j] * b_col[j] for j in range(n)) for i in range(n)]
    Y = [0.0] * n
    for i in range(n):
        Y[i] = Pb[i] - sum(L[i][j] * Y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = Y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
        if abs(U[i][i]) > 1e-15:
            x[i] /= U[i][i]
    return x


def matrix_inverse_lu(A: List[List[float]]) -> dict:
    """Compute the inverse using LU Decomposition: solve AX = I for each column."""
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Input matrix A ({n}×{n}), det(A) = {round(det, 6)}",
                  "matrix": [[round(v, 6) for v in row] for row in A]})

    start_time = time.time()
    L, U, P_mat = _lu_factorize(A, n, steps)
    steps.append({"step": 1, "description": "Ma trận L (lower triangular)",
                  "matrix": [[round(L[i][j], 6) for j in range(n)] for i in range(n)]})
    steps.append({"step": 2, "description": "Ma trận U (upper triangular)",
                  "matrix": [[round(U[i][j], 6) for j in range(n)] for i in range(n)]})

    inv = np.zeros((n, n), dtype=float)
    for col_idx in range(n):
        x = _solve_lu_single(L, U, P_mat, [1.0 if i == col_idx else 0.0 for i in range(n)], n)
        for i in range(n):
            inv[i, col_idx] = x[i]

    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 3, "description": "Giải AX = I qua LU → A⁻¹", "inverse": inv_rounded})

    execution_time = time.time() - start_time
    verify = np.dot(mat, np.array(inv, dtype=float))
    verify_rounded = [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)]
    is_accurate = bool(np.allclose(verify, np.eye(n), atol=1e-8))

    latex_rows = [" & ".join(str(v) for v in row) for row in inv_rounded]
    inv_latex = r"A^{-1} = \begin{bmatrix}" + " \\\\ ".join(latex_rows) + r"\end{bmatrix}"

    return {
        "success": True,
        "message": f"Ma trận nghịch đảo tính thành công bằng LU Decomposition. det(A) = {round(det, 6)}",
        "determinant": round(det, 10),
        "rank": rank_val,
        "inverse": inv_rounded,
        "inverse_latex": inv_latex,
        "verification": verify_rounded,
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(execution_time, 6),
    }


# =============================================================================
# 4. Cholesky Decomposition Inverse
# =============================================================================

def matrix_inverse_cholesky(A: List[List[float]]) -> dict:
    """Compute the inverse using Cholesky Decomposition for SPD matrices."""
    err = _validate_matrix_inverse(A)
    if err:
        return err

    n = len(A)
    mat = np.array(A, dtype=float)
    det_val = float(np.linalg.det(mat))
    rank_val = _compute_rank(A)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Input matrix A ({n}×{n})",
                  "matrix": [[round(v, 6) for v in row] for row in A]})

    start_time = time.time()

    if not np.allclose(mat, mat.T, atol=1e-10):
        return {"success": False,
                "message": "Phương pháp Cholesky yêu cầu ma trận đối xứng. Ma trận A không đối xứng.",
                "steps": [], "execution_time": round(time.time() - start_time, 6)}

    try:
        L_np = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return {"success": False,
                "message": "Phương pháp Cholesky yêu cầu ma trận xác định dương. Ma trận A không xác định dương.",
                "steps": [], "execution_time": round(time.time() - start_time, 6)}

    L = [[float(L_np[i, j]) for j in range(n)] for i in range(n)]
    steps.append({"step": 1, "description": "Ma trận L (Cholesky): A = L·L^T",
                  "matrix": [[round(L[i][j], 6) for j in range(n)] for i in range(n)]})

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

    inv = np.dot(L_inv.T, L_inv)
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 2, "description": "A⁻¹ = (L⁻¹)^T · L⁻¹", "inverse": inv_rounded})

    execution_time = time.time() - start_time
    verify = np.dot(mat, inv)
    verify_rounded = [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)]
    is_accurate = bool(np.allclose(verify, np.eye(n), atol=1e-8))

    latex_rows = [" & ".join(str(v) for v in row) for row in inv_rounded]
    inv_latex = r"A^{-1} = \begin{bmatrix}" + " \\\\ ".join(latex_rows) + r"\end{bmatrix}"

    return {
        "success": True,
        "message": f"Ma trận nghịch đảo tính thành công bằng Cholesky Decomposition. det(A) = {round(det_val, 6)}",
        "determinant": round(det_val, 10),
        "rank": rank_val,
        "inverse": inv_rounded,
        "inverse_latex": inv_latex,
        "verification": verify_rounded,
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(execution_time, 6),
    }


# =============================================================================
# 5. Pseudoinverse SVD (Moore-Penrose)
# =============================================================================

def pseudoinverse_svd(A: List[List[float]]) -> dict:
    """Compute the Moore-Penrose pseudoinverse A⁺ using SVD.
    
    A⁺ = V · Σ⁺ · U^T where Σ⁺ is the reciprocal of non-zero singular values.
    Works for ANY matrix (not just square invertible).
    """
    if not A or not A[0]:
        return {"success": False, "message": "Ma trận A không được rỗng."}

    m = len(A)
    n = len(A[0])
    mat = np.array(A, dtype=float)

    start_time = time.time()
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Input matrix A ({m}×{n})",
                  "matrix": [[round(v, 6) for v in row] for row in A]})

    # SVD decomposition
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    rank = int(np.sum(s > 1e-12 * max(m, n) * max(s[0], 0)))

    steps.append({"step": 1,
                  "description": f"SVD: A = U·Σ·V^T. rank(A) = {rank}",
                  "u_matrix": [[round(float(U[i, j]), 6) for j in range(U.shape[1])] for i in range(m)],
                  "singular_values": [round(float(sv), 10) for sv in s],
                  "vt_matrix": [[round(float(Vt[i, j]), 6) for j in range(n)] for i in range(Vt.shape[0])]})

    # Compute Σ⁺ (reciprocal of non-zero singular values)
    s_inv = np.zeros((n, m))
    for i in range(rank):
        s_inv[i, i] = 1.0 / s[i]

    # Compute pseudoinverse: A⁺ = V^T^T · Σ⁺ · U^T = V · Σ⁺ · U^T
    pinv = Vt.T @ s_inv @ U.T
    pinv_rounded = [[round(float(pinv[i, j]), 10) for j in range(m)] for i in range(n)]

    execution_time = time.time() - start_time
    steps.append({"step": 2, "description": f"A⁺ = V·Σ⁺·U^T (Moore-Penrose pseudoinverse, {n}×{m})",
                  "inverse": pinv_rounded})

    # Verify: A·A⁺·A ≈ A and A⁺·A·A⁺ ≈ A⁺
    verify_1 = mat @ pinv @ mat
    verify_2 = pinv @ mat @ pinv
    is_accurate = bool(np.allclose(verify_1, mat, atol=1e-8) and np.allclose(verify_2, pinv, atol=1e-8))

    return {
        "success": True,
        "message": f"Pseudoinverse tính thành công bằng SVD. rank(A) = {rank}, condition number = {s[0] / s[rank-1] if rank > 0 else float('inf'):.4f}",
        "rank": rank,
        "singular_values": [round(float(sv), 10) for sv in s],
        "condition_number": round(float(s[0] / s[rank - 1]) if rank > 0 else float('inf'), 6),
        "inverse": pinv_rounded,
        "inverse_latex": "",  # Too large for display
        "verification": [[round(float(verify_1[i, j]), 10) for j in range(n)] for i in range(m)],
        "is_accurate": is_accurate,
        "steps": steps,
        "execution_time": round(execution_time, 6),
    }