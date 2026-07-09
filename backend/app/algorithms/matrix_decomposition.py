"""Matrix Decomposition algorithms: LU, Cholesky, QR, SVD, Schur."""
import sys
import numpy as np
from typing import List
import time

MACHINE_EPSILON = sys.float_info.epsilon


def _matrix_to_latex(mat: List[List[float]], precision: int = 4) -> str:
    if not mat:
        return r"\begin{bmatrix}\end{bmatrix}"
    rows = []
    for row in mat:
        cells = " & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row)
        rows.append(cells)
    return r"\begin{bmatrix}" + " \\\\ ".join(rows) + r"\end{bmatrix}"


def lu_decomposition(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": f"LU Decomposition yeu cau ma tran vuong (A hien tai: {m}x{n})."}
    a = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Ma tran A ({n}x{n}). Tinh LU voi partial pivoting."})
    start_time = time.time()
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
        for j in range(k, n):
            U[k][j] = a[k][j] - sum(L[k][p] * U[p][j] for p in range(k))
        L[k][k] = 1.0
        for i in range(k + 1, n):
            L[i][k] = a[i][k] - sum(L[i][p] * U[p][k] for p in range(k))
            if abs(U[k][k]) > 1e-15:
                L[i][k] /= U[k][k]
    execution_time = time.time() - start_time
    L_rounded = [[round(L[i][j], 10) for j in range(n)] for i in range(n)]
    U_rounded = [[round(U[i][j], 10) for j in range(n)] for i in range(n)]
    P_rounded = [[round(P_mat[i][j], 10) for j in range(n)] for i in range(n)]
    verify_pa = np.dot(np.array(P_mat), np.array(A))
    verify_lu = np.dot(np.array(L), np.array(U))
    is_accurate = bool(np.allclose(verify_pa, verify_lu, atol=1e-8))
    return {
        "success": True,
        "message": f"LU Decomposition thanh cong. A = P.L.U ({n}x{n})",
        "L": L_rounded, "U": U_rounded, "P": P_rounded,
        "L_latex": _matrix_to_latex(L_rounded),
        "U_latex": _matrix_to_latex(U_rounded),
        "P_latex": _matrix_to_latex(P_rounded),
        "verification": [[round(float(verify_pa[i, j]), 10) for j in range(n)] for i in range(n)] if not is_accurate else None,
        "is_accurate": is_accurate, "steps": steps, "execution_time": round(execution_time, 6),
    }


def cholesky_decomposition(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": f"Cholesky yeu cau ma tran vuong (A hien tai: {m}x{n})."}
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Ma tran A ({n}x{n}). Kiem tra doi xung va xac dinh duong."})
    start_time = time.time()
    if not np.allclose(mat, mat.T, atol=1e-10):
        return {"success": False, "message": "Ma tran khong doi xung. Cholesky yeu cau A = A^T."}
    try:
        L_np = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return {"success": False, "message": "Ma tran khong xac dinh duong. Cholesky yeu cau A > 0."}
    L = [[float(L_np[i, j]) for j in range(n)] for i in range(n)]
    execution_time = time.time() - start_time
    L_rounded = [[round(L[i][j], 10) for j in range(n)] for i in range(n)]
    return {
        "success": True,
        "message": f"Cholesky Decomposition thanh cong. A = L.L^T ({n}x{n})",
        "L": L_rounded, "L_latex": _matrix_to_latex(L_rounded),
        "is_accurate": True, "steps": steps, "execution_time": round(execution_time, 6),
    }


def qr_decomposition(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Ma tran A ({m}x{n}). Phan ra QR."})
    start_time = time.time()
    Q_np, R_np = np.linalg.qr(mat)
    execution_time = time.time() - start_time
    Q_rounded = [[round(float(Q_np[i, j]), 10) for j in range(n)] for i in range(m)]
    R_rounded = [[round(float(R_np[i, j]), 10) for j in range(n)] for i in range(n)]
    is_orthogonal = bool(np.allclose(Q_np.T @ Q_np, np.eye(n), atol=1e-8))
    is_accurate = bool(np.allclose(Q_np @ R_np, mat, atol=1e-8))
    return {
        "success": True, "message": f"QR Decomposition thanh cong. A = Q.R ({m}x{n})",
        "Q": Q_rounded, "R": R_rounded,
        "Q_latex": _matrix_to_latex(Q_rounded) if m <= 6 and n <= 6 else f"Q({m}x{n})",
        "R_latex": _matrix_to_latex(R_rounded),
        "is_orthogonal": is_orthogonal, "is_accurate": is_accurate,
        "steps": steps, "execution_time": round(execution_time, 6),
    }


def svd_decomposition(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Ma tran A ({m}x{n}). Phan ra SVD."})
    start_time = time.time()
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    execution_time = time.time() - start_time
    rank = int(np.sum(s > 1e-12 * max(m, n) * max(s[0], 0)))
    condition_number = float(s[0] / s[rank - 1]) if rank > 0 else float('inf')
    U_rounded = [[round(float(U[i, j]), 10) for j in range(U.shape[1])] for i in range(m)]
    Vt_rounded = [[round(float(Vt[i, j]), 10) for j in range(n)] for i in range(Vt.shape[0])]
    return {
        "success": True,
        "message": f"SVD Decomposition thanh cong. A = U.Sigma.V^T. rank={rank}, kappa(A)={condition_number:.4f}",
        "U": U_rounded, "Vt": Vt_rounded,
        "singular_values": [round(float(sv), 10) for sv in s],
        "rank": rank, "condition_number": round(condition_number, 6),
        "U_latex": _matrix_to_latex(U_rounded) if m <= 6 and U.shape[1] <= 6 else f"U({m}x{U.shape[1]})",
        "Vt_latex": _matrix_to_latex(Vt_rounded) if Vt.shape[0] <= 6 and n <= 6 else f"V^T({Vt.shape[0]}x{n})",
        "is_accurate": True, "steps": steps, "execution_time": round(execution_time, 6),
    }


def schur_decomposition(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    if m != n:
        return {"success": False, "message": f"Schur Decomposition yeu cau ma tran vuong (A hien tai: {m}x{n})."}
    mat = np.array(A, dtype=float)
    steps: list[dict] = []
    steps.append({"step": 0, "description": f"Ma tran A ({n}x{n}). Phan ra Schur."})
    start_time = time.time()
    try:
        from scipy.linalg import schur
        T, Q = schur(mat, output='complex')
    except ImportError:
        return {"success": False, "message": "Can cai dat scipy de dung Schur Decomposition."}
    execution_time = time.time() - start_time
    T_real = [[round(float(T[i, j].real), 10) for j in range(n)] for i in range(n)]
    Q_rounded = [[round(float(Q[i, j].real), 10) for j in range(n)] for i in range(n)]
    eigenvalues = [round(float(T[i, i].real), 10) for i in range(n)]
    verify = Q @ T @ Q.conj().T
    is_accurate = bool(np.allclose(verify, mat, atol=1e-8))
    return {
        "success": True, "message": f"Schur Decomposition thanh cong. A = Q.T.Q^H.",
        "T": T_real, "Q": Q_rounded, "eigenvalues": eigenvalues,
        "T_latex": _matrix_to_latex(T_real),
        "Q_latex": _matrix_to_latex(Q_rounded) if n <= 6 else f"Q({n}x{n})",
        "is_accurate": is_accurate, "steps": steps, "execution_time": round(execution_time, 6),
    }