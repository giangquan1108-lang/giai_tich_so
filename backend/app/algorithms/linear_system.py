"""Unified Linear System AX=B solvers: direct + iterative methods."""
import sys
from math import isfinite
import numpy as np
from typing import List, Tuple, Optional
import copy
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


def _analyze_system(A: List[List[float]], B: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    rank_A = _matrix_rank(A)
    max_rank_aug = rank_A
    inconsistent_columns = []
    for ci in range(p):
        b_col = [B[i][ci] for i in range(m)]
        aug_col = [list(A[i]) + [b_col[i]] for i in range(m)]
        rank_aug_col = _matrix_rank(aug_col)
        if rank_aug_col > max_rank_aug:
            max_rank_aug = rank_aug_col
        if rank_A < rank_aug_col:
            inconsistent_columns.append(ci + 1)
    rank_aug = max_rank_aug
    if inconsistent_columns:
        solution_type = "inconsistent"
        cols_str = ", ".join(str(c) for c in inconsistent_columns)
        message = (f"Vô nghiệm: rank(A)={rank_A} < rank([A|B_cột{{{cols_str}}}])={rank_aug}. "
                   f"Hệ không nhất quán tại cột B thứ {cols_str}.")
    elif rank_A == n:
        solution_type = "unique"
        message = f"Nghiệm duy nhất: rank(A)={rank_A} = rank([A|B])={rank_aug} = số ẩn={n}"
    else:
        solution_type = "infinite"
        message = f"Vô số nghiệm: rank(A)={rank_A} = rank([A|B])={rank_aug} < số ẩn={n}"
    return {"rank_A": rank_A, "rank_augmented": rank_aug, "n": n, "m": m, "p": p,
            "solution_type": solution_type, "message": message}


def _detect_matrix_properties(A: List[List[float]]) -> dict:
    m = len(A)
    n = len(A[0]) if A else 0
    mat = np.array(A, dtype=float)
    is_square = (m == n)
    is_symmetric = False
    is_positive_definite = False
    is_tridiagonal = False
    is_diagonally_dominant_strict = False
    is_diagonally_dominant_weak = False
    if is_square and m > 0:
        is_symmetric = bool(np.allclose(mat, mat.T, atol=1e-10))
        if is_symmetric:
            try:
                eigvals = np.linalg.eigvalsh(mat)
                is_positive_definite = bool(np.all(eigvals > 1e-12))
            except Exception:
                pass
        tridiag = True
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(mat[i, j]) > 1e-12:
                    tridiag = False
                    break
            if not tridiag:
                break
        is_tridiagonal = tridiag
        strict = True; weak = True
        for i in range(n):
            diag = abs(mat[i, i])
            off_sum = sum(abs(mat[i, j]) for j in range(n) if j != i)
            if diag < off_sum: strict = False
            if diag <= off_sum: weak = False
        is_diagonally_dominant_strict = strict
        is_diagonally_dominant_weak = weak
    recommendations = []
    if is_symmetric and is_positive_definite:
        recommendations.append("Ma trận đối xứng xác định dương → khuyến nghị Cholesky.")
    if is_diagonally_dominant_strict and is_square:
        recommendations.append("Ma trận chéo trội nghiêm ngặt → phương pháp lặp hội tụ nhanh.")
    if not is_square:
        recommendations.append("Ma trận không vuông → chỉ dùng direct methods (Gauss/Gauss-Jordan).")
    return {"is_square": is_square, "is_symmetric": is_symmetric, "is_positive_definite": is_positive_definite,
            "is_tridiagonal": is_tridiagonal, "is_diagonally_dominant_strict": is_diagonally_dominant_strict,
            "is_diagonally_dominant_weak": is_diagonally_dominant_weak, "recommendations": recommendations}


def _get_convergence_data(iterations: List[dict]) -> dict:
    if not iterations:
        return {"iterations": [], "errors": [], "x_values": {}}
    k_values = [it["k"] for it in iterations]
    error_values = [it.get("error", 0) for it in iterations]
    if iterations and "x" in iterations[0]:
        m = len(iterations[0]["x"])
        x_values = {f"x{i+1}": [it["x"][i] for it in iterations] for i in range(m)}
    else:
        x_values = {}
    return {"iterations": k_values, "errors": error_values, "x_values": x_values}


def _compute_rref_analysis(aug: List[List[float]], m: int, n: int, p: int = 1) -> dict:
    tol = 1e-14
    pivot_cols = []
    pivot_row_for_col = {}
    for i in range(m):
        for j in range(n):
            if abs(aug[i][j]) > tol and all(abs(aug[i][k]) < tol for k in range(j)):
                pivot_cols.append(j)
                pivot_row_for_col[j] = i
                break
    pivot_cols = sorted(set(pivot_cols))
    free_cols = [j for j in range(n) if j not in pivot_cols]
    free_variables = [f"x{j+1}" for j in free_cols]
    x_p_matrix = []
    for col_b in range(p):
        x_p = [0.0] * n
        for pivot_col in pivot_cols:
            pivot_row = pivot_row_for_col[pivot_col]
            x_p[pivot_col] = aug[pivot_row][n + col_b]
        x_p_matrix.append([round(val, 7) for val in x_p])
    x_p_transposed = [[x_p_matrix[ci][ri] for ci in range(p)] for ri in range(n)]
    basis_vectors = []
    for f in free_cols:
        v = [0.0] * n
        v[f] = 1.0
        for pivot_col in pivot_cols:
            pivot_row = pivot_row_for_col[pivot_col]
            v[pivot_col] = -aug[pivot_row][f]
        basis_vectors.append(v)
    n_free = len(free_cols)
    if n_free == 0:
        general_latex = ""
    else:
        parts = [r"x = x_p + " + " + ".join(f"c_{{{i+1}}} \\cdot v_{{{i+1}}}" for i in range(n_free))]
        parts.append(r"\text{với:}")
        if p == 1:
            x_p_first = [row[0] for row in x_p_transposed]
            parts.append(r"x_p = \begin{bmatrix}" + " \\\\ ".join(str(v) for v in x_p_first) + r"\end{bmatrix}")
        else:
            first_col = [row[0] for row in x_p_transposed]
            parts.append(r"x_p\text{(cột 1)} = \begin{bmatrix}" + " \\\\ ".join(str(v) for v in first_col) + r"\end{bmatrix}")
        for i in range(n_free):
            bv = [round(val, 7) for val in basis_vectors[i]]
            parts.append(f"v_{{{i+1}}} = \\begin{{bmatrix}}" + " \\\\ ".join(str(v) for v in bv) + r"\end{bmatrix}")
        general_latex = r" \\[6pt] ".join(parts)
    return {"pivot_columns": pivot_cols, "free_columns": free_cols, "free_variables": free_variables,
            "particular_solution": x_p_transposed, "basis_vectors": [list(bv) for bv in basis_vectors],
            "general_solution_latex": general_latex}


def _build_infinite_result(aug: List[List[float]], m: int, n: int,
                           analysis: dict, steps: List[dict],
                           execution_time: float,
                           B: Optional[List[List[float]]] = None) -> dict:
    tol = 1e-15
    p = analysis.get("p", 1)
    if B is not None:
        aug_full = [list(A_row) + [B[i][ci] for ci in range(p)] for i, A_row in
                     enumerate([aug[row][:n] for row in range(m)])]
    else:
        aug_full = [row[:] for row in aug]
    pivot_rows = []
    for i in range(m):
        for j in range(n):
            if abs(aug_full[i][j]) > tol:
                pivot_rows.append(i)
                break
    for i in pivot_rows:
        pivot_col = -1
        for j in range(n):
            if abs(aug_full[i][j]) > tol:
                pivot_col = j
                break
        if pivot_col >= 0:
            pv = aug_full[i][pivot_col]
            for j in range(n + p):
                aug_full[i][j] /= pv
    for i in pivot_rows:
        pivot_col = -1
        for j in range(n):
            if abs(aug_full[i][j]) > tol:
                pivot_col = j
                break
        if pivot_col < 0: continue
        for k in range(m):
            if k == i: continue
            if abs(aug_full[k][pivot_col]) > tol:
                factor = aug_full[k][pivot_col]
                for j in range(n + p):
                    aug_full[k][j] -= factor * aug_full[i][j]
    rref_data = _compute_rref_analysis(aug_full, m, n, p)
    steps.append({"step": len(steps),
                  "description": f"Reduced Row Echelon Form. rank(A)={analysis['rank_A']} < n={n} → vô số nghiệm",
                  "matrix": [row[:n+1] for row in aug_full]})
    rank_A_comp = _matrix_rank([aug_full[i][:n] for i in range(m)])
    rank_aug_comp = _matrix_rank([aug_full[i][:n+1] for i in range(m)])
    return {"success": True, "message": analysis["message"], "solution": None,
            "steps": steps, "pivot_info": None, "execution_time": round(execution_time, 7),
            "analysis": analysis, "solution_type": "infinite", "rank_A": rank_A_comp,
            "rank_augmented": rank_aug_comp, "free_variables": rref_data["free_variables"],
            "particular_solution": rref_data["particular_solution"], "basis_vectors": rref_data["basis_vectors"],
            "general_solution_latex": rref_data["general_solution_latex"],
            "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}


def _transpose_solution(x_cols, n, p):
    return [[x_cols[j][i] for j in range(p)] for i in range(n)]


def _matrix_to_latex(mat, precision=7, augmented=False):
    if not mat: return r"\begin{bmatrix}\end{bmatrix}"
    if augmented:
        n_a = len(mat[0]) - 1
        rows = [" & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row) for row in mat]
        return r"\left[\begin{array}{" + "c" * n_a + "|c}\n" + " \\\\ \n".join(rows) + r"\n\end{array}\right]"
    else:
        rows = [" & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row) for row in mat]
        return r"\begin{bmatrix}" + " \\\\ ".join(rows) + r"\end{bmatrix}"


def _solve_single_column_gaussian(A, b, n, m, steps, pivot_info, col_idx=0, p=1):
    aug = [list(A[i]) + [b[i]] for i in range(m)]
    min_dim = min(m, n)
    col = 0
    col_label = f" (cột B thứ {col_idx + 1})" if p > 1 else ""
    for k in range(min_dim):
        if col >= n: break
        max_row = k
        max_val = abs(aug[k][col]) if k < m else 0
        for i in range(k + 1, m):
            if abs(aug[i][col]) > max_val:
                max_val = abs(aug[i][col]); max_row = i
        if max_val < 1e-15: col += 1; continue
        pivot_info.append({"step": len(steps), "pivot_row": k + 1, "pivot_column": col + 1, "pivot_value": round(aug[k][col], 7)})
        pivot_val = round(aug[k][col], 4)
        latex_desc = rf"a_{{{k+1}{col+1}}} = {pivot_val}"
        row_ops_latex = []
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            row_ops_latex.append(rf"R_{{{k+1}}} \leftrightarrow R_{{{max_row+1}}}")
        elim_row_ops = []
        for i in range(k + 1, m):
            if abs(aug[i][col]) < 1e-15: continue
            factor = aug[i][col] / aug[k][col]
            for j in range(col, n + 1): aug[i][j] -= factor * aug[k][j]
            elim_row_ops.append({"target_row": i + 1, "factor": round(factor, 7), "pivot_row": k + 1})
            row_ops_latex.append(rf"R_{{{i+1}}} \leftarrow R_{{{i+1}}} - ({round(factor, 4)}) R_{{{k+1}}}")
        description = f"Chọn pivot hàng {k+1}, cột {col+1}" + col_label
        if elim_row_ops:
            description += "; " + "; ".join(f"R{op['target_row']}=R{op['target_row']}-({round(op['factor'], 4)})·R{op['pivot_row']}" for op in elim_row_ops)
        steps.append({"step": len(steps), "description": description,
                      "matrix": [[round(v, 7) for v in row] for row in aug],
                      "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True),
                      "row_operations": elim_row_ops if elim_row_ops else None,
                      "row_operations_latex": row_ops_latex,
                      "pivot_col": col + 1, "pivot_row": k + 1, "pivot_value": pivot_val, "phase": "elimination"})
        col += 1
    steps.append({"step": len(steps), "description": f"Ma trận tam giác trên sau khử Gauss{col_label}",
                  "matrix": [[round(v, 7) for v in row] for row in aug],
                  "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True), "phase": "upper_triangular"})
    x = [0.0] * n
    if n > 0:
        for i in range(min(n, m) - 1, -1, -1):
            if all(abs(aug[i][j]) < 1e-15 for j in range(n)):
                if abs(aug[i][n]) > 1e-12: return None
                x[i] = 0.0; continue
            x[i] = aug[i][n]
            for j in range(i + 1, n): x[i] -= aug[i][j] * x[j]
            if abs(aug[i][i]) > 1e-15: x[i] /= aug[i][i]
            else: x[i] = 0.0
    back_sub_latex_lines = []
    for i in range(min(n, m) - 1, -1, -1):
        if all(abs(aug[i][j]) < 1e-15 for j in range(n)):
            back_sub_latex_lines.append(rf"x_{{{i+1}}} = 0\;(\text{{tự do}})")
        else:
            rhs = round(aug[i][n], 7)
            terms_latex = [rf"{round(aug[i][j], 7)}" + r"\,x_{" + str(j+1) + "}" for j in range(i + 1, n) if abs(aug[i][j]) > 1e-15 and j < min(n, m)]
            div = round(aug[i][i] if abs(aug[i][i]) > 1e-15 else 1.0, 6)
            if terms_latex:
                back_sub_latex_lines.append(rf"x_{{{i+1}}} = \frac{{{rhs} - ({'+'.join(terms_latex)})}}{{{div}}} = {round(x[i], 7)}")
            else:
                back_sub_latex_lines.append(rf"x_{{{i+1}}} = \frac{{{rhs}}}{{{div}}} = {round(x[i], 7)}")
    steps.append({"step": len(steps), "description": "Back substitution" + col_label,
                  "back_substitution_latex": r"\\".join(back_sub_latex_lines),
                  "solution_vector": [round(xi, 7) for xi in x], "phase": "back_substitution"})
    return x


def gaussian_elimination(A, B):
    m = len(A); n = len(A[0]) if A else 0; p = len(B[0]) if B else 0
    analysis = _analyze_system(A, B)
    steps = []; pivot_info = []
    if analysis["solution_type"] == "inconsistent":
        steps.append({"step": 0, "description": analysis["message"]})
        return {"success": True, "message": analysis["message"], "solution": None, "steps": steps, "pivot_info": [],
                "execution_time": 0.0, "analysis": analysis, "solution_type": "inconsistent",
                "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                "free_variables": None, "particular_solution": None, "basis_vectors": None,
                "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}
    b0 = [B[i][0] for i in range(m)]
    aug0 = [list(A[i]) + [b0[i]] for i in range(m)]
    condition_step = {"step": 0, "description": f"Phân tích điều kiện hệ phương trình AX=B ({m}×{n}, B có {p} cột)",
                       "phase": "condition_analysis",
                       "matrix_properties": {"m": m, "n": n, "p": p, "is_square": (m == n),
                                              "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                                              "solution_type": analysis["solution_type"]}}
    if m == n and m <= 100:
        try:
            det_val = float(np.linalg.det(np.array(A, dtype=float)))
            if not isfinite(det_val): det_val = 0.0
            condition_step["determinant"] = round(det_val, 10)
            condition_step["is_singular"] = abs(det_val) < 1e-12
            if abs(det_val) < 1e-12:
                condition_step["singular_warning"] = f"det(A) ≈ 0 → ma trận suy biến hoặc gần suy biến"
        except Exception: pass
    steps.append(condition_step)
    steps.append({"step": 1, "description": f"Augmented matrix [A|B_col1] ({m}×{n+1})", "matrix": [row[:] for row in aug0]})
    start_time = time.time()
    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(m)]
        x_col = _solve_single_column_gaussian(A, b_col, n, m, steps, pivot_info, col_idx=col_idx, p=p)
        if x_col is None:
            steps.append({"step": len(steps), "description": f"Phát hiện mâu thuẫn tại cột B thứ {col_idx+1}: 0·x = b ≠ 0 → hệ vô nghiệm"})
            return {"success": True, "message": f"Hệ vô nghiệm: mâu thuẫn tại cột B thứ {col_idx+1}.",
                    "solution": None, "steps": steps, "pivot_info": pivot_info,
                    "execution_time": round(time.time() - start_time, 6), "analysis": analysis,
                    "solution_type": "inconsistent", "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}
        x_cols.append([round(xi, 10) for xi in x_col])
    execution_time = time.time() - start_time
    rank_A_comp = analysis["rank_A"]; rank_aug_comp = analysis["rank_augmented"]
    aug_check = [list(A[i]) + [B[i][0]] for i in range(m)]
    if rank_A_comp < n:
        return _build_infinite_result(aug_check, m, n, analysis, steps, execution_time, B)
    solution = _transpose_solution(x_cols, n, p)
    steps.append({"step": len(steps), "description": f"Back substitution result ({n}×{p})", "solution": solution})
    return {"success": True, "message": analysis["message"], "solution": solution, "steps": steps, "pivot_info": pivot_info,
            "execution_time": round(execution_time, 7), "analysis": analysis, "solution_type": "unique",
            "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}


def _compute_rref_full(A, B, m, n, p, steps):
    aug = [list(A[i]) + [B[i][ci] for ci in range(p)] for i in range(m)]
    min_dim = min(m, n); col = 0
    for k in range(min_dim):
        if col >= n: break
        max_row = k; max_val = abs(aug[k][col]) if k < m else 0
        for i in range(k + 1, m):
            if abs(aug[i][col]) > max_val: max_val = abs(aug[i][col]); max_row = i
        if max_val < 1e-15: col += 1; continue
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            steps.append({"step": len(steps), "description": f"Doi hang: R{k+1} <-> R{max_row+1}",
                          "matrix": [[round(v, 7) for v in row] for row in aug],
                          "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True),
                          "row_operations_latex": [rf"R_{{{k+1}}} \leftrightarrow R_{{{max_row+1}}}"], "phase": "row_swap"})
        pivot = aug[k][col]; pivot_rounded = round(pivot, 4)
        for j in range(col, n + p): aug[k][j] /= pivot
        steps.append({"step": len(steps), "description": f"Chuan hoa pivot: R{k+1} = R{k+1} / {pivot_rounded}",
                      "matrix": [[round(v, 7) for v in row] for row in aug],
                      "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True),
                      "row_operations_latex": [rf"R_{{{k+1}}} \leftarrow \frac{{R_{{{k+1}}}}}{{{pivot_rounded}}}"],
                      "phase": "normalize", "pivot_row": k + 1, "pivot_col": col + 1})
        row_ops = []; row_ops_latex = []
        for i in range(m):
            if i == k or abs(aug[i][col]) < 1e-15: continue
            factor = aug[i][col]
            for j in range(col, n + p): aug[i][j] -= factor * aug[k][j]
            row_ops.append({"target_row": i + 1, "factor": round(factor, 7), "pivot_row": k + 1})
            row_ops_latex.append(rf"R_{{{i+1}}} \leftarrow R_{{{i+1}}} - ({round(factor, 4)}) R_{{{k+1}}}")
        if row_ops:
            steps.append({"step": len(steps), "description": f"Khu cot {col+1} tren cac hang khac",
                          "matrix": [[round(v, 7) for v in row] for row in aug],
                          "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True),
                          "row_operations": row_ops, "row_operations_latex": row_ops_latex, "phase": "eliminate"})
        col += 1
    steps.append({"step": len(steps), "description": "Dang bac thang rut gon (RREF)",
                  "matrix": [[round(v, 7) for v in row] for row in aug],
                  "matrix_latex": _matrix_to_latex(aug, precision=7, augmented=True), "phase": "rref"})
    return aug


def gauss_jordan(A, B):
    m = len(A); n = len(A[0]) if A else 0; p = len(B[0]) if B else 0
    analysis = _analyze_system(A, B); steps = []
    condition_step = {"step": 0, "description": f"Phân tích điều kiện hệ phương trình AX=B ({m}×{n}, B có {p} cột)",
                       "phase": "condition_analysis",
                       "matrix_properties": {"m": m, "n": n, "p": p, "is_square": (m == n),
                                              "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                                              "solution_type": analysis["solution_type"]}}
    if m == n and m <= 100:
        try:
            det_val = float(np.linalg.det(np.array(A, dtype=float)))
            if not isfinite(det_val): det_val = 0.0
            condition_step["determinant"] = round(det_val, 10)
            condition_step["is_singular"] = abs(det_val) < 1e-12
            if abs(det_val) < 1e-12:
                condition_step["singular_warning"] = f"det(A) ≈ 0 → ma trận suy biến hoặc gần suy biến"
        except Exception: pass
    steps.append(condition_step)
    steps.append({"step": 1, "description": f"Augmented matrix [A|B] ({m}×{n+p})",
                  "matrix": [list(A[i]) + [B[i][ci] for ci in range(p)] for i in range(m)]})
    start_time = time.time()
    rref_full = _compute_rref_full(A, B, m, n, p, steps)
    execution_time = time.time() - start_time
    rank_A_comp = _matrix_rank(A)
    b0 = [B[i][0] for i in range(m)]
    aug_check = [list(A[i]) + [b0[i]] for i in range(m)]
    rank_aug_comp = _matrix_rank(aug_check)
    if rank_A_comp < rank_aug_comp:
        steps.append({"step": len(steps), "description": f"rank(A) = {rank_A_comp} < rank([A|B]) = {rank_aug_comp} -> he vo nghiem"})
        return {"success": True, "message": analysis["message"], "solution": None, "steps": steps,
                "execution_time": round(execution_time, 7), "analysis": analysis, "solution_type": "inconsistent",
                "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
                "free_variables": None, "particular_solution": None, "basis_vectors": None,
                "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}
    if rank_A_comp < n:
        return _build_infinite_result_from_rref(rref_full, m, n, p, analysis, steps, execution_time)
    solution = _extract_solution_from_rref(rref_full, n, p)
    sol_lines = []; sol_latex_lines = []
    for i in range(n):
        xi = round(solution[i][0], 7)
        sol_lines.append(f"x{i+1} = {xi}")
        sol_latex_lines.append(rf"x_{{{i+1}}} = {xi}")
    if p > 1:
        for ci in range(1, p):
            for i in range(n):
                solution[i][ci] = round(solution[i][ci], 7)
                sol_lines[i] += f", col{ci+1}: {solution[i][ci]}"
    steps.append({"step": len(steps), "description": "Nghiem tu RREF", "solution_lines": sol_lines,
                  "back_substitution_latex": r"\\".join(sol_latex_lines), "phase": "solution"})
    steps.append({"step": len(steps), "description": f"Reduced row echelon form ({n}×{p})", "solution": solution})
    return {"success": True, "message": analysis["message"], "solution": solution, "steps": steps,
            "execution_time": round(execution_time, 7), "analysis": analysis, "solution_type": "unique",
            "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}


def _extract_solution_from_rref(rref, n, p):
    return [[rref[i][n + ci] for ci in range(p)] for i in range(n)]


def _build_infinite_result_from_rref(rref_full, m, n, p, analysis, steps, execution_time):
    rref_data = _compute_rref_analysis(rref_full, m, n, p)
    rank_a = _matrix_rank([rref_full[i][:n] for i in range(m)])
    steps.append({"step": len(steps),
                  "description": f"Reduced Row Echelon Form. rank(A)={analysis['rank_A']} < n={n} -> vo so nghiem",
                  "matrix": [row[:n+1] for row in rref_full],
                  "matrix_latex": _matrix_to_latex([row[:n+1] for row in rref_full], precision=7, augmented=True)})
    sol_latex_lines = [rf"x_{{{i+1}}}^{{(p)}} = {round(rref_data['particular_solution'][i][0], 7)}" for i in range(n)]
    steps.append({"step": len(steps), "description": "Nghiem rieng Xp va co so nullspace",
                  "back_substitution_latex": r"\\".join(sol_latex_lines), "phase": "solution"})
    return {"success": True, "message": analysis["message"], "solution": None, "steps": steps, "pivot_info": None,
            "execution_time": round(execution_time, 7), "analysis": analysis, "solution_type": "infinite",
            "rank_A": rank_a, "rank_augmented": _matrix_rank([rref_full[i][:n+1] for i in range(m)]),
            "free_variables": rref_data["free_variables"],
            "particular_solution": rref_data["particular_solution"], "basis_vectors": rref_data["basis_vectors"],
            "general_solution_latex": rref_data["general_solution_latex"],
            "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}


def _lu_factorize(A, n, steps):
    a = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]; U = [[0.0] * n for _ in range(n)]
    P_mat = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for k in range(n):
        max_row = k; max_val = abs(a[k][k])
        for i in range(k + 1, n):
            if abs(a[i][k]) > max_val: max_val = abs(a[i][k]); max_row = i
        if max_row != k:
            a[k], a[max_row] = a[max_row], a[k]
            P_mat[k], P_mat[max_row] = P_mat[max_row], P_mat[k]
            for j in range(k): L[k][j], L[max_row][j] = L[max_row][j], L[k][j]
            steps.append({"step": len(steps), "description": f"Đổi hàng R{k+1} <-> R{max_row+1} (partial pivoting)",
                          "matrix": [[round(v, 7) for v in row] for row in a],
                          "phase": "lu_elimination", "pivot_col": k + 1, "pivot_row": k + 1})
        for j in range(k, n): U[k][j] = a[k][j] - sum(L[k][p] * U[p][j] for p in range(k))
        L[k][k] = 1.0
        elim_factors = []
        for i in range(k + 1, n):
            L[i][k] = a[i][k] - sum(L[i][p] * U[p][k] for p in range(k))
            if abs(U[k][k]) > 1e-15: L[i][k] /= U[k][k]
            elim_factors.append({"row": i + 1, "factor": round(L[i][k], 7)})
        desc = f"Khử cột {k+1}: pivot U[{k+1},{k+1}] = {round(U[k][k], 7)}"
        if elim_factors:
            factor_strs = [f"L[{op['row']},{k+1}] = {op['factor']}" for op in elim_factors]
            desc += "; " + "; ".join(factor_strs)
        combined = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                combined[i][j] = L[i][j] if i > j else U[i][j]
        steps.append({"step": len(steps), "description": desc, "matrix": [[round(v, 7) for v in row] for row in combined],
                      "pivot_value": round(U[k][k], 7), "pivot_col": k + 1, "pivot_row": k + 1,
                      "elimination_factors": elim_factors if elim_factors else None, "phase": "lu_elimination"})
    return L, U, P_mat


def _solve_lu_single(L, U, P_mat, b_col, n):
    Pb = [sum(P_mat[i][j] * b_col[j] for j in range(n)) for i in range(n)]
    Y = [0.0] * n
    for i in range(n): Y[i] = Pb[i] - sum(L[i][j] * Y[j] for j in range(i))
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = Y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))
        if abs(U[i][i]) > 1e-15: x[i] /= U[i][i]
    return x


def lu_decomposition(A, B):
    m = len(A); n = len(A[0]) if A else 0; p = len(B[0]) if B else 0
    if m != n: return {"success": False, "message": f"LU Decomposition yêu cầu ma trận vuông (A hiện tại: {m}×{n})", "steps": []}
    analysis = _analyze_system(A, B)
    steps = [{"step": 0, "description": "Initial matrix A", "matrix": [row[:] for row in A]}]
    start_time = time.time()
    L, U, P_mat = _lu_factorize(A, n, steps)
    steps.append({"step": len(steps), "description": "L matrix", "matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)]})
    steps.append({"step": len(steps), "description": "U matrix", "matrix": [[round(U[i][j], 7) for j in range(n)] for i in range(n)]})
    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(n)]
        x_col = _solve_lu_single(L, U, P_mat, b_col, n)
        x_cols.append([round(xi, 10) for xi in x_col])
    solution = _transpose_solution(x_cols, n, p)
    execution_time = time.time() - start_time
    return {"success": True, "message": "System solved successfully using LU Decomposition", "solution": solution,
            "steps": steps, "execution_time": round(execution_time, 7), "analysis": analysis,
            "solution_type": "unique", "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}


def _cholesky_solve_single(L, b_col, n):
    Y = [0.0] * n
    for i in range(n):
        if abs(L[i][i]) < 1e-15: return None
        Y[i] = (b_col[i] - sum(L[i][j] * Y[j] for j in range(i))) / L[i][i]
    Lt = [[L[j][i] for j in range(n)] for i in range(n)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(Lt[i][i]) < 1e-15: return None
        x[i] = (Y[i] - sum(Lt[i][j] * x[j] for j in range(i + 1, n))) / Lt[i][i]
    return x


def cholesky_decomposition(A, B):
    m = len(A); n = len(A[0]) if A else 0; p = len(B[0]) if B else 0
    if m != n: return {"success": False, "message": f"Cholesky Decomposition yêu cầu ma trận vuông (A hiện tại: {m}×{n})", "steps": []}
    analysis = _analyze_system(A, B); steps = []; start_time = time.time()
    condition_step = {"step": 0, "description": f"Phân tích điều kiện: Cholesky yêu cầu ma trận vuông ({n}x{n}), đối xứng, xác định dương",
                       "phase": "condition_analysis",
                       "matrix_properties": {"m": m, "n": n, "p": p, "is_square": True,
                                              "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                                              "solution_type": analysis["solution_type"]}}
    if m <= 100:
        try:
            det_val = float(np.linalg.det(np.array(A, dtype=float)))
            if not isfinite(det_val): det_val = 0.0
            condition_step["determinant"] = round(det_val, 10)
        except Exception: pass
    steps.append(condition_step)
    is_symmetric = True
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-10: is_symmetric = False
    steps.append({"step": len(steps), "description": f"Kiểm tra đối xứng A = A^T → {'ĐẠT' if is_symmetric else 'KHÔNG ĐẠT'}",
                  "phase": "cholesky_check", "is_symmetric": is_symmetric})
    if not is_symmetric:
        return {"success": False, "message": "Ma trận không đối xứng. Phương pháp Cholesky không thể áp dụng.",
                "steps": steps, "execution_time": round(time.time() - start_time, 6)}
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    steps.append({"step": len(steps), "description": f"Tính L[{i+1},{i+1}]: val <= 0 → KHÔNG xác định dương", "phase": "cholesky_compute"})
                    return {"success": False, "message": "Ma trận không xác định dương.", "steps": steps, "execution_time": round(time.time() - start_time, 6)}
                L[i][j] = np.sqrt(val)
            else:
                if abs(L[j][j]) > 1e-15: L[i][j] = (A[i][j] - s) / L[j][j]
                else: L[i][j] = 0.0
        L_current = [[round(L[ri][rj], 7) if L[ri][rj] != 0 or ri >= rj else 0.0 for rj in range(n)] for ri in range(n)]
        steps.append({"step": len(steps), "description": f"Hàng {i+1}", "matrix": L_current, "phase": "cholesky_compute", "row_idx": i + 1})
    L_rounded = [[round(L[i][j], 7) for j in range(n)] for i in range(n)]
    steps.append({"step": len(steps), "description": "Ma trận tam giác dưới L (A = L·L^T)", "matrix": L_rounded, "phase": "cholesky_L"})
    L_np = np.array(L, dtype=float); LLT = L_np @ L_np.T
    max_err = float(np.max(np.abs(LLT - np.array(A, dtype=float))))
    steps.append({"step": len(steps), "description": f"Kiểm tra L·L^T ≈ A → sai số max = {max_err:.2e}", "phase": "cholesky_verify"})
    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(n)]
        Y = [0.0] * n
        for i in range(n):
            if abs(L[i][i]) < 1e-15: return {"success": False, "message": f"Cholesky solve failed for column {col_idx+1}", "steps": steps, "execution_time": round(time.time() - start_time, 6)}
            Y[i] = (b_col[i] - sum(L[i][j] * Y[j] for j in range(i))) / L[i][i]
        steps.append({"step": len(steps), "description": f"Thế xuôi LY = B{' (cột B thứ '+str(col_idx+1)+')' if p>1 else ''}",
                      "solution_vector": [round(yi, 7) for yi in Y], "phase": "cholesky_forward"})
        Lt = [[L[j][i] for j in range(n)] for i in range(n)]
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(Lt[i][i]) < 1e-15: return {"success": False, "message": f"Cholesky solve failed for column {col_idx+1}", "steps": steps, "execution_time": round(time.time() - start_time, 6)}
            x[i] = (Y[i] - sum(Lt[i][j] * x[j] for j in range(i + 1, n))) / Lt[i][i]
        steps.append({"step": len(steps), "description": f"Thế ngược L^T X = Y{' (cột B thứ '+str(col_idx+1)+')' if p>1 else ''}",
                      "solution_vector": [round(xi, 7) for xi in x], "phase": "cholesky_back"})
        x_cols.append([round(xi, 10) for xi in x])
    solution = _transpose_solution(x_cols, n, p)
    steps.append({"step": len(steps), "description": f"Nghiệm X ({n}x{p})", "solution": solution, "phase": "solution"})
    execution_time = time.time() - start_time
    return {"success": True, "message": "Giải thành công bằng Cholesky Decomposition", "solution": solution,
            "steps": steps, "execution_time": round(execution_time, 7), "analysis": analysis,
            "solution_type": "unique", "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0, "convergence_data": None}



# =============================================================================
# MATRIX INVERSE
# =============================================================================

def _validate_matrix_inverse(A):
    if not A or not A[0]: return {"success": False, "message": "Ma trận A không được rỗng."}
    m = len(A); n = len(A[0])
    for i in range(m):
        for j in range(n):
            if not np.isfinite(float(A[i][j])):
                return {"success": False, "message": f"Dữ liệu không hợp lệ tại A[{i+1}][{j+1}] = {A[i][j]}."}
    if m != n: return {"success": False, "message": f"Không thể tính ma trận nghịch đảo vì A không phải ma trận vuông (A là {m}×{n})."}
    mat = np.array(A, dtype=float); det = float(np.linalg.det(mat))
    if abs(det) < 1e-15:
        return {"success": False, "message": f"Ma trận suy biến (det(A) = {det:.6e} ≈ 0).", "determinant": round(det, 10)}
    return None


def _compute_rank(A): return _matrix_rank(A)


def matrix_inverse_gauss_jordan(A):
    err = _validate_matrix_inverse(A)
    if err: return err
    n = len(A); mat = np.array(A, dtype=float); det = float(np.linalg.det(mat)); rank_val = _compute_rank(A)
    aug = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    steps = [{"step": 0, "description": f"Augmented matrix [A|I] ({n}×{2*n})", "matrix": [[round(v, 7) for v in row] for row in aug]}]
    start_time = time.time(); col = 0
    for k in range(n):
        max_row = k; max_val = abs(aug[k][col]) if k < n else 0
        for i in range(k + 1, n):
            if abs(aug[i][col]) > max_val: max_val = abs(aug[i][col]); max_row = i
        if max_val < 1e-15:
            return {"success": False, "message": f"Ma trận suy biến (zero pivot tại cột {col+1}).",
                    "determinant": round(det, 10), "rank": rank_val}
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            steps.append({"step": len(steps), "description": f"Swap R{k+1} <-> R{max_row+1}",
                          "matrix": [[round(v, 7) for v in row] for row in aug]})
        pivot = aug[k][col]
        for j in range(2 * n): aug[k][j] /= pivot
        steps.append({"step": len(steps), "description": f"R{k+1} = R{k+1} / ({round(pivot, 7)})",
                      "matrix": [[round(v, 7) for v in row] for row in aug]})
        for i in range(n):
            if i == k or abs(aug[i][col]) < 1e-15: continue
            factor = aug[i][col]
            for j in range(2 * n): aug[i][j] -= factor * aug[k][j]
            steps.append({"step": len(steps), "description": f"R{i+1} = R{i+1} - ({round(factor, 7)}) * R{k+1}",
                          "matrix": [[round(v, 7) for v in row] for row in aug]})
        col += 1
    execution_time = time.time() - start_time
    inv = [[aug[i][n + j] for j in range(n)] for i in range(n)]
    inv_rounded = [[round(v, 10) for v in row] for row in inv]
    verify = np.dot(mat, np.array(inv, dtype=float))
    steps.append({"step": len(steps), "description": f"[I|A⁻¹] — Inverse computed. det(A) = {round(det, 7)}, rank(A) = {rank_val}", "inverse": inv_rounded})
    return {"success": True, "message": f"Ma trận nghịch đảo tính thành công bằng Gauss-Jordan. det(A) = {round(det, 7)}",
            "determinant": round(det, 10), "rank": rank_val, "inverse": inv_rounded,
            "verification": [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)],
            "is_accurate": bool(np.allclose(verify, np.eye(n), atol=1e-8)), "steps": steps, "execution_time": round(execution_time, 7)}


def matrix_inverse_adjoint(A):
    err = _validate_matrix_inverse(A)
    if err: return err
    n = len(A); mat = np.array(A, dtype=float); det = float(np.linalg.det(mat)); rank_val = _compute_rank(A)
    steps = [{"step": 0, "description": f"Input matrix A ({n}×{n}), det(A) = {round(det, 7)}",
              "matrix": [[round(v, 7) for v in row] for row in A]}]
    start_time = time.time()
    cof = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(mat, i, 0), j, 1)
            cof[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    steps.append({"step": 1, "description": "Ma trận cofactor C", "matrix": [[round(float(cof[i, j]), 6) for j in range(n)] for i in range(n)]})
    adj = cof.T
    steps.append({"step": 2, "description": "Ma trận phụ hợp Adj(A) = C^T", "matrix": [[round(float(adj[i, j]), 6) for j in range(n)] for i in range(n)]})
    inv = adj / det
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 3, "description": f"A⁻¹ = Adj(A) / det(A) = Adj(A) / {round(det, 7)}", "inverse": inv_rounded})
    execution_time = time.time() - start_time
    verify = np.dot(mat, inv)
    return {"success": True, "message": f"Ma trận nghịch đảo tính thành công bằng Adjoint. det(A) = {round(det, 7)}",
            "determinant": round(det, 10), "rank": rank_val, "inverse": inv_rounded,
            "verification": [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)],
            "is_accurate": bool(np.allclose(verify, np.eye(n), atol=1e-8)), "steps": steps, "execution_time": round(execution_time, 7)}


def matrix_inverse_lu(A):
    err = _validate_matrix_inverse(A)
    if err: return err
    n = len(A); mat = np.array(A, dtype=float); det = float(np.linalg.det(mat)); rank_val = _compute_rank(A)
    steps = [{"step": 0, "description": f"Input matrix A ({n}×{n}), det(A) = {round(det, 7)}",
              "matrix": [[round(v, 7) for v in row] for row in A]}]
    start_time = time.time()
    L, U, P_mat = _lu_factorize(A, n, steps)
    steps.append({"step": len(steps), "description": "Ma trận L", "matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)]})
    steps.append({"step": len(steps), "description": "Ma trận U", "matrix": [[round(U[i][j], 7) for j in range(n)] for i in range(n)]})
    inv = np.zeros((n, n), dtype=float)
    for col_idx in range(n):
        x = _solve_lu_single(L, U, P_mat, [1.0 if i == col_idx else 0.0 for i in range(n)], n)
        for i in range(n): inv[i, col_idx] = x[i]
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": len(steps), "description": "Giải AX = I qua LU → A⁻¹", "inverse": inv_rounded})
    execution_time = time.time() - start_time
    verify = np.dot(mat, np.array(inv, dtype=float))
    return {"success": True, "message": f"Ma trận nghịch đảo tính thành công bằng LU Decomposition. det(A) = {round(det, 7)}",
            "determinant": round(det, 10), "rank": rank_val, "inverse": inv_rounded,
            "verification": [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)],
            "is_accurate": bool(np.allclose(verify, np.eye(n), atol=1e-8)), "steps": steps, "execution_time": round(execution_time, 7)}


def matrix_inverse_cholesky(A):
    err = _validate_matrix_inverse(A)
    if err: return err
    n = len(A); mat = np.array(A, dtype=float); det_val = float(np.linalg.det(mat)); rank_val = _compute_rank(A)
    steps = [{"step": 0, "description": f"Input matrix A ({n}×{n})", "matrix": [[round(v, 7) for v in row] for row in A]}]
    start_time = time.time()
    if not np.allclose(mat, mat.T, atol=1e-10):
        return {"success": False, "message": "Phương pháp Cholesky yêu cầu ma trận đối xứng.", "steps": [], "execution_time": round(time.time() - start_time, 6)}
    try: L_np = np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
        return {"success": False, "message": "Phương pháp Cholesky yêu cầu ma trận xác định dương.", "steps": [], "execution_time": round(time.time() - start_time, 6)}
    L = [[float(L_np[i, j]) for j in range(n)] for i in range(n)]
    steps.append({"step": 1, "description": "Ma trận L (Cholesky): A = L·L^T", "matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)]})
    L_inv = np.zeros((n, n), dtype=float)
    for col_idx in range(n):
        x = [0.0] * n
        for i in range(n):
            x[i] = (1.0 if i == col_idx else 0.0) - sum(L[i][j] * x[j] for j in range(i))
            if abs(L[i][i]) > 1e-15: x[i] /= L[i][i]
        for i in range(n): L_inv[i, col_idx] = x[i]
    inv = np.dot(L_inv.T, L_inv)
    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 1, "description": "A⁻¹ = (L⁻¹)^T · L⁻¹", "inverse": inv_rounded})
    execution_time = time.time() - start_time
    verify = np.dot(mat, inv)
    return {"success": True, "message": f"Ma trận nghịch đảo tính thành công bằng Cholesky Decomposition. det(A) = {round(det_val, 7)}",
            "determinant": round(det_val, 10), "rank": rank_val, "inverse": inv_rounded,
            "verification": [[round(float(verify[i, j]), 10) for j in range(n)] for i in range(n)],
            "is_accurate": bool(np.allclose(verify, np.eye(n), atol=1e-8)), "steps": steps, "execution_time": round(execution_time, 7)}


# =============================================================================
# ITERATIVE METHODS
# =============================================================================

def _check_iterative_b(A, B):
    m = len(A)
    if len(B) != m: return {"success": False, "message": f"Kích thước B không khớp: rows(B)={len(B)} must equal rows(A)={m}.", "iterations": []}
    p = len(B[0]) if B else 0
    if p != 1: return {"success": False, "message": "Phương pháp lặp hiện chỉ hỗ trợ vector vế phải.", "iterations": []}
    return None


def jacobi(A, B, initial_guess=None, epsilon=1e-6, max_iterations=100):
    """Jacobi — auto-detects row or column diagonal dominance and uses appropriate formula."""
    eps_eff = _effective_epsilon(epsilon)
    m = len(A); n = len(A[0]) if A else 0
    err = _check_iterative_b(A, B)
    if err: return err
    b_vec = [B[i][0] for i in range(m)]
    if m != n:
        return {"success": False, "message": f"Jacobi chỉ hỗ trợ ma trận vuông (A hiện tại: {m}×{n})", "iterations": []}
    analysis = _analyze_system(A, B); steps = []
    x_guess = list(initial_guess) if initial_guess else [0.0] * n

    # ── Check BOTH row and column diagonal dominance ──
    is_dd_row = True; is_dd_col = True
    dd_row_details = []; dd_col_details = []
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) <= row_sum:
            is_dd_row = False
            dd_row_details.append(f"Hàng {i+1}: |a_{{{i+1}{i+1}}}| = {round(abs(A[i][i]), 6)} <= {round(row_sum, 7)}")
        col_sum = sum(abs(A[j][i]) for j in range(n) if j != i)
        if abs(A[i][i]) <= col_sum:
            is_dd_col = False
            dd_col_details.append(f"Cột {i+1}: |a_{{{i+1}{i+1}}}| = {round(abs(A[i][i]), 6)} <= {round(col_sum, 7)}")

    # Decide which formula to use: row-dom → row formula; col-dom only → column (transpose) formula
    use_column_formula = (not is_dd_row) and is_dd_col
    dominance_type = "column" if use_column_formula else ("row" if is_dd_row else "none")
    is_dd = is_dd_row or is_dd_col

    desc_parts = []
    desc_parts.append(f"Chéo trội hàng: {'ĐẠT' if is_dd_row else 'KHÔNG ĐẠT'}")
    desc_parts.append(f"Chéo trội cột: {'ĐẠT' if is_dd_col else 'KHÔNG ĐẠT'}")
    if use_column_formula:
        desc_parts.append("→ Sử dụng công thức chéo trội CỘT (chuyển vị)")
    elif is_dd_row:
        desc_parts.append("→ Sử dụng công thức chéo trội HÀNG")
    else:
        desc_parts.append("→ Cảnh báo: không chéo trội, có thể không hội tụ")

    steps.append({"step": 0,
                  "description": f"Phân tích điều kiện hội tụ Jacobi: {'; '.join(desc_parts)}",
                  "phase": "condition_analysis",
                  "matrix_properties": {"m": m, "n": n, "p": 1, "is_square": True,
                                         "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                                         "solution_type": analysis["solution_type"],
                                         "is_diagonally_dominant": is_dd,
                                         "dominance_type": dominance_type},
                  "convergence_details": dd_row_details + dd_col_details})

    if is_dd_row:
        rec = "Hội tụ đảm bảo (chéo trội hàng)"
    elif is_dd_col:
        rec = "Hội tụ đảm bảo (chéo trội cột — dùng công thức chuyển vị)"
    else:
        rec = "Có thể không hội tụ (không chéo trội hàng lẫn cột)"
    convergence_analysis = {"is_diagonally_dominant": is_dd,
                            "dominance_type": dominance_type,
                            "is_dd_row": is_dd_row, "is_dd_col": is_dd_col,
                            "row_details": dd_row_details if dd_row_details else ["Mọi hàng đều chéo trội"],
                            "col_details": dd_col_details if dd_col_details else ["Mọi cột đều chéo trội"],
                            "recommendation": rec,
                            "initial_guess": [round(v, 7) for v in x_guess]}

    # ── Decompose A = D + L + U (shared by both) ──
    D = [[0.0] * n for _ in range(n)]; L = [[0.0] * n for _ in range(n)]; U_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: D[i][j] = A[i][j]
            elif i > j: L[i][j] = A[i][j]
            else: U_mat[i][j] = A[i][j]
    steps.append({"step": len(steps), "description": "Phân rã A = D + L + U (D: chéo chính, L: tam giác dưới, U: tam giác trên)",
                  "phase": "iterative_decompose",
                  "D_matrix": [[round(D[i][j], 7) for j in range(n)] for i in range(n)],
                  "L_matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)],
                  "U_matrix": [[round(U_mat[i][j], 7) for j in range(n)] for i in range(n)]})

    D_inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if abs(D[i][i]) > 1e-15: D_inv[i][i] = 1.0 / D[i][i]
        else: return {"success": False, "message": f"Zero diagonal at row {i+1}", "iterations": [], "steps": steps,
                      "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                      "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}

    L_plus_U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j: L_plus_U[i][j] = A[i][j]

    # B_iter and d_vec (same for both formulas)
    B_iter = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n): B_iter[i][j] = -D_inv[i][i] * L_plus_U[i][j]
        B_iter[i][i] = 0.0
    d_vec = [0.0] * n
    for i in range(n): d_vec[i] = D_inv[i][i] * b_vec[i]
    steps.append({"step": len(steps), "description": "Ma trận lặp B = -D⁻¹(L+U) và vector d = D⁻¹b (x = Bx + d)",
                  "phase": "iteration_matrix",
                  "B_matrix": [[round(B_iter[i][j], 7) for j in range(n)] for i in range(n)],
                  "d_vector": [round(v, 7) for v in d_vec]})

    # ── Formula (row or column depending on dominance) ──
    formula_lines = []
    formula_type = "CHÉO TRỘI CỘT (dùng a_ji)" if use_column_formula else "CHÉO TRỘI HÀNG (dùng a_ij)"
    for i in range(n):
        if use_column_formula:
            terms = [f"{round(-A[j][i] / A[i][i], 7)} · x_{j+1}^{{(k)}}" for j in range(n) if j != i and abs(A[j][i]) > 1e-15]
        else:
            terms = [f"{round(-A[i][j] / A[i][i], 7)} · x_{j+1}^{{(k)}}" for j in range(n) if j != i and abs(A[i][j]) > 1e-15]
        d_term = round(b_vec[i] / A[i][i], 7)
        if terms: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {' + '.join(terms)} + ({d_term})")
        else: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {d_term}")
    steps.append({"step": len(steps), "description": f"Công thức lặp Jacobi cho từng ẩn ({formula_type})",
                  "phase": "iterative_formula", "formula_lines": formula_lines, "dominance_type": dominance_type})

    # ── Iterative steps ──
    x = x_guess; start_time = time.time(); iterations = []; error = 0.0
    show_detail_steps = min(3, max_iterations)
    for k in range(max_iterations):
        x_new = [0.0] * n; iter_details = []
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                return {"success": False, "message": f"Zero diagonal element at row {i+1}", "iterations": iterations,
                        "steps": steps, "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
            if use_column_formula:
                s = sum(A[j][i] * x[j] for j in range(n) if j != i)
            else:
                s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b_vec[i] - s) / A[i][i]
            if k < show_detail_steps:
                if use_column_formula:
                    sum_terms = " + ".join(f"{round(A[j][i], 7)}·{round(x[j], 7)}" for j in range(n) if j != i and abs(A[j][i]) > 1e-15)
                else:
                    sum_terms = " + ".join(f"{round(A[i][j], 7)}·{round(x[j], 7)}" for j in range(n) if j != i and abs(A[i][j]) > 1e-15)
                iter_details.append(f"x_{i+1} = ({round(b_vec[i], 7)} - ({sum_terms})) / {round(A[i][i], 7)} = {round(x_new[i], 7)}")
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if k < show_detail_steps:
            steps.append({"step": len(steps),
                          "description": f"Vòng lặp {k+1} (ε = {round(error, 10)})" + (f" < {eps_eff} → hội tụ!" if error < eps_eff else ""),
                          "phase": "iterative_steps", "iteration_details": iter_details,
                          "x_current": [round(xi, 7) for xi in x], "x_new": [round(xi, 7) for xi in x_new]})
        if error < eps_eff:
            steps.append({"step": len(steps), "description": f"Hội tụ sau {k+1} vòng lặp (ε = {round(error, 10)} < {eps_eff})",
                          "solution_vector": [round(xi, 7) for xi in x_new], "phase": "solution"})
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations", "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10), "iterations": iterations,
                    "execution_time": round(time.time() - start_time, 7),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": steps, "analysis": analysis,
                    "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    steps.append({"step": len(steps), "description": f"Không hội tụ sau {max_iterations} vòng lặp (ε cuối = {round(error, 10)})", "phase": "solution"})
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations", "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10), "iterations": iterations,
            "execution_time": round(time.time() - start_time, 7), "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": steps, "analysis": analysis,
            "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


def simple_iteration(B_mat, d_vec, initial_guess=None, epsilon=1e-6, max_iterations=100):
    """Simple Iteration — x = Bx + d, x⁽ᵏ⁺¹⁾ = Bx⁽ᵏ⁾ + d."""
    eps_eff = _effective_epsilon(epsilon)
    m = len(B_mat); n = len(B_mat[0]) if B_mat else 0
    if m != n:
        return {"success": False, "message": f"Simple Iteration yêu cầu ma trận B vuông (B hiện tại: {m}×{n})", "iterations": []}
    if len(d_vec) != n:
        return {"success": False, "message": f"Kích thước d không khớp: len(d)={len(d_vec)} must equal rows(B)={n}.", "iterations": []}
    x_guess = list(initial_guess) if initial_guess else [0.0] * n
    steps = []
    # Phase 1: condition analysis
    steps.append({"step": 0,
                  "description": f"Phân tích điều kiện hội tụ Simple Iteration: ρ(B) < 1 là điều kiện cần và đủ",
                  "phase": "condition_analysis",
                  "matrix_properties": {"m": m, "n": n, "p": 1, "is_square": True}})
    convergence_analysis = {"method": "Simple Iteration",
                            "condition": "ρ(B) < 1",
                            "recommendation": "Hội tụ nếu bán kính phổ của B nhỏ hơn 1",
                            "initial_guess": [round(v, 7) for v in x_guess]}
    # Phase 2: iteration matrix is B itself
    steps.append({"step": len(steps), "description": "Ma trận lặp B và vector d (x = Bx + d)",
                  "phase": "iteration_matrix",
                  "B_matrix": [[round(B_mat[i][j], 7) for j in range(n)] for i in range(n)],
                  "d_vector": [round(v, 7) for v in d_vec]})
    # Phase 3: formula
    formula_lines = []
    for i in range(n):
        terms = [f"{round(B_mat[i][j], 7)} · x_{j+1}^{{(k)}}" for j in range(n) if abs(B_mat[i][j]) > 1e-15]
        d_term = round(d_vec[i], 7)
        if terms: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {' + '.join(terms)} + ({d_term})")
        else: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {d_term}")
    steps.append({"step": len(steps), "description": "Công thức lặp Simple Iteration", "phase": "iterative_formula", "formula_lines": formula_lines})
    # Phase 4: iterative steps
    x = x_guess; start_time = time.time(); iterations = []; error = 0.0
    show_detail_steps = min(3, max_iterations)
    for k in range(max_iterations):
        x_new = [0.0] * n; iter_details = []
        for i in range(n):
            s = sum(B_mat[i][j] * x[j] for j in range(n))
            x_new[i] = s + d_vec[i]
            if k < show_detail_steps:
                sum_terms = " + ".join(f"{round(B_mat[i][j], 7)}·{round(x[j], 7)}" for j in range(n) if abs(B_mat[i][j]) > 1e-15)
                iter_details.append(f"x_{i+1} = ({sum_terms}) + {round(d_vec[i], 7)} = {round(x_new[i], 7)}")
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if k < show_detail_steps:
            steps.append({"step": len(steps),
                          "description": f"Vòng lặp {k+1} (ε = {round(error, 10)})" + (f" < {eps_eff} → hội tụ!" if error < eps_eff else ""),
                          "phase": "iterative_steps", "iteration_details": iter_details,
                          "x_current": [round(xi, 7) for xi in x], "x_new": [round(xi, 7) for xi in x_new]})
        if error < eps_eff:
            steps.append({"step": len(steps), "description": f"Hội tụ sau {k+1} vòng lặp (ε = {round(error, 10)} < {eps_eff})",
                          "solution_vector": [round(xi, 7) for xi in x_new], "phase": "solution"})
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations", "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10), "iterations": iterations,
                    "execution_time": round(time.time() - start_time, 7),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": steps, "analysis": None,
                    "convergence_analysis": convergence_analysis,
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    steps.append({"step": len(steps), "description": f"Không hội tụ sau {max_iterations} vòng lặp (ε cuối = {round(error, 10)})", "phase": "solution"})
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations", "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10), "iterations": iterations,
            "execution_time": round(time.time() - start_time, 7), "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": steps, "analysis": None,
            "convergence_analysis": convergence_analysis,
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


def seidel(B_mat, d_vec, initial_guess=None, epsilon=1e-6, max_iterations=100):
    """Seidel Iteration — x = Bx + d, updates xᵢ using newest values immediately."""
    eps_eff = _effective_epsilon(epsilon)
    m = len(B_mat); n = len(B_mat[0]) if B_mat else 0
    if m != n:
        return {"success": False, "message": f"Seidel yêu cầu ma trận B vuông (B hiện tại: {m}×{n})", "iterations": []}
    if len(d_vec) != n:
        return {"success": False, "message": f"Kích thước d không khớp: len(d)={len(d_vec)} must equal rows(B)={n}.", "iterations": []}
    x_guess = list(initial_guess) if initial_guess else [0.0] * n
    steps = []
    # Phase 1: condition analysis
    steps.append({"step": 0,
                  "description": f"Phân tích điều kiện hội tụ Seidel: ρ(B) < 1 là điều kiện cần và đủ",
                  "phase": "condition_analysis",
                  "matrix_properties": {"m": m, "n": n, "p": 1, "is_square": True}})
    convergence_analysis = {"method": "Seidel Iteration",
                            "condition": "ρ(B) < 1",
                            "recommendation": "Hội tụ nếu bán kính phổ của B nhỏ hơn 1 (thường nhanh hơn Simple Iteration)",
                            "initial_guess": [round(v, 7) for v in x_guess]}
    # Phase 2: iteration matrix is B itself
    steps.append({"step": len(steps), "description": "Ma trận lặp B và vector d (x = Bx + d, cập nhật từng biến bằng giá trị mới nhất)",
                  "phase": "iteration_matrix",
                  "B_matrix": [[round(B_mat[i][j], 7) for j in range(n)] for i in range(n)],
                  "d_vector": [round(v, 7) for v in d_vec]})
    # Phase 3: formula (Seidel uses updated x_j within same iteration)
    formula_lines = []
    for i in range(n):
        terms_before = [f"{round(B_mat[i][j], 7)} · x_{j+1}^{{(k+1)}}" for j in range(i) if abs(B_mat[i][j]) > 1e-15]
        terms_after = [f"{round(B_mat[i][j], 7)} · x_{j+1}^{{(k)}}" for j in range(i, n) if abs(B_mat[i][j]) > 1e-15]
        d_term = round(d_vec[i], 7)
        all_terms = terms_before + terms_after
        if all_terms: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {' + '.join(all_terms)} + ({d_term})")
        else: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {d_term}")
    steps.append({"step": len(steps), "description": "Công thức lặp Seidel (dùng giá trị mới nhất)", "phase": "iterative_formula", "formula_lines": formula_lines})
    # Phase 4: iterative steps
    x = x_guess; start_time = time.time(); iterations = []; error = 0.0
    show_detail_steps = min(3, max_iterations)
    for k in range(max_iterations):
        x_new = list(x); iter_details = []
        for i in range(n):
            s = sum(B_mat[i][j] * x_new[j] for j in range(n))  # includes b_ii·x_i⁽ᵏ⁾ via x_new (=x_old until updated)
            x_new[i] = s + d_vec[i]
            if k < show_detail_steps:
                before_terms = " + ".join(f"{round(B_mat[i][j], 7)}·{round(x_new[j], 7)}" for j in range(i) if abs(B_mat[i][j]) > 1e-15)
                after_terms = " + ".join(f"{round(B_mat[i][j], 7)}·{round(x[j], 7)}" for j in range(i + 1, n) if abs(B_mat[i][j]) > 1e-15)
                sum_terms = (before_terms + " + " + after_terms) if (before_terms and after_terms) else (before_terms or after_terms or "0")
                iter_details.append(f"x_{i+1} = ({sum_terms}) + {round(d_vec[i], 7)} = {round(x_new[i], 7)}")
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if k < show_detail_steps:
            steps.append({"step": len(steps),
                          "description": f"Vòng lặp {k+1} (ε = {round(error, 10)})" + (f" < {eps_eff} → hội tụ!" if error < eps_eff else ""),
                          "phase": "iterative_steps", "iteration_details": iter_details,
                          "x_current": [round(xi, 7) for xi in x], "x_new": [round(xi, 7) for xi in x_new]})
        if error < eps_eff:
            steps.append({"step": len(steps), "description": f"Hội tụ sau {k+1} vòng lặp (ε = {round(error, 10)} < {eps_eff})",
                          "solution_vector": [round(xi, 7) for xi in x_new], "phase": "solution"})
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations", "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10), "iterations": iterations,
                    "execution_time": round(time.time() - start_time, 7),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": steps, "analysis": None,
                    "convergence_analysis": convergence_analysis,
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    steps.append({"step": len(steps), "description": f"Không hội tụ sau {max_iterations} vòng lặp (ε cuối = {round(error, 10)})", "phase": "solution"})
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations", "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10), "iterations": iterations,
            "execution_time": round(time.time() - start_time, 7), "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": steps, "analysis": None,
            "convergence_analysis": convergence_analysis,
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


def gauss_seidel(A, B, initial_guess=None, epsilon=1e-6, max_iterations=100):
    """Gauss-Seidel — uses A = D+L+U, B = -(D+L)⁻¹U, d = (D+L)⁻¹b, updates xᵢ immediately within each iteration."""
    eps_eff = _effective_epsilon(epsilon)
    m = len(A); n = len(A[0]) if A else 0
    err = _check_iterative_b(A, B)
    if err: return err
    b_vec = [B[i][0] for i in range(m)]
    if m != n:
        return {"success": False, "message": f"Gauss-Seidel chỉ hỗ trợ ma trận vuông (A hiện tại: {m}×{n})", "iterations": []}
    analysis = _analyze_system(A, B); steps = []
    x_guess = list(initial_guess) if initial_guess else [0.0] * n
    is_dd = True; dd_details = []
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) <= row_sum:
            is_dd = False
            dd_details.append(f"Hàng {i+1}: |a_{{{i+1}{i+1}}}| = {round(abs(A[i][i]), 6)} <= {round(row_sum, 7)}")
    steps.append({"step": 0,
                  "description": f"Phân tích điều kiện hội tụ Gauss-Seidel: {'ĐẠT chéo trội nghiêm ngặt' if is_dd else 'KHÔNG chéo trội nghiêm ngặt'}",
                  "phase": "condition_analysis",
                  "matrix_properties": {"m": m, "n": n, "p": 1, "is_square": True,
                                         "rank_A": analysis["rank_A"], "rank_augmented": analysis["rank_augmented"],
                                         "solution_type": analysis["solution_type"], "is_diagonally_dominant": is_dd},
                  "convergence_details": dd_details if dd_details else ["Tất cả các hàng đều chéo trội nghiêm ngặt → hội tụ đảm bảo"]})
    convergence_analysis = {"is_diagonally_dominant": is_dd,
                            "row_details": dd_details if dd_details else ["Mọi hàng đều chéo trội → hội tụ đảm bảo"],
                            "recommendation": "Hội tụ đảm bảo" if is_dd else "Có thể không hội tụ (không chéo trội nghiêm ngặt)",
                            "initial_guess": [round(v, 7) for v in x_guess]}
    # Phase 2: Decompose A = D + L + U
    D = [[0.0] * n for _ in range(n)]; L = [[0.0] * n for _ in range(n)]; U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: D[i][j] = A[i][j]
            elif i > j: L[i][j] = A[i][j]
            else: U[i][j] = A[i][j]
    steps.append({"step": len(steps), "description": "Phân rã A = D + L + U",
                  "phase": "iterative_decompose",
                  "D_matrix": [[round(D[i][j], 7) for j in range(n)] for i in range(n)],
                  "L_matrix": [[round(L[i][j], 7) for j in range(n)] for i in range(n)],
                  "U_matrix": [[round(U[i][j], 7) for j in range(n)] for i in range(n)]})
    # Phase 3: B = -(D+L)⁻¹U, d = (D+L)⁻¹b
    # Compute (D+L)⁻¹ via forward-substitution (since D+L is lower-triangular)
    try:
        DL_inv_np = np.linalg.inv(np.array([[D[i][j] + L[i][j] for j in range(n)] for i in range(n)], dtype=float))
    except np.linalg.LinAlgError:
        return {"success": False, "message": "Không thể nghịch đảo (D+L)", "iterations": [], "steps": steps,
                "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
    U_np = np.array(U, dtype=float)
    B_np = -DL_inv_np @ U_np
    d_np = DL_inv_np @ np.array(b_vec, dtype=float)
    B_iter = [[float(B_np[i, j]) for j in range(n)] for i in range(n)]
    d_vec = [float(d_np[i]) for i in range(n)]
    steps.append({"step": len(steps),
                  "description": "Ma trận lặp B = -(D+L)⁻¹U và vector d = (D+L)⁻¹b (x = Bx + d)",
                  "phase": "iteration_matrix",
                  "B_matrix": [[round(B_iter[i][j], 7) for j in range(n)] for i in range(n)],
                  "d_vector": [round(v, 7) for v in d_vec]})
    # Phase 4: Formula (GS uses updated x_j within same iteration)
    formula_lines = []
    for i in range(n):
        terms_before = [f"{round(-A[i][j] / A[i][i], 7)} · x_{j+1}^{{(k+1)}}" for j in range(i) if abs(A[i][j]) > 1e-15]
        terms_after = [f"{round(-A[i][j] / A[i][i], 7)} · x_{j+1}^{{(k)}}" for j in range(i + 1, n) if abs(A[i][j]) > 1e-15]
        d_term = round(b_vec[i] / A[i][i], 7)
        all_terms = terms_before + terms_after
        if all_terms: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {' + '.join(all_terms)} + ({d_term})")
        else: formula_lines.append(f"x_{i+1}^{{(k+1)}} = {d_term}")
    steps.append({"step": len(steps), "description": "Công thức lặp Gauss-Seidel cho từng ẩn (dùng giá trị mới nhất)",
                  "phase": "iterative_formula", "formula_lines": formula_lines})
    # Phase 5: Iterative steps (first 3 with details)
    x = x_guess; start_time = time.time(); iterations = []; error = 0.0
    show_detail_steps = min(3, max_iterations)
    for k in range(max_iterations):
        x_new = list(x); iter_details = []
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                return {"success": False, "message": f"Zero diagonal element at row {i+1}", "iterations": iterations,
                        "steps": steps, "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                        "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)  # uses x_new (updated values)
            x_new[i] = (b_vec[i] - s) / A[i][i]
            if k < show_detail_steps:
                before_terms = " + ".join(f"{round(A[i][j], 7)}·{round(x_new[j], 7)}" for j in range(i) if abs(A[i][j]) > 1e-15)
                after_terms = " + ".join(f"{round(A[i][j], 7)}·{round(x[j], 7)}" for j in range(i + 1, n) if abs(A[i][j]) > 1e-15)
                sum_terms = (before_terms + " + " + after_terms) if (before_terms and after_terms) else (before_terms or after_terms or "0")
                iter_details.append(f"x_{i+1} = ({round(b_vec[i], 7)} - ({sum_terms})) / {round(A[i][i], 7)} = {round(x_new[i], 7)}")
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if k < show_detail_steps:
            steps.append({"step": len(steps),
                          "description": f"Vòng lặp {k+1} (ε = {round(error, 10)})" + (f" < {eps_eff} → hội tụ!" if error < eps_eff else ""),
                          "phase": "iterative_steps", "iteration_details": iter_details,
                          "x_current": [round(xi, 7) for xi in x], "x_new": [round(xi, 7) for xi in x_new]})
        if error < eps_eff:
            steps.append({"step": len(steps), "description": f"Hội tụ sau {k+1} vòng lặp (ε = {round(error, 10)} < {eps_eff})",
                          "solution_vector": [round(xi, 7) for xi in x_new], "phase": "solution"})
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations", "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10), "iterations": iterations,
                    "execution_time": round(time.time() - start_time, 6),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": steps, "analysis": analysis,
                    "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    steps.append({"step": len(steps), "description": f"Không hội tụ sau {max_iterations} vòng lặp (ε cuối = {round(error, 10)})", "phase": "solution"})
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations", "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10), "iterations": iterations,
            "execution_time": round(time.time() - start_time, 6), "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": steps, "analysis": analysis,
            "diagonally_dominant": is_dd, "convergence_analysis": convergence_analysis,
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


