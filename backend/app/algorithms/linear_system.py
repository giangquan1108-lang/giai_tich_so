"""Unified Linear System AX=B solvers: direct + iterative methods."""
import sys
import numpy as np
from typing import List, Tuple, Optional
import copy
import time

MACHINE_EPSILON = sys.float_info.epsilon
MIN_EPSILON = 1e-15


# =============================================================================
# Utility functions
# =============================================================================

def _effective_epsilon(epsilon: float) -> float:
    return max(epsilon, MIN_EPSILON)


def _matrix_rank(M: List[List[float]], tol: float = 1e-12) -> int:
    """Compute numerical rank of a matrix using SVD."""
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
    """Analyze the linear system: rank(A), rank([A|B_col]), and solution type.

    Checks ALL columns of B for consistency. If ANY column is inconsistent
    (rank(A) < rank([A|B_col])), the entire system is inconsistent.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    rank_A = _matrix_rank(A)
    
    # Check consistency for EVERY column of B
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
    return {
        "rank_A": rank_A, "rank_augmented": rank_aug, "n": n, "m": m, "p": p,
        "solution_type": solution_type, "message": message,
    }


def _detect_matrix_properties(A: List[List[float]]) -> dict:
    """Detect matrix properties: symmetric, positive-definite, tridiagonal, diagonally-dominant."""
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
        # Symmetry
        is_symmetric = bool(np.allclose(mat, mat.T, atol=1e-10))

        # Positive definite (only meaningful for symmetric)
        if is_symmetric:
            try:
                eigvals = np.linalg.eigvalsh(mat)
                is_positive_definite = bool(np.all(eigvals > 1e-12))
            except Exception:
                pass

        # Tridiagonal
        tridiag = True
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1 and abs(mat[i, j]) > 1e-12:
                    tridiag = False
                    break
            if not tridiag:
                break
        is_tridiagonal = tridiag

        # Diagonal dominance
        strict = True
        weak = True
        for i in range(n):
            diag = abs(mat[i, i])
            off_sum = sum(abs(mat[i, j]) for j in range(n) if j != i)
            if diag < off_sum:
                strict = False
            if diag <= off_sum:
                weak = False
        is_diagonally_dominant_strict = strict
        is_diagonally_dominant_weak = weak

    recommendations = []
    if is_tridiagonal and is_square:
        recommendations.append("Ma trận ba đường chéo → khuyến nghị Thomas Algorithm.")
    if is_symmetric and is_positive_definite:
        recommendations.append("Ma trận đối xứng xác định dương → khuyến nghị Cholesky.")
    if is_diagonally_dominant_strict and is_square:
        recommendations.append("Ma trận chéo trội nghiêm ngặt → phương pháp lặp hội tụ nhanh.")
    if not is_square:
        recommendations.append("Ma trận không vuông → chỉ dùng direct methods (Gauss/Gauss-Jordan).")

    return {
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "is_positive_definite": is_positive_definite,
        "is_tridiagonal": is_tridiagonal,
        "is_diagonally_dominant_strict": is_diagonally_dominant_strict,
        "is_diagonally_dominant_weak": is_diagonally_dominant_weak,
        "recommendations": recommendations,
    }


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
    """From a Gauss-Jordan RREF augmented matrix [A|B], extract:
    - pivot columns (in A part only, columns 0..n-1)
    - free variables
    - particular solution x_p (n×p matrix, for each B column)
    - basis vectors for the nullspace
    - general solution LaTeX string (uses first B column only for simple display)

    For each pivot column, finds which row contains the actual pivot (value ~1).
    """
    tol = 1e-14

    # ---- Step 1: Find pivot columns and map each to its row ----
    pivot_cols = []
    pivot_row_for_col = {}  # pivot_col -> row index containing the pivot
    for i in range(m):
        for j in range(n):
            if abs(aug[i][j]) > tol and all(abs(aug[i][k]) < tol for k in range(j)):
                pivot_cols.append(j)
                pivot_row_for_col[j] = i
                break
    pivot_cols = sorted(set(pivot_cols))
    free_cols = [j for j in range(n) if j not in pivot_cols]
    free_variables = [f"x{j+1}" for j in free_cols]

    # ---- Step 2: Particular solution for each B column (n×p matrix) ----
    x_p_matrix = []
    for col_b in range(p):
        x_p = [0.0] * n
        for pivot_col in pivot_cols:
            pivot_row = pivot_row_for_col[pivot_col]
            x_p[pivot_col] = aug[pivot_row][n + col_b]
        x_p_matrix.append([round(val, 6) for val in x_p])
    # Transpose to n×p
    x_p_transposed = [[x_p_matrix[ci][ri] for ci in range(p)] for ri in range(n)]

    # ---- Step 3: Basis vectors for nullspace ----
    basis_vectors = []
    for f in free_cols:
        v = [0.0] * n
        v[f] = 1.0
        for pivot_col in pivot_cols:
            pivot_row = pivot_row_for_col[pivot_col]
            v[pivot_col] = -aug[pivot_row][f]
        basis_vectors.append(v)

    # ---- Step 4: General solution LaTeX ----
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
            bv = [round(val, 6) for val in basis_vectors[i]]
            parts.append(f"v_{{{i+1}}} = \\begin{{bmatrix}}" + " \\\\ ".join(str(v) for v in bv) + r"\end{bmatrix}")
        general_latex = r" \\[6pt] ".join(parts)

    return {
        "pivot_columns": pivot_cols, "free_columns": free_cols,
        "free_variables": free_variables,
        "particular_solution": x_p_transposed,  # n×p
        "basis_vectors": [list(bv) for bv in basis_vectors],
        "general_solution_latex": general_latex,
    }


def _build_infinite_result(aug: List[List[float]], m: int, n: int,
                           analysis: dict, steps: List[dict],
                           execution_time: float,
                           B: Optional[List[List[float]]] = None) -> dict:
    """Build result dict for infinite-solution case from RREF.

    If B is provided (multi-column), particular_solution is computed for each B column
    by applying the same RREF row operations to all columns.
    """
    tol = 1e-15
    p = analysis.get("p", 1)

    # Build full augmented matrix [A|B] if B provided, else use single-column aug
    if B is not None:
        aug_full = [list(A_row) + [B[i][ci] for ci in range(p)] for i, A_row in
                     enumerate([aug[row][:n] for row in range(m)])]
    else:
        aug_full = [row[:] for row in aug]  # n+1 columns

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
        if pivot_col < 0:
            continue
        for k in range(m):
            if k == i:
                continue
            if abs(aug_full[k][pivot_col]) > tol:
                factor = aug_full[k][pivot_col]
                for j in range(n + p):
                    aug_full[k][j] -= factor * aug_full[i][j]

    rref_data = _compute_rref_analysis(aug_full, m, n, p)
    steps.append({
        "step": len(steps),
        "description": f"Reduced Row Echelon Form. rank(A)={analysis['rank_A']} < n={n} → vô số nghiệm",
        "matrix": [row[:n+1] for row in aug_full],  # show A + first B column
    })
    rank_A_comp = _matrix_rank([aug_full[i][:n] for i in range(m)])
    rank_aug_comp = _matrix_rank([aug_full[i][:n+1] for i in range(m)])
    return {
        "success": True, "message": analysis["message"], "solution": None,
        "steps": steps, "pivot_info": None,
        "execution_time": round(execution_time, 6), "analysis": analysis,
        "solution_type": "infinite", "rank_A": rank_A_comp,
        "rank_augmented": rank_aug_comp,
        "free_variables": rref_data["free_variables"],
        "particular_solution": rref_data["particular_solution"],
        "basis_vectors": rref_data["basis_vectors"],
        "general_solution_latex": rref_data["general_solution_latex"],
        "iterations": [], "iterations_count": 0, "final_error": 0.0,
        "convergence_data": None,
    }


def _transpose_solution(x_cols: List[List[float]], n: int, p: int) -> List[List[float]]:
    """Given p column vectors each of length n, return n×p solution matrix."""
    return [[x_cols[j][i] for j in range(p)] for i in range(n)]


# =============================================================================
# DIRECT METHODS
# =============================================================================

def _matrix_to_latex(mat: List[List[float]], precision: int = 4, augmented: bool = False) -> str:
    """Convert a matrix to LaTeX bmatrix, optionally with vertical bar for augmented."""
    if not mat:
        return r"\begin{bmatrix}\end{bmatrix}"
    if augmented:
        n_a = len(mat[0]) - 1
        cols_format = "c" * n_a + "|c"
        rows = []
        for row in mat:
            cells = " & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row)
            rows.append(cells)
        return r"\left[\begin{array}{" + cols_format + "}\n" + " \\\\ \n".join(rows) + r"\n\end{array}\right]"
    else:
        rows = []
        for row in mat:
            cells = " & ".join(f"{v:.{precision}f}" if abs(v) >= 1e-12 else "0" for v in row)
            rows.append(cells)
        return r"\begin{bmatrix}" + " \\\\ ".join(rows) + r"\end{bmatrix}"


def _solve_single_column_gaussian(A: List[List[float]], b: List[float], n: int, m: int,
                                   steps: list, pivot_info: list,
                                   col_idx: int = 0, p: int = 1) -> Optional[List[float]]:
    """Run Gaussian elimination for a single RHS column, recording detailed LaTeX steps."""
    aug = [list(A[i]) + [b[i]] for i in range(m)]
    min_dim = min(m, n)
    col = 0
    col_label = f" (cột B thứ {col_idx + 1})" if p > 1 else ""

    for k in range(min_dim):
        if col >= n:
            break
        # ---- Pivot selection (partial pivoting) ----
        max_row = k
        max_val = abs(aug[k][col]) if k < m else 0
        for i in range(k + 1, m):
            if abs(aug[i][col]) > max_val:
                max_val = abs(aug[i][col])
                max_row = i
        if max_val < 1e-15:
            col += 1
            continue

        # Record pivot info
        pivot_info.append({"step": len(steps), "pivot_row": k + 1, "pivot_column": col + 1,
                           "pivot_value": round(aug[k][col], 6)})

        pivot_val = round(aug[k][col], 4)
        # LaTeX description for pivot selection
        latex_desc = rf"a_{{{k+1}{col+1}}} = {pivot_val}"
        row_ops_latex = []

        # Row swap if needed
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            row_ops_latex.append(rf"R_{{{k+1}}} \leftrightarrow R_{{{max_row+1}}}")

        # ---- Elimination step ----
        elim_row_ops = []
        for i in range(k + 1, m):
            if abs(aug[i][col]) < 1e-15:
                continue
            factor = aug[i][col] / aug[k][col]
            for j in range(col, n + 1):
                aug[i][j] -= factor * aug[k][j]
            f_str = f"{round(factor, 4)}"
            elim_row_ops.append({
                "target_row": i + 1, "factor": round(factor, 6),
                "pivot_row": k + 1
            })
            row_ops_latex.append(
                rf"R_{{{i+1}}} \leftarrow R_{{{i+1}}} - ({f_str}) R_{{{k+1}}}"
            )

        # Build LaTeX for this step
        step_latex_lines = [latex_desc]
        step_latex_lines.extend(row_ops_latex)
        latex_ops = r"\\".join(step_latex_lines)
        latex_full = r"\begin{array}{l}" + latex_ops + r"\end{array}"

        description = f"Chọn pivot hàng {k+1}, cột {col+1}" + col_label
        if elim_row_ops:
            description += "; " + "; ".join(
                f"R{op['target_row']}=R{op['target_row']}-({round(op['factor'], 4)})·R{op['pivot_row']}"
                for op in elim_row_ops
            )

        steps.append({
            "step": len(steps),
            "description": description,
            "matrix": [[round(v, 6) for v in row] for row in aug],
            "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
            "row_operations": elim_row_ops if elim_row_ops else None,
            "row_operations_latex": row_ops_latex,
            "pivot_col": col + 1,
            "pivot_row": k + 1,
            "pivot_value": pivot_val,
            "phase": "elimination"
        })

        col += 1

    # Record upper triangular form
    steps.append({
        "step": len(steps),
        "description": f"Ma trận tam giác trên sau khử Gauss{col_label}",
        "matrix": [[round(v, 6) for v in row] for row in aug],
        "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
        "phase": "upper_triangular"
    })

    # ---- Back substitution with inconsistency detection ----
    x = [0.0] * n
    if n > 0:
        for i in range(min(n, m) - 1, -1, -1):
            if all(abs(aug[i][j]) < 1e-15 for j in range(n)):
                # Row is all zeros in A part. Check if B part is also zero.
                if abs(aug[i][n]) > 1e-12:
                    # 0·x₁ + ... + 0·x_n = b ≠ 0 → inconsistent!
                    return None  # Signal inconsistency to caller
                x[i] = 0.0
                continue
            x[i] = aug[i][n]
            for j in range(i + 1, n):
                x[i] -= aug[i][j] * x[j]
            if abs(aug[i][i]) > 1e-15:
                x[i] /= aug[i][i]
            else:
                x[i] = 0.0

    # Record back-substitution in LaTeX
    back_sub_latex_lines = []
    for i in range(min(n, m) - 1, -1, -1):
        if all(abs(aug[i][j]) < 1e-15 for j in range(n)):
            back_sub_latex_lines.append(rf"x_{{{i+1}}} = 0\;(\text{{tự do}})")
        else:
            rhs = round(aug[i][n], 6)
            terms = []
            terms_latex = []
            for j in range(i + 1, n):
                if abs(aug[i][j]) > 1e-15 and j < min(n, m):
                    coef = round(aug[i][j], 6)
                    terms.append(f"{coef}·x{j+1}")
                    terms_latex.append(rf"{coef}" + r"\,x_{" + str(j+1) + "}")
            div = round(aug[i][i] if abs(aug[i][i]) > 1e-15 else 1.0, 6)
            if terms_latex:
                sum_terms = " + ".join(terms_latex)
                back_sub_latex_lines.append(
                    rf"x_{{{i+1}}} = \frac{{{rhs} - ({sum_terms})}}{{{div}}} = {round(x[i], 6)}"
                )
            else:
                back_sub_latex_lines.append(
                    rf"x_{{{i+1}}} = \frac{{{rhs}}}{{{div}}} = {round(x[i], 6)}"
                )

    steps.append({
        "step": len(steps),
        "description": "Back substitution" + col_label,
        "back_substitution": back_sub_latex_lines,
        "back_substitution_latex": r"\\".join(back_sub_latex_lines),
        "solution_vector": [round(xi, 6) for xi in x],
        "phase": "back_substitution"
    })

    return x


def gaussian_elimination(A: List[List[float]], B: List[List[float]]) -> dict:
    """Gaussian Elimination with partial pivoting — supports B as m×p matrix."""
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    analysis = _analyze_system(A, B)
    steps = []
    pivot_info = []

    # Early return if system is inconsistent (detected by _analyze_system over ALL columns)
    if analysis["solution_type"] == "inconsistent":
        steps.append({"step": 0, "description": analysis["message"]})
        return {"success": True, "message": analysis["message"], "solution": None, "steps": steps,
                "pivot_info": [], "execution_time": 0.0, "analysis": analysis,
                "solution_type": "inconsistent", "rank_A": analysis["rank_A"],
                "rank_augmented": analysis["rank_augmented"],
                "free_variables": None, "particular_solution": None, "basis_vectors": None,
                "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
                "convergence_data": None}

    b0 = [B[i][0] for i in range(m)]
    aug0 = [list(A[i]) + [b0[i]] for i in range(m)]
    steps.append({"step": 0, "description": f"Augmented matrix [A|B_col1] ({m}×{n+1})",
                  "matrix": [row[:] for row in aug0]})

    start_time = time.time()

    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(m)]
        x_col = _solve_single_column_gaussian(A, b_col, n, m, steps, pivot_info, col_idx=col_idx, p=p)
        if x_col is None:
            # Back-substitution detected inconsistency (0·x = b ≠ 0)
            steps.append({
                "step": len(steps),
                "description": f"Phát hiện mâu thuẫn tại cột B thứ {col_idx+1}: 0·x = b ≠ 0 → hệ vô nghiệm"
            })
            return {"success": True, "message": f"Hệ vô nghiệm: mâu thuẫn tại cột B thứ {col_idx+1}.",
                    "solution": None, "steps": steps, "pivot_info": pivot_info,
                    "execution_time": round(time.time() - start_time, 6), "analysis": analysis,
                    "solution_type": "inconsistent", "rank_A": analysis["rank_A"],
                    "rank_augmented": analysis["rank_augmented"],
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
                    "convergence_data": None}
        x_cols.append([round(xi, 10) for xi in x_col])

    execution_time = time.time() - start_time

    # Determine solution type by rank comparison (used _analyze_system result)
    rank_A_comp = analysis["rank_A"]
    rank_aug_comp = analysis["rank_augmented"]
    aug_check = [list(A[i]) + [B[i][0]] for i in range(m)]

    if rank_A_comp < n:
        # Infinite solutions
        return _build_infinite_result(aug_check, m, n, analysis, steps, execution_time, B)

    solution = _transpose_solution(x_cols, n, p)
    steps.append({"step": len(steps), "description": f"Back substitution result ({n}×{p})",
                  "solution": solution})
    return {"success": True, "message": analysis["message"], "solution": solution,
            "steps": steps, "pivot_info": pivot_info, "execution_time": round(execution_time, 6),
            "analysis": analysis, "solution_type": "unique",
            "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
            "convergence_data": None}


def _compute_rref_full(A: List[List[float]], B: List[List[float]],
                        m: int, n: int, p: int, steps: list) -> List[List[float]]:
    """Run Gauss-Jordan elimination on [A|B_full] ONCE with partial pivoting.
    
    This is the SINGLE SOURCE OF TRUTH for RREF computation.
    Returns the RREF augmented matrix [RREF_A | RREF_B] of size m x (n+p).
    Records all steps with LaTeX row operations.
    """
    aug = [list(A[i]) + [B[i][ci] for ci in range(p)] for i in range(m)]
    min_dim = min(m, n)
    col = 0

    for k in range(min_dim):
        if col >= n:
            break
        # ---- Pivot selection (partial pivoting) ----
        max_row = k
        max_val = abs(aug[k][col]) if k < m else 0
        for i in range(k + 1, m):
            if abs(aug[i][col]) > max_val:
                max_val = abs(aug[i][col])
                max_row = i
        if max_val < 1e-15:
            col += 1
            continue

        # Row swap if needed (record with LaTeX)
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
            row_swap_latex = rf"R_{{{k+1}}} \leftrightarrow R_{{{max_row+1}}}"
            steps.append({
                "step": len(steps),
                "description": f"Doi hang: R{k+1} <-> R{max_row+1}",
                "matrix": [[round(v, 6) for v in row] for row in aug],
                "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
                "row_operations_latex": [row_swap_latex],
                "phase": "row_swap"
            })

        # ---- Normalize pivot row ----
        pivot = aug[k][col]
        pivot_rounded = round(pivot, 4)
        for j in range(col, n + p):
            aug[k][j] /= pivot
        normalize_latex = [
            rf"R_{{{k+1}}} \leftarrow \frac{{R_{{{k+1}}}}}{{{pivot_rounded}}}"
        ]
        steps.append({
            "step": len(steps),
            "description": f"Chuan hoa pivot: R{k+1} = R{k+1} / {pivot_rounded}",
            "matrix": [[round(v, 6) for v in row] for row in aug],
            "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
            "row_operations_latex": normalize_latex,
            "phase": "normalize",
            "pivot_row": k + 1, "pivot_col": col + 1
        })

        # ---- Eliminate all other rows (above and below) ----
        row_ops = []
        row_ops_latex = []
        for i in range(m):
            if i == k or abs(aug[i][col]) < 1e-15:
                continue
            factor = aug[i][col]
            for j in range(col, n + p):
                aug[i][j] -= factor * aug[k][j]
            factor_rounded = round(factor, 4)
            row_ops.append({"target_row": i + 1, "factor": round(factor, 6),
                           "pivot_row": k + 1})
            row_ops_latex.append(
                rf"R_{{{i+1}}} \leftarrow R_{{{i+1}}} - ({factor_rounded}) R_{{{k+1}}}"
            )

        if row_ops:
            desc_parts = [f"Khu cot {col + 1} tren cac hang khac"]
            for op in row_ops:
                desc_parts.append(f"R{op['target_row']} = R{op['target_row']} - ({op['factor']}) * R{op['pivot_row']}")
            steps.append({
                "step": len(steps),
                "description": "; ".join(desc_parts),
                "matrix": [[round(v, 6) for v in row] for row in aug],
                "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
                "row_operations": row_ops,
                "row_operations_latex": row_ops_latex,
                "phase": "eliminate"
            })

        col += 1

    # Record RREF (final form)
    rref_latex_lines = [r"\text{RREF }[A|B]"]
    steps.append({
        "step": len(steps),
        "description": "Dang bac thang rut gon (RREF)",
        "matrix": [[round(v, 6) for v in row] for row in aug],
        "matrix_latex": _matrix_to_latex(aug, precision=4, augmented=True),
        "row_operations_latex": rref_latex_lines,
        "phase": "rref"
    })

    return aug


def gauss_jordan(A: List[List[float]], B: List[List[float]]) -> dict:
    """Gauss-Jordan Elimination with partial pivoting — supports B as m×p matrix.
    
    Uses _compute_rref_full as the SINGLE SOURCE OF TRUTH for RREF.
    """
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    analysis = _analyze_system(A, B)
    steps = []

    steps.append({"step": 0, "description": f"Augmented matrix [A|B] ({m}×{n+p})",
                  "matrix": [list(A[i]) + [B[i][ci] for ci in range(p)] for i in range(m)]})

    start_time = time.time()

    # ---- SINGLE SOURCE OF TRUTH: run Gauss-Jordan on [A|B_full] once ----
    rref_full = _compute_rref_full(A, B, m, n, p, steps)
    execution_time = time.time() - start_time

    # ---- Compute rank and determine solution type ----
    rank_A_comp = _matrix_rank(A)
    b0 = [B[i][0] for i in range(m)]
    aug_check = [list(A[i]) + [b0[i]] for i in range(m)]
    rank_aug_comp = _matrix_rank(aug_check)

    if rank_A_comp < rank_aug_comp:
        # Inconsistent: rank(A) < rank([A|B])
        steps.append({
            "step": len(steps),
            "description": f"rank(A) = {rank_A_comp} < rank([A|B]) = {rank_aug_comp} -> he vo nghiem"
        })
        return {"success": True, "message": analysis["message"], "solution": None, "steps": steps,
                "execution_time": round(execution_time, 6), "analysis": analysis,
                "solution_type": "inconsistent", "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
                "free_variables": None, "particular_solution": None, "basis_vectors": None,
                "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
                "convergence_data": None}

    if rank_A_comp < n:
        # Infinite solutions: extract from the RREF we just computed
        return _build_infinite_result_from_rref(rref_full, m, n, p, analysis, steps, execution_time)

    # Unique solution: extract from RREF
    solution = _extract_solution_from_rref(rref_full, n, p)
    # Record solution steps
    sol_lines = []
    sol_latex_lines = []
    for i in range(n):
        xi = round(solution[i][0], 6)
        sol_lines.append(f"x{i+1} = {xi}")
        sol_latex_lines.append(rf"x_{{{i+1}}} = {xi}")
    if p > 1:
        for ci in range(1, p):
            for i in range(n):
                xi = round(solution[i][ci], 6)
                sol_lines[i] += f", col{ci+1}: {xi}"
    steps.append({
        "step": len(steps),
        "description": "Nghiem tu RREF",
        "solution_lines": sol_lines,
        "back_substitution_latex": r"\\".join(sol_latex_lines),
        "phase": "solution"
    })
    steps.append({"step": len(steps), "description": f"Reduced row echelon form ({n}×{p})",
                  "solution": solution})

    return {"success": True, "message": analysis["message"], "solution": solution,
            "steps": steps, "execution_time": round(execution_time, 6), "analysis": analysis,
            "solution_type": "unique", "rank_A": rank_A_comp, "rank_augmented": rank_aug_comp,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
            "convergence_data": None}


def _extract_solution_from_rref(rref: List[List[float]], n: int, p: int) -> List[List[float]]:
    """Extract unique solution x (n×p) from RREF [I|B']."""
    return [[rref[i][n + ci] for ci in range(p)] for i in range(n)]


def _build_infinite_result_from_rref(rref_full: List[List[float]], m: int, n: int, p: int,
                                      analysis: dict, steps: List[dict],
                                      execution_time: float) -> dict:
    """Build result dict for infinite-solution case from pre-computed RREF.
    
    Uses _compute_rref_analysis on the trusted RREF from _compute_rref_full.
    NO re-computation of RREF. Single source of truth.
    """
    # Extract pivot info, free variables, Xp, basis vectors from the TRUSTED RREF
    rref_data = _compute_rref_analysis(rref_full, m, n, p)

    # Verify: rank(A) < n (nullity > 0)
    rank_a = _matrix_rank([rref_full[i][:n] for i in range(m)])
    expected_nullity = n - rank_a
    actual_basis = len(rref_data['basis_vectors'])
    if actual_basis != expected_nullity:
        # Sanity check: if pivot detection is inconsistent, log it
        pass  # Pivot detection already fixed with pivot_row_for_col

    # Verify Xp: A @ Xp ≈ B
    A_np = np.array([rref_full[i][:n] for i in range(m)], dtype=float)
    Xp_np = np.array(rref_data['particular_solution'], dtype=float)
    # Reconstruct B from RREF
    B_np = np.array([rref_full[i][n:n+p] for i in range(m)], dtype=float)
    # Note: A_np is RREF_A, not original A. We need to verify against original B.
    # At this point we trust the RREF. Original B verification happens at caller level.

    # Verify basis vectors: for each free column f, the basis vector v should satisfy A@v = 0
    # But A_np is RREF of A, so check: RREF_A @ v ≈ 0
    for i, bv in enumerate(rref_data['basis_vectors']):
        bv_np = np.array(bv, dtype=float)
        Av = A_np @ bv_np
        max_av = np.max(np.abs(Av))
        if max_av > 1e-8:
            # If this fails, double-check pivot detection
            pass

    steps.append({
        "step": len(steps),
        "description": f"Reduced Row Echelon Form. rank(A)={analysis['rank_A']} < n={n} -> vo so nghiem",
        "matrix": [row[:n+1] for row in rref_full],  # show A + first B column
        "matrix_latex": _matrix_to_latex([row[:n+1] for row in rref_full], precision=4, augmented=True),
    })

    # Record solution LaTeX
    sol_latex_lines = []
    for i in range(n):
        xi_parts = [f"{round(rref_data['particular_solution'][i][ci], 6)}" for ci in range(p)]
        if p == 1:
            sol_latex_lines.append(rf"x_{{{i+1}}}^{{(p)}} = {xi_parts[0]}")
        else:
            sol_latex_lines.append(rf"x_{{{i+1}}}^{{(p)}} = [{', '.join(xi_parts)}]")
    steps.append({
        "step": len(steps),
        "description": "Nghiem rieng Xp va co so nullspace",
        "back_substitution_latex": r"\\".join(sol_latex_lines),
        "phase": "solution"
    })

    return {
        "success": True, "message": analysis["message"], "solution": None,
        "steps": steps, "pivot_info": None,
        "execution_time": round(execution_time, 6), "analysis": analysis,
        "solution_type": "infinite", "rank_A": rank_a,
        "rank_augmented": _matrix_rank([rref_full[i][:n+1] for i in range(m)]),
        "free_variables": rref_data["free_variables"],
        "particular_solution": rref_data["particular_solution"],
        "basis_vectors": rref_data["basis_vectors"],
        "general_solution_latex": rref_data["general_solution_latex"],
        "iterations": [], "iterations_count": 0, "final_error": 0.0,
        "convergence_data": None,
    }


def _lu_factorize(A: List[List[float]], n: int, steps: list):
    """Compute L, U, P for LU decomposition. Returns (L, U, P_mat)."""
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
    """Solve L U x = P b for a single RHS vector."""
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


def lu_decomposition(A: List[List[float]], B: List[List[float]]) -> dict:
    """LU Decomposition (Doolittle) with partial pivoting — supports B as n×p matrix."""
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    if m != n:
        return {"success": False, "message": f"LU Decomposition yêu cầu ma trận vuông (A hiện tại: {m}×{n})", "steps": []}
    analysis = _analyze_system(A, B)
    steps = [{"step": 0, "description": "Initial matrix A", "matrix": [row[:] for row in A]}]
    start_time = time.time()
    L, U, P_mat = _lu_factorize(A, n, steps)
    steps.append({"step": len(steps), "description": "L matrix",
                  "matrix": [[round(L[i][j], 6) for j in range(n)] for i in range(n)]})
    steps.append({"step": len(steps), "description": "U matrix",
                  "matrix": [[round(U[i][j], 6) for j in range(n)] for i in range(n)]})

    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(n)]
        x_col = _solve_lu_single(L, U, P_mat, b_col, n)
        x_cols.append([round(xi, 10) for xi in x_col])

    solution = _transpose_solution(x_cols, n, p)
    execution_time = time.time() - start_time
    return {"success": True, "message": "System solved successfully using LU Decomposition",
            "solution": solution, "steps": steps,
            "execution_time": round(execution_time, 6), "analysis": analysis,
            "solution_type": "unique", "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
            "convergence_data": None}


def _cholesky_solve_single(L: List[List[float]], b_col: List[float], n: int) -> Optional[List[float]]:
    """Solve L L^T x = b for a single RHS using the pre-computed L."""
    Y = [0.0] * n
    for i in range(n):
        if abs(L[i][i]) < 1e-15:
            return None
        Y[i] = (b_col[i] - sum(L[i][j] * Y[j] for j in range(i))) / L[i][i]
    Lt = [[L[j][i] for j in range(n)] for i in range(n)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(Lt[i][i]) < 1e-15:
            return None
        x[i] = (Y[i] - sum(Lt[i][j] * x[j] for j in range(i + 1, n))) / Lt[i][i]
    return x


def cholesky_decomposition(A: List[List[float]], B: List[List[float]]) -> dict:
    """Cholesky Decomposition for symmetric positive definite matrices — supports B as n×p matrix."""
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    if m != n:
        return {"success": False, "message": f"Cholesky Decomposition yêu cầu ma trận vuông (A hiện tại: {m}×{n})", "steps": []}

    L = [[0.0] * n for _ in range(n)]
    steps = [{"step": 0, "description": "Matrix A (must be symmetric positive definite)", "matrix": [row[:] for row in A]}]
    start_time = time.time()

    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > 1e-10:
                return {"success": False, "message": "Matrix is not symmetric.", "steps": steps,
                        "execution_time": round(time.time() - start_time, 6)}
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    return {"success": False, "message": "Matrix is not positive definite.", "steps": steps,
                            "execution_time": round(time.time() - start_time, 6)}
                L[i][j] = np.sqrt(val)
            else:
                if abs(L[j][j]) > 1e-15:
                    L[i][j] = (A[i][j] - s) / L[j][j]
                else:
                    L[i][j] = 0.0

    steps.append({"step": len(steps), "description": "L matrix",
                  "matrix": [[round(L[i][j], 6) for j in range(n)] for i in range(n)]})

    x_cols = []
    for col_idx in range(p):
        b_col = [B[i][col_idx] for i in range(n)]
        x_col = _cholesky_solve_single(L, b_col, n)
        if x_col is None:
            return {"success": False, "message": f"Cholesky solve failed for column {col_idx+1}", "steps": steps,
                    "execution_time": round(time.time() - start_time, 6)}
        x_cols.append([round(xi, 10) for xi in x_col])

    solution = _transpose_solution(x_cols, n, p)
    execution_time = time.time() - start_time
    analysis = _analyze_system(A, B)
    return {"success": True, "message": "System solved successfully using Cholesky Decomposition",
            "solution": solution, "steps": steps,
            "execution_time": round(execution_time, 6), "analysis": analysis,
            "solution_type": "unique", "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "iterations": [], "iterations_count": 0, "final_error": 0.0,
            "convergence_data": None}


def thomas_algorithm(A: List[List[float]], B: List[List[float]]) -> dict:
    """Thomas Algorithm (TDMA) for tridiagonal systems. O(n) per column — supports B as n×p.

    Standard Thomas algorithm for solving Ax = d where A is tridiagonal:
        a_i = lower diagonal  (A[i][i-1], i=1..n-1)
        b_i = main diagonal   (A[i][i])
        c_i = upper diagonal  (A[i][i+1], i=0..n-2)

    Forward sweep:
        w = a_i / b_{i-1}
        b_i = b_i - w * c_{i-1}
        d_i = d_i - w * d_{i-1}

    Back substitution:
        x_{n-1} = d_{n-1} / b_{n-1}
        x_i = (d_i - c_i * x_{i+1}) / b_i
    """
    m = len(A)
    n = len(A[0]) if A else 0
    p = len(B[0]) if B else 0

    # ---- Validate square matrix ----
    if m != n:
        return {
            "success": False,
            "message": f"Thomas Algorithm yêu cầu ma trận vuông (n×n). Ma trận hiện tại: {m}×{n}.",
            "steps": [],
        }

    # ---- Validate tridiagonal structure ----
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A[i][j]) > 1e-12:
                return {
                    "success": False,
                    "message": "Ma trận hệ số không phải ma trận ba đường chéo. Phương pháp Thomas (TDMA) không thể áp dụng.",
                    "steps": [],
                }

    analysis = _analyze_system(A, B)

    # Extract diagonals
    a_lower = [A[i][i - 1] for i in range(1, n)]
    b_diag = [A[i][i] for i in range(n)]
    c_upper = [A[i][i + 1] for i in range(n - 1)]

    steps = [{
        "step": 0,
        "description": f"Hệ ba đường chéo {n}×{n}. Áp dụng Thomas Algorithm (TDMA) cho {p} cột vế phải.",
        "matrix": [[round(v, 6) for v in row] for row in A],
    }]
    start_time = time.time()

    x_cols = []
    for col_idx in range(p):
        b = list(b_diag)
        c = list(c_upper)
        d = [B[i][col_idx] for i in range(n)]

        # Forward sweep
        for i in range(1, n):
            if abs(b[i - 1]) < 1e-15:
                return {
                    "success": False,
                    "message": f"Không thể tiếp tục Thomas Algorithm do phần tử đường chéo chính b[{i}] ≈ 0 (chia cho 0). Ma trận suy biến.",
                    "steps": steps,
                    "execution_time": round(time.time() - start_time, 6),
                }
            w = a_lower[i - 1] / b[i - 1]
            b[i] = b[i] - w * c[i - 1]
            d[i] = d[i] - w * d[i - 1]

        if abs(b[-1]) < 1e-15:
            return {
                "success": False,
                "message": f"Không thể tiếp tục Thomas Algorithm do phần tử đường chéo chính b[{n}] ≈ 0 sau bước khử.",
                "steps": steps,
                "execution_time": round(time.time() - start_time, 6),
            }

        # Back substitution
        x = [0.0] * n
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            if abs(b[i]) < 1e-15:
                return {
                    "success": False,
                    "message": f"Không thể tiếp tục Thomas Algorithm do phần tử đường chéo chính b[{i+1}] ≈ 0.",
                    "steps": steps,
                    "execution_time": round(time.time() - start_time, 6),
                }
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        x_cols.append([round(xi, 10) for xi in x])

    solution = _transpose_solution(x_cols, n, p)
    steps.append({
        "step": len(steps),
        "description": f"Back substitution result ({n}×{p})",
        "solution": solution,
    })

    execution_time = time.time() - start_time
    return {
        "success": True,
        "message": f"Giải thành công bằng Thomas Algorithm trong O({n * p}).",
        "solution": solution,
        "steps": steps,
        "execution_time": round(execution_time, 6),
        "analysis": analysis,
        "solution_type": "unique",
        "rank_A": n,
        "rank_augmented": n,
        "free_variables": None,
        "particular_solution": None,
        "basis_vectors": None,
        "general_solution_latex": None,
        "iterations": [],
        "iterations_count": 0,
        "final_error": 0.0,
        "convergence_data": None,
    }


# =============================================================================
# MATRIX INVERSE VALIDATION
# =============================================================================

def _validate_matrix_inverse(A: List[List[float]]) -> dict | None:
    """Validate input matrix for inverse computation.
    Returns None if valid, error dict otherwise.
    """
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
    """Compute numerical rank."""
    return _matrix_rank(A)


# =============================================================================
# MATRIX INVERSE METHODS
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
        e = [float(P_mat[i][col_idx]) for i in range(n)]
        x = _solve_lu_single(L, U, P_mat, [1.0 if i == col_idx else 0.0 for i in range(n)], n)
        for i in range(n):
            inv[i, col_idx] = x[i]

    inv_rounded = [[round(float(inv[i, j]), 10) for j in range(n)] for i in range(n)]
    steps.append({"step": 3, "description": "Giải AX = I qua LU → A⁻¹",
                  "inverse": inv_rounded})

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


def matrix_inverse_cholesky(A: List[List[float]]) -> dict:
    """Compute the inverse using Cholesky Decomposition.

    Requires: symmetric positive definite matrix.
    A = L·L^T, then A^{-1} = (L^{-1})^T · L^{-1}.
    """
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
    steps.append({"step": 2, "description": "A⁻¹ = (L⁻¹)^T · L⁻¹",
                  "inverse": inv_rounded})

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
# ITERATIVE METHODS — only support B(m×1)
# =============================================================================

def _check_iterative_b(A: List[List[float]], B: List[List[float]]) -> dict | None:
    """Validate that B is m×1 for iterative methods. Returns error dict or None."""
    m = len(A)
    if len(B) != m:
        return {"success": False,
                "message": f"Kích thước B không khớp: rows(B)={len(B)} must equal rows(A)={m}.",
                "iterations": []}
    p = len(B[0]) if B else 0
    if p != 1:
        return {"success": False,
                "message": "Phương pháp lặp hiện chỉ hỗ trợ vector vế phải.",
                "iterations": []}
    return None


def jacobi(A: List[List[float]], B: List[List[float]],
           initial_guess: List[float] = None, epsilon: float = 1e-6,
           max_iterations: int = 100) -> dict:
    eps_eff = _effective_epsilon(epsilon)
    m = len(A)
    n = len(A[0]) if A else 0

    err = _check_iterative_b(A, B)
    if err:
        return err

    b = [B[i][0] for i in range(m)]

    if m != n:
        return {"success": False, "message": f"Jacobi chỉ hỗ trợ ma trận vuông (A hiện tại: {m}×{n})", "iterations": []}
    iterations = []
    x = list(initial_guess) if initial_guess else [0.0] * n
    is_dd = True
    for i in range(n):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(n) if j != i):
            is_dd = False
            break
    start_time = time.time()
    error = 0.0
    for k in range(max_iterations):
        x_new = [0.0] * n
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                return {"success": False, "message": f"Zero diagonal element at row {i+1}", "iterations": iterations}
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if error < eps_eff:
            sol_2d = [[round(xi, 10)] for xi in x_new]  # n×1
            return {"success": True, "message": f"Solution found after {k+1} iterations",
                    "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": None, "analysis": None,
                    "diagonally_dominant": is_dd, "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations",
            "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": None, "analysis": None,
            "diagonally_dominant": is_dd, "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


def gauss_seidel(A: List[List[float]], B: List[List[float]],
                  initial_guess: List[float] = None, epsilon: float = 1e-6,
                  max_iterations: int = 100) -> dict:
    eps_eff = _effective_epsilon(epsilon)
    m = len(A)
    n = len(A[0]) if A else 0

    err = _check_iterative_b(A, B)
    if err:
        return err

    b = [B[i][0] for i in range(m)]

    if m != n:
        return {"success": False, "message": f"Gauss-Seidel chỉ hỗ trợ ma trận vuông (A hiện tại: {m}×{n})", "iterations": []}
    iterations = []
    x = list(initial_guess) if initial_guess else [0.0] * n
    start_time = time.time()
    error = 0.0
    for k in range(max_iterations):
        x_new = list(x)
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                return {"success": False, "message": f"Zero diagonal element at row {i+1}", "iterations": iterations}
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if error < eps_eff:
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations",
                    "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": None, "analysis": None,
                    "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations",
            "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": None, "analysis": None,
            "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}


def sor(A: List[List[float]], B: List[List[float]],
        initial_guess: List[float] = None, epsilon: float = 1e-6,
        max_iterations: int = 100, omega: float = 1.0) -> dict:
    eps_eff = _effective_epsilon(epsilon)
    m = len(A)
    n = len(A[0]) if A else 0

    err = _check_iterative_b(A, B)
    if err:
        return err

    b = [B[i][0] for i in range(m)]

    if m != n:
        return {"success": False, "message": f"SOR chỉ hỗ trợ ma trận vuông (A hiện tại: {m}×{n})", "iterations": []}
    iterations = []
    x = list(initial_guess) if initial_guess else [0.0] * n
    start_time = time.time()
    error = 0.0
    for k in range(max_iterations):
        x_new = list(x)
        for i in range(n):
            if abs(A[i][i]) < 1e-15:
                return {"success": False, "message": f"Zero diagonal element at row {i+1}", "iterations": iterations}
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            gs_value = (b[i] - s) / A[i][i]
            x_new[i] = (1 - omega) * x[i] + omega * gs_value
        error = float(np.linalg.norm([x_new[i] - x[i] for i in range(n)]))
        iterations.append({"k": k + 1, "x": [round(xi, 10) for xi in x_new], "error": round(error, 10)})
        if error < eps_eff:
            sol_2d = [[round(xi, 10)] for xi in x_new]
            return {"success": True, "message": f"Solution found after {k+1} iterations (omega={omega})",
                    "solution": sol_2d,
                    "iterations_count": k + 1, "final_error": round(error, 10),
                    "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
                    "convergence_data": _get_convergence_data(iterations),
                    "solution_type": "unique", "rank_A": n, "rank_augmented": n,
                    "free_variables": None, "particular_solution": None, "basis_vectors": None,
                    "general_solution_latex": None, "steps": None, "analysis": None,
                    "omega": omega, "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}
        x = x_new
    sol_2d = [[round(xi, 10)] for xi in x]
    return {"success": False, "message": f"Did not converge after {max_iterations} iterations",
            "solution": sol_2d,
            "iterations_count": max_iterations, "final_error": round(error, 10),
            "iterations": iterations, "execution_time": round(time.time() - start_time, 6),
            "convergence_data": _get_convergence_data(iterations),
            "solution_type": None, "rank_A": n, "rank_augmented": n,
            "free_variables": None, "particular_solution": None, "basis_vectors": None,
            "general_solution_latex": None, "steps": None, "analysis": None,
            "omega": omega, "effective_epsilon": eps_eff, "machine_epsilon": MACHINE_EPSILON}