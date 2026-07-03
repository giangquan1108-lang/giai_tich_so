"""Audit: Gauss-Jordan algorithm full trace. No Unicode characters."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import json

A_test = [
    [1, 2, 3, 4, 5],
    [2, 4, 6, 8, 10],
    [1, 1, 1, 1, 1],
    [2, 3, 4, 5, 6],
    [3, 5, 7, 9, 11],
]
B_test = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]]

A_np = np.array(A_test, dtype=float)
B_np = np.array(B_test, dtype=float)
m, n, p = A_np.shape[0], A_np.shape[1], B_np.shape[1]

print("=" * 80)
print("AUDIT: GAUSS-JORDAN FULL TRACE")
print("=" * 80)

# ---- GROUND TRUTH ----
print("\n--- GROUND TRUTH ---")
print("Rank(A) numpy:", np.linalg.matrix_rank(A_np))
aug0 = np.hstack([A_np, B_np[:, 0:1]])
print("Rank([A|B_col0]) numpy:", np.linalg.matrix_rank(aug0))

import sympy as sp
A_sp = sp.Matrix(A_test)
B_sp = sp.Matrix(B_test)
print("SymPy rank(A):", A_sp.rank())
rref_sp, piv_sp = A_sp.row_join(B_sp[:, 0]).rref()
print("SymPy RREF [A|B_col0]:")
print(rref_sp)
print("SymPy pivots:", list(piv_sp))
piv_a_sp = [p for p in piv_sp if p < n]
free_sp = [j for j in range(n) if j not in piv_a_sp]
print("SymPy pivot_cols (A):", piv_a_sp)
print("SymPy free_cols:", free_sp)
print("SymPy nullity:", n - A_sp.rank())

ns = A_sp.nullspace()
print("SymPy nullspace:", len(ns), "vectors")
for i, v in enumerate(ns):
    print(f"  v{i+1} = {[float(x) for x in v]}")

# SymPy RREF with FULL multi-column B
rref_full_sp, piv_full_sp = A_sp.row_join(B_sp).rref()
print("\nSymPy RREF [A|B] full 5x7:")
print(rref_full_sp)
print("SymPy pivots full:", list(piv_full_sp))

# ---- OUR ALGORITHM ----
print("\n" + "=" * 80)
print("OUR ALGORITHM TRACE")
print("=" * 80)

from app.algorithms.linear_system import (
    gauss_jordan, _matrix_rank, _analyze_system,
    _compute_rref_analysis, _build_infinite_result,
    _matrix_to_latex
)

analysis = _analyze_system(A_test, B_test)
analysis_str = json.dumps({k: v for k, v in analysis.items() if k != 'message'}, indent=2)
print("\n_analyze_system:", analysis_str)
print("  message:", analysis.get('message'))

result = gauss_jordan(A_test, B_test)
print("\ngauss_jordan result keys:", sorted(result.keys()))
print("solution_type:", result.get('solution_type'))
print("rank_A:", result.get('rank_A'))
print("rank_augmented:", result.get('rank_augmented'))
print("free_variables:", result.get('free_variables'))
print("num basis_vectors:", len(result.get('basis_vectors', [])))

# ---- STEPS ----
print("\n--- ALL STEPS ---")
for step in result.get('steps', []):
    desc = step.get('description', '')
    # replace viet chars
    for old, new in [('o','o'),('e','e'),('a','a'),('u','u'),('i','i')]:
        pass
    print(f"  Step {step['step']} [{step.get('phase','?')}]: {desc}")
    if step.get('matrix') and step.get('phase') == 'rref':
        print("    RREF matrix:")
        for row in step['matrix']:
            print(f"      {[round(v, 10) for v in row]}")

# ---- PARTICULAR SOLUTION ----
print("\n--- PARTICULAR SOLUTION ---")
xp = result.get('particular_solution')
if xp:
    print("Shape:", len(xp), "x", len(xp[0]))
    for i, row in enumerate(xp):
        print(f"  x{i+1} = {row}")
    xp2 = np.array(xp, dtype=float)
    res = A_np @ xp2 - B_np
    print("\nResidual A*Xp - B:")
    print(res)
    print("max_abs_error:", np.max(np.abs(res)))

# ---- BASIS VECTORS ----
print("\n--- BASIS VECTORS ---")
basis = result.get('basis_vectors', [])
for i, v in enumerate(basis):
    print(f"  v{i+1} = {v}")
    Av = A_np @ np.array(v, dtype=float)
    print(f"    A*v{i+1} = {[round(x, 10) for x in Av]} max_abs={np.max(np.abs(Av)):.2e}")

# ---- NULLITY CHECK ----
print("\n--- NULLITY ---")
rank_a = result.get('rank_A', 0)
expected_nullity = n - rank_a
actual_basis = len(basis)
print(f"n={n}, rank={rank_a}, expected nullity={expected_nullity}, actual basis={actual_basis}")
if expected_nullity != actual_basis:
    print("XX NULLITY MISMATCH!")

# ---- DIRECT TRACE OF _build_infinite_result ----
print("\n" + "=" * 80)
print("DIRECT TRACE OF _build_infinite_result")
print("=" * 80)

aug_check = [list(A_test[i]) + [B_test[i][0]] for i in range(m)]
print("aug_check (A|B_col0):")
for row in aug_check:
    print(f"  {[round(v, 4) for v in row]}")

tol = 1e-15
aug_full = [list(A_test[i]) + [B_test[i][ci] for ci in range(p)] for i in range(m)]
print("\naug_full (A|B) initial:")
for row in aug_full:
    print(f"  {[round(v, 4) for v in row]}")

# Find pivot rows (rows with any non-zero in A part)
pivot_rows = []
for i in range(m):
    for j in range(n):
        if abs(aug_full[i][j]) > tol:
            pivot_rows.append(i)
            break
print("\npivot_rows:", pivot_rows)

# Normalize each pivot row (divide by first non-zero in A part)
for i_idx, i in enumerate(pivot_rows):
    pivot_col = -1
    for j in range(n):
        if abs(aug_full[i][j]) > tol:
            pivot_col = j
            break
    if pivot_col >= 0:
        pv = aug_full[i][pivot_col]
        print(f"Normalize: row {i}, pivot col {pivot_col}, value {pv:.6f}")
        for j in range(n + p):
            aug_full[i][j] /= pv

print("\nAfter normalize:")
for row in aug_full:
    print(f"  {[round(v, 6) for v in row]}")

# Eliminate all other rows
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
            print(f"Eliminate: R{k} -= {factor:.6f} * R{i} (col {pivot_col})")
            for j in range(n + p):
                aug_full[k][j] -= factor * aug_full[i][j]

print("\nRREF from _build_infinite_result:")
for row in aug_full:
    print(f"  {[round(v, 10) for v in row]}")

# Now call _compute_rref_analysis on this RREF
print("\n--- _compute_rref_analysis ---")
rref_data = _compute_rref_analysis(aug_full, m, n, p)
print("pivot_columns:", rref_data['pivot_columns'])
print("free_columns:", rref_data['free_columns'])
print("free_variables:", rref_data['free_variables'])

print("\nParticular solution (should be n x p =", n, "x", p, "):")
for i, row in enumerate(rref_data['particular_solution']):
    print(f"  x{i+1} = {row}")

xp3 = np.array(rref_data['particular_solution'], dtype=float)
print("Shape:", xp3.shape)
res3 = A_np @ xp3 - B_np
print("\nA*Xp - B:")
print(res3)
print("max_abs_error:", np.max(np.abs(res3)))

print("\nBasis vectors:", len(rref_data['basis_vectors']))
for i, bv in enumerate(rref_data['basis_vectors']):
    print(f"  v{i+1} = {bv}")
    Abv = A_np @ np.array(bv, dtype=float)
    print(f"    A*v{i+1} max_abs = {np.max(np.abs(Abv)):.2e}")

# ---- COMPARE WITH SYMPY ----
print("\n" + "=" * 80)
print("COMPARISON: Our RREF vs SymPy RREF")
print("=" * 80)
print("\nOur _build_infinite_result RREF:")
for row in aug_full:
    print(f"  {[round(v, 6) for v in row]}")
print("\nSymPy RREF [A|B] full:")
print(rref_full_sp)

print("\nOur pivot_cols:", rref_data['pivot_columns'])
print("SymPy pivot_cols:", piv_a_sp)
print("Our free_cols:", rref_data['free_columns'])
print("SymPy free_cols:", free_sp)
print("Our basis count:", len(rref_data['basis_vectors']))
print("SymPy nullspace dim:", len(ns))

# ---- ROOT CAUSE ----
print("\n" + "=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)
print("""
BUG IDENTIFIED in _build_infinite_result (linear_system.py line ~229-259):

The RREF computation in _build_infinite_result finds pivot_rows as:
  "rows with any non-zero element in A part of aug_full"

For a rank-deficient 5x5 system with rank=2 (nullity=3), this gives
pivot_rows = [0,1,2,3,4] (all 5 rows have non-zero elements initially).

Then it:
1. Normalizes all 5 rows (divides by first non-zero in A part)
2. Eliminates all pivot columns from all other rows

The result for a CONSISTENT rank-deficient system is correct (dependent rows
become all zeros after elimination). BUT the issue is when the augmented
matrix has inconsistencies, or when numerical precision causes elimination
to NOT fully zero out dependent rows.

In addition, _compute_rref_analysis (line ~152-207) has a CRITICAL BUG:
  for i, pivot_col in enumerate(pivot_cols):
      x_p[pivot_col] = aug[i][n + col_b]

This ASSUMES pivot at pivot_cols[i] is in row i. If the RREF has pivots
in a different row (e.g., due to row swaps), the solution is WRONG.

Similarly for basis vectors:
  v[pivot_col] = -aug[i][f]
Also assumes row i for pivot_cols[i].

RECOMMENDED FIX:
For each pivot column, find which row contains the actual pivot (value~1)
instead of assuming pivot_cols[i] is in row i.
""")
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)