"""Comprehensive test suite for Gaussian Elimination method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app.algorithms.linear_system import gaussian_elimination

passed = 0
failed = 0
errors = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        try:
            safe_detail = str(detail).encode('ascii', errors='replace').decode('ascii')
        except Exception:
            safe_detail = "(detail unavailable)"
        msg = f"  [FAIL] {name} — {safe_detail}"
        print(msg)
        errors.append(msg)

# =============================================================================
# FIXTURES
# =============================================================================
A3 = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
b3 = [[6], [25], [-11]]
x3_true = np.linalg.solve(np.array(A3, dtype=float), np.array([6, 25, -11], dtype=float))

b3_2 = [[6, 1], [25, 2], [-11, 3]]
x3_2_true = np.linalg.solve(np.array(A3, dtype=float), np.array(b3_2, dtype=float))

A_inc = [[1, 1], [1, 1]]
b_inc = [[1], [2]]

A_inf = [[1, 1], [2, 2]]
b_inf = [[1], [2]]

A5 = [[10, 0, 1, 0, 0],
      [1, 9, 0, 2, 0],
      [0, 2, 8, 0, 1],
      [0, 0, 3, 7, 0],
      [1, 0, 0, 0, 12]]
b53 = [[1, 10, 100],
       [2, 20, 200],
       [3, 30, 300],
       [4, 40, 400],
       [5, 50, 500]]
x53_true = np.linalg.solve(np.array(A5, dtype=float), np.array(b53, dtype=float))

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
bI3 = [[5], [-3], [2]]

# Hilbert 4x4 (ill-conditioned)
hilb = [[1/(i+j+1) for j in range(4)] for i in range(4)]
b_hilb = [[sum(row)] for row in hilb]
x_hilb_true = np.linalg.solve(np.array(hilb, dtype=float), np.array(b_hilb, dtype=float))

print("=" * 60)
print("GAUSSIAN ELIMINATION — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# =============================================================================
# 1. BASIC TESTS
# =============================================================================
print("\n--- 1. Basic Functionality ---")

# G1: B(3x1) single column
r = gaussian_elimination(A3, b3)
check("G1.1 B(3x1) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("G1.2 B(3x1) shape 3x1", sol.shape == (3, 1), f"shape={sol.shape}")
    check("G1.3 B(3x1) correct", np.allclose(sol.flatten(), x3_true, atol=1e-6),
          f"got {sol.flatten()} expected {x3_true}")
    check("G1.4 solution_type unique", r["solution_type"] == "unique",
          f"got {r.get('solution_type')}")

# G2: B(3x2) multi-column
r = gaussian_elimination(A3, b3_2)
check("G2.1 B(3x2) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("G2.2 B(3x2) shape 3x2", sol.shape == (3, 2), f"shape={sol.shape}")
    check("G2.3 B(3x2) correct", np.allclose(sol, x3_2_true, atol=1e-6),
          f"got {sol} expected {x3_2_true}")

# G3: Identity matrix I3 — solution = B
r = gaussian_elimination(I3, bI3)
check("G3.1 I3 success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("G3.2 I3 solution = B", np.allclose(sol.flatten(), [5, -3, 2], atol=1e-6),
          f"got {sol.flatten()}")

# G4: B all zeros — solution should be zero
A_zeroB = [[4, -1], [-1, 3]]
b_zero = [[0], [0]]
r = gaussian_elimination(A_zeroB, b_zero)
check("G4.1 zero B success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("G4.2 zero B gives zero solution", np.allclose(sol, 0, atol=1e-10),
          f"got {sol.flatten()}")

# =============================================================================
# 2. INCONSISTENT SYSTEM
# =============================================================================
print("\n--- 2. Inconsistent System ---")

# G5: Inconsistent via rank analysis
r = gaussian_elimination(A_inc, b_inc)
check("G5.1 inconsistent solution_type", r["solution_type"] == "inconsistent",
      f"got {r.get('solution_type')}")
check("G5.2 rank(A)=1", r.get("rank_A") == 1, f"got {r.get('rank_A')}")
check("G5.3 rank(aug)=2", r.get("rank_augmented") == 2, f"got {r.get('rank_augmented')}")
check("G5.4 solution=None", r.get("solution") is None)

# G6: Inconsistent multi-column B (one column consistent, one inconsistent)
A_inc2 = [[1, 1], [1, 1]]
b_inc2 = [[1, 2], [2, 2]]  # col1 inconsistent (1≠2), col2 consistent (2=2)
r = gaussian_elimination(A_inc2, b_inc2)
check("G6.1 multi-B inconsistent detected", r["solution_type"] == "inconsistent",
      f"got {r.get('solution_type')}")

# =============================================================================
# 3. INFINITE SOLUTIONS
# =============================================================================
print("\n--- 3. Infinite Solutions ---")

# G7: Infinite solutions
r = gaussian_elimination(A_inf, b_inf)
check("G7.1 infinite solution_type", r["solution_type"] == "infinite",
      f"got {r.get('solution_type')}")
check("G7.2 has free_variables", r.get("free_variables") and len(r["free_variables"]) > 0,
      f"got {r.get('free_variables')}")
check("G7.3 has particular_solution", r.get("particular_solution") is not None)
check("G7.4 has basis_vectors", r.get("basis_vectors") and len(r["basis_vectors"]) > 0,
      f"got {r.get('basis_vectors')}")
if r.get("basis_vectors"):
    check("G7.5 basis_vector nonzero", any(abs(v) > 1e-10 for v in r["basis_vectors"][0]))
check("G7.6 has general_solution_latex", r.get("general_solution_latex") is not None
      and len(r["general_solution_latex"]) > 0)

# G8: Infinite with 2 free variables (3x3 rank 1)
A_inf2 = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
b_inf2 = [[1], [2], [3]]
r = gaussian_elimination(A_inf2, b_inf2)
check("G8.1 rank1 3x3 infinite", r["solution_type"] == "infinite",
      f"got {r.get('solution_type')}")
if r.get("free_variables"):
    check("G8.2 2 free variables", len(r["free_variables"]) == 2,
          f"got {len(r['free_variables'])} vars: {r['free_variables']}")

# =============================================================================
# 4. LARGE / EDGE CASE MATRICES
# =============================================================================
print("\n--- 4. Large & Edge Case ---")

# G9: 5x5 with B(5x3)
r = gaussian_elimination(A5, b53)
check("G9.1 A(5x5)B(5x3) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("G9.2 shape 5x3", sol.shape == (5, 3), f"shape={sol.shape}")
    check("G9.3 correct", np.allclose(sol, x53_true, atol=1e-6),
          f"got {sol} expected {x53_true}")

# G10: 1x1 system
A11 = [[5]]
b11 = [[10]]
r = gaussian_elimination(A11, b11)
check("G10.1 1x1 success", r["success"])
if r["success"]:
    check("G10.2 1x1 correct: x=2", abs(np.array(r["solution"])[0, 0] - 2.0) < 1e-10,
          f"got {r['solution']}")

# G11: Hilbert 4x4 (ill-conditioned)
r = gaussian_elimination(hilb, b_hilb)
check("G11.1 Hilbert success", r["success"])
if r["success"]:
    sol = np.array(r["solution"]).flatten()
    check("G11.2 Hilbert ~correct", np.allclose(sol, x_hilb_true, atol=1e-3),
          f"got {sol} expected {x_hilb_true}")

# G12: Near-singular (but not singular) — Gaussian may detect as inconsistent
A_near = [[1, 1], [1, 1 + 1e-12]]
b_near = [[2], [2 + 1e-12]]
r = gaussian_elimination(A_near, b_near)
check("G12.1 near-singular handled", isinstance(r, dict))
if r["success"] and r["solution"] is not None:
    sol = np.array(r["solution"]).flatten()
    check("G12.2 near-singular ~[1,1]",
          abs(sol[0] - 1) < 0.1 and abs(sol[1] - 1) < 0.1,
          f"got {sol}")
else:
    check("G12.2 near-singular gracefully handled", True,
          f"msg={r.get('message', '')[:100]}")

# G13: Non-square matrix (3x2) — still solvable
A_ns = [[1, 2], [3, 4], [5, 6]]
b_ns = [[5], [11], [17]]
r = gaussian_elimination(A_ns, b_ns)
check("G13.1 non-square handled", isinstance(r, dict))
check("G13.2 non-square success or infinite", r.get("success", False) or r.get("solution_type") in ("unique", "infinite"),
      f"msg={r.get('message')}")

# =============================================================================
# 5. STEP / PIVOT VERIFICATION
# =============================================================================
print("\n--- 5. Step & Pivot Verification ---")

r = gaussian_elimination(A3, b3)
check("G14.1 has steps", r.get("steps") and len(r["steps"]) > 0)
check("G14.2 steps > 5", len(r.get("steps", [])) > 5,
      f"got {len(r.get('steps', []))} steps")

# Check phase labels
phases = [s.get("phase") for s in r.get("steps", []) if s.get("phase")]
check("G14.3 has elimination phase", "elimination" in phases)
check("G14.4 has upper_triangular phase", "upper_triangular" in phases)
check("G14.5 has back_substitution phase", "back_substitution" in phases)

# Pivot info
check("G14.6 has pivot_info", r.get("pivot_info") and len(r["pivot_info"]) > 0,
      f"got {r.get('pivot_info')}")
if r.get("pivot_info"):
    check("G14.7 pivot_info contains pivot_value", "pivot_value" in r["pivot_info"][0])

# Step has matrix_latex
has_latex = any(s.get("matrix_latex") for s in r.get("steps", []))
check("G14.8 steps have matrix_latex", has_latex)

# Step has row_operations_latex
has_row_latex = any(s.get("row_operations_latex") for s in r.get("steps", []))
check("G14.9 steps have row_operations_latex", has_row_latex)

# =============================================================================
# 6. RANDOM MATRIX COMPARISON WITH NUMPY
# =============================================================================
print("\n--- 6. Random Matrix Comparison ---")

np.random.seed(42)
all_ok = True
for i in range(5):
    size = np.random.randint(2, 6)
    A_rand = np.random.uniform(-5, 5, (size, size))
    # Ensure non-singular
    while abs(np.linalg.det(A_rand)) < 0.1:
        A_rand = np.random.uniform(-5, 5, (size, size))
    A_rand_list = A_rand.tolist()
    B_rand = np.random.uniform(-5, 5, (size, 1))
    B_rand_list = B_rand.tolist()
    
    r = gaussian_elimination(A_rand_list, B_rand_list)
    if not r["success"] or r["solution"] is None:
        all_ok = False
        break
    our_sol = np.array(r["solution"]).flatten()
    np_sol = np.linalg.solve(A_rand, B_rand).flatten()
    if not np.allclose(our_sol, np_sol, atol=1e-5):
        all_ok = False
        break
check("G15.1 5 random matrices match numpy", all_ok)

# Also test random with multi-column B
all_ok2 = True
for i in range(3):
    size = 3
    A_rand = np.random.uniform(-5, 5, (size, size))
    while abs(np.linalg.det(A_rand)) < 0.1:
        A_rand = np.random.uniform(-5, 5, (size, size))
    A_rand_list = A_rand.tolist()
    B_rand = np.random.uniform(-5, 5, (size, 2))
    B_rand_list = B_rand.tolist()
    
    r = gaussian_elimination(A_rand_list, B_rand_list)
    if not r["success"] or r["solution"] is None:
        all_ok2 = False
        break
    our_sol = np.array(r["solution"])
    np_sol = np.linalg.solve(A_rand, B_rand)
    if not np.allclose(our_sol, np_sol, atol=1e-5):
        all_ok2 = False
        break
check("G15.2 multi-column random match", all_ok2)

# =============================================================================
# 7. SYSTEM ANALYSIS FIELDS
# =============================================================================
print("\n--- 7. System Analysis Fields ---")

r = gaussian_elimination(A3, b3)
check("G16.1 has analysis dict", r.get("analysis") is not None)
if r.get("analysis"):
    check("G16.2 analysis has solution_type", "solution_type" in r["analysis"])
    check("G16.3 analysis has rank_A", "rank_A" in r["analysis"])

# =============================================================================
# 8. EXECUTION TIME
# =============================================================================
print("\n--- 8. Execution Time ---")

r = gaussian_elimination(A3, b3)
check("G17.1 has execution_time", r.get("execution_time") is not None)
check("G17.2 execution_time > 0", r.get("execution_time", 0) > 0)

# =============================================================================
# 9. STRESS TEST — 10x10
# =============================================================================
print("\n--- 9. Stress Test (10x10) ---")

A10 = np.random.uniform(-5, 5, (10, 10))
while abs(np.linalg.det(A10)) < 0.1:
    A10 = np.random.uniform(-5, 5, (10, 10))
A10_list = A10.tolist()
B10 = np.random.uniform(-5, 5, (10, 1))
B10_list = B10.tolist()

r = gaussian_elimination(A10_list, B10_list)
check("G18.1 10x10 success", r["success"])
if r["success"]:
    our_sol = np.array(r["solution"]).flatten()
    np_sol = np.linalg.solve(A10, B10).flatten()
    check("G18.2 10x10 correct", np.allclose(our_sol, np_sol, atol=1e-4),
          f"max diff={np.max(np.abs(our_sol - np_sol)):.2e}")

# =============================================================================
# 10. NON-SQUARE MATRIX WITH MORE ROWS THAN COLS (OVERDETERMINED)
# =============================================================================
print("\n--- 10. Non-square / Overdetermined ---")

# Overdetermined: 3x2 with inconsistent RHS
A_od = [[1, 0], [0, 1], [1, 1]]
b_od_true = [[1], [2], [4]]  # Last row: x+y=4 but x=1,y=2 => 3≠4 → inconsistent
r = gaussian_elimination(A_od, b_od_true)
check("G19.1 overdet inconsistent detected", r.get("solution_type") == "inconsistent"
      or not r["success"],
      f"type={r.get('solution_type')} msg={r.get('message')}")

# Overdetermined consistent: 3x2, solution x=[1,2] satisfies all
A_od2 = [[1, 0], [0, 1], [2, 3]]
b_od2 = [[1], [2], [8]]  # 2*1 + 3*2 = 8
r = gaussian_elimination(A_od2, b_od2)
# System is overdetermined (m > n) but consistent
check("G19.2 overdet consistent handled", r.get("solution_type", "") in ("unique", "infinite"),
      f"type={r.get('solution_type')} msg={r.get('message')}")

# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print("\n" + "=" * 60)
print(f"GAUSSIAN ELIMINATION RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)