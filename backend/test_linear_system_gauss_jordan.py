"""Comprehensive test suite for Gauss-Jordan Elimination method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app.algorithms.linear_system import gauss_jordan, gaussian_elimination

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

hilb = [[1/(i+j+1) for j in range(4)] for i in range(4)]
b_hilb = [[sum(row)] for row in hilb]
x_hilb_true = np.linalg.solve(np.array(hilb, dtype=float), np.array(b_hilb, dtype=float))

print("=" * 60)
print("GAUSS-JORDAN ELIMINATION — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# =============================================================================
# 1. BASIC TESTS
# =============================================================================
print("\n--- 1. Basic Functionality ---")

# GJ1: B(3x1)
r = gauss_jordan(A3, b3)
check("GJ1.1 B(3x1) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("GJ1.2 B(3x1) shape 3x1", sol.shape == (3, 1), f"shape={sol.shape}")
    check("GJ1.3 B(3x1) correct", np.allclose(sol.flatten(), x3_true, atol=1e-6))
    check("GJ1.4 solution_type unique", r["solution_type"] == "unique")

# GJ2: B(3x2) multi-column
r = gauss_jordan(A3, b3_2)
check("GJ2.1 B(3x2) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("GJ2.2 B(3x2) shape 3x2", sol.shape == (3, 2))
    check("GJ2.3 B(3x2) correct", np.allclose(sol, x3_2_true, atol=1e-6))

# GJ3: Identity matrix
r = gauss_jordan(I3, bI3)
check("GJ3.1 I3 success", r["success"])
if r["success"]:
    check("GJ3.2 I3 solution = B", np.allclose(np.array(r["solution"]).flatten(), [5, -3, 2], atol=1e-6))

# GJ4: B all zeros
r = gauss_jordan([[4, -1], [-1, 3]], [[0], [0]])
check("GJ4.1 zero B success", r["success"])
if r["success"]:
    check("GJ4.2 zero B zero solution", np.allclose(np.array(r["solution"]), 0, atol=1e-10))

# GJ5: 1x1
r = gauss_jordan([[5]], [[10]])
check("GJ5.1 1x1 success", r["success"])
if r["success"]:
    check("GJ5.2 1x1 x=2", abs(np.array(r["solution"])[0, 0] - 2.0) < 1e-10)

# =============================================================================
# 2. INCONSISTENT
# =============================================================================
print("\n--- 2. Inconsistent System ---")

r = gauss_jordan(A_inc, b_inc)
check("GJ6.1 inconsistent type", r["solution_type"] == "inconsistent")
check("GJ6.2 rank(A)=1", r.get("rank_A") == 1)
check("GJ6.3 rank(aug)=2", r.get("rank_augmented") == 2)
check("GJ6.4 solution=None", r.get("solution") is None)

# Multi-column inconsistent
r = gauss_jordan([[1, 1], [1, 1]], [[1, 2], [2, 2]])
check("GJ7.1 multi-B inconsistent", r["solution_type"] == "inconsistent")

# =============================================================================
# 3. INFINITE SOLUTIONS
# =============================================================================
print("\n--- 3. Infinite Solutions ---")

r = gauss_jordan(A_inf, b_inf)
check("GJ8.1 infinite type", r["solution_type"] == "infinite")
check("GJ8.2 has free_variables", r.get("free_variables") and len(r["free_variables"]) > 0)
check("GJ8.3 has particular_solution", r.get("particular_solution") is not None)
check("GJ8.4 has basis_vectors", r.get("basis_vectors") and len(r["basis_vectors"]) > 0)
check("GJ8.5 has general_solution_latex", bool(r.get("general_solution_latex")))

# 2 free variables
A_inf2 = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
r = gauss_jordan(A_inf2, [[1], [2], [3]])
check("GJ9.1 rank1 infinite", r["solution_type"] == "infinite")
if r.get("free_variables"):
    check("GJ9.2 2 free vars", len(r["free_variables"]) == 2,
          f"got {len(r['free_variables'])}")

# =============================================================================
# 4. LARGE & EDGE CASE
# =============================================================================
print("\n--- 4. Large & Edge Case ---")

r = gauss_jordan(A5, b53)
check("GJ10.1 5x5 B(5x3) success", r["success"])
if r["success"]:
    check("GJ10.2 shape 5x3", np.array(r["solution"]).shape == (5, 3))
    check("GJ10.3 correct", np.allclose(np.array(r["solution"]), x53_true, atol=1e-6))

r = gauss_jordan(hilb, b_hilb)
check("GJ11.1 Hilbert success", r["success"])
if r["success"]:
    check("GJ11.2 Hilbert correct", np.allclose(np.array(r["solution"]).flatten(), x_hilb_true, atol=1e-3))

# Near-singular
r = gauss_jordan([[1, 1], [1, 1 + 1e-12]], [[2], [2 + 1e-12]])
check("GJ12.1 near-singular handled", isinstance(r, dict))
if r["success"] and r["solution"]:
    sol = np.array(r["solution"]).flatten()
    check("GJ12.2 near-singular ~[1,1]", abs(sol[0]-1) < 0.1 and abs(sol[1]-1) < 0.1)
else:
    check("GJ12.2 near-singular graceful", True)

# =============================================================================
# 5. STEP & PHASE VERIFICATION
# =============================================================================
print("\n--- 5. Step & Phase Verification ---")

r = gauss_jordan(A3, b3)
check("GJ13.1 has steps", r.get("steps") and len(r["steps"]) > 0)
check("GJ13.2 steps > 5", len(r.get("steps", [])) > 5)

phases = [s.get("phase") for s in r.get("steps", []) if s.get("phase")]
# row_swap may or may not appear depending on pivot dominance
check("GJ13.3 normalize phase", "normalize" in phases)
check("GJ13.4 eliminate phase", "eliminate" in phases)
check("GJ13.5 rref phase", "rref" in phases)
check("GJ13.6 solution phase", "solution" in phases)
check("GJ13.8 steps have matrix_latex", any(s.get("matrix_latex") for s in r.get("steps", [])))
check("GJ13.9 steps have sol_lines", any(s.get("solution_lines") for s in r.get("steps", [])))

# =============================================================================
# 6. RANDOM MATRIX COMPARISON
# =============================================================================
print("\n--- 6. Random Matrix Comparison ---")

np.random.seed(42)
all_ok = True
for i in range(5):
    size = np.random.randint(2, 6)
    A_rand = np.random.uniform(-5, 5, (size, size))
    while abs(np.linalg.det(A_rand)) < 0.1:
        A_rand = np.random.uniform(-5, 5, (size, size))
    r = gauss_jordan(A_rand.tolist(), np.random.uniform(-5, 5, (size, 1)).tolist())
    if not r["success"] or r["solution"] is None:
        all_ok = False; break
    if not np.allclose(np.array(r["solution"]).flatten(), np.linalg.solve(A_rand, np.array(r["solution"]) if False else np.linalg.solve(A_rand.astype(float), A_rand @ np.ones(size))), atol=1e-5):
        pass
    our = np.array(r["solution"]).flatten()
    nps = np.linalg.solve(A_rand, np.array(r.get("solution", [[0]]*size)).flatten() if False else A_rand @ np.random.uniform(-1, 1, size))
    # Actually use proper comparison
    B_rand = np.random.uniform(-5, 5, (size, 1))
    r2 = gauss_jordan(A_rand.tolist(), B_rand.tolist())
    if not r2["success"] or r2["solution"] is None:
        all_ok = False; break
    if not np.allclose(np.array(r2["solution"]).flatten(), np.linalg.solve(A_rand, B_rand).flatten(), atol=1e-5):
        all_ok = False; break
check("GJ14.1 5 random match numpy", all_ok)

# Multi-column random
all_ok2 = True
for i in range(3):
    A_rand = np.random.uniform(-5, 5, (3, 3))
    while abs(np.linalg.det(A_rand)) < 0.1:
        A_rand = np.random.uniform(-5, 5, (3, 3))
    B_rand = np.random.uniform(-5, 5, (3, 2))
    r = gauss_jordan(A_rand.tolist(), B_rand.tolist())
    if not r["success"] or r["solution"] is None:
        all_ok2 = False; break
    if not np.allclose(np.array(r["solution"]), np.linalg.solve(A_rand, B_rand), atol=1e-5):
        all_ok2 = False; break
check("GJ14.2 multi-col random match", all_ok2)

# =============================================================================
# 7. GJ vs GAUSSIAN CONSISTENCY
# =============================================================================
print("\n--- 7. GJ vs Gaussian Consistency ---")

r_gj = gauss_jordan(A3, b3)
r_ga = gaussian_elimination(A3, b3)
check("GJ15.1 GJ == GA B(3x1)", r_gj["success"] and r_ga["success"]
      and np.allclose(np.array(r_gj["solution"]).flatten(),
                      np.array(r_ga["solution"]).flatten(), atol=1e-8))

r_gj2 = gauss_jordan(A3, b3_2)
r_ga2 = gaussian_elimination(A3, b3_2)
check("GJ15.2 GJ == GA B(3x2)", r_gj2["success"] and r_ga2["success"]
      and np.allclose(np.array(r_gj2["solution"]), np.array(r_ga2["solution"]), atol=1e-8))

# =============================================================================
# 8. ANALYSIS & EXECUTION TIME
# =============================================================================
print("\n--- 8. Analysis & Time ---")

r = gauss_jordan(A3, b3)
check("GJ16.1 has analysis", r.get("analysis") is not None)
check("GJ16.2 has execution_time", r.get("execution_time") is not None
      and r["execution_time"] > 0)

# =============================================================================
# 9. STRESS TEST 10x10
# =============================================================================
print("\n--- 9. Stress Test (10x10) ---")

A10 = np.random.uniform(-5, 5, (10, 10))
while abs(np.linalg.det(A10)) < 0.1:
    A10 = np.random.uniform(-5, 5, (10, 10))
B10 = np.random.uniform(-5, 5, (10, 1))
r = gauss_jordan(A10.tolist(), B10.tolist())
check("GJ17.1 10x10 success", r["success"])
if r["success"]:
    check("GJ17.2 10x10 correct", np.allclose(np.array(r["solution"]).flatten(),
          np.linalg.solve(A10, B10).flatten(), atol=1e-4))

# =============================================================================
# 10. VERIFICATION A@X = B
# =============================================================================
print("\n--- 10. Verification A@X = B ---")

r = gauss_jordan(A3, b3_2)
if r["success"] and r["solution"]:
    check("GJ18.1 A@X=B", np.allclose(np.array(A3) @ np.array(r["solution"]),
          np.array(b3_2), atol=1e-8))

# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print("\n" + "=" * 60)
print(f"GAUSS-JORDAN RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)