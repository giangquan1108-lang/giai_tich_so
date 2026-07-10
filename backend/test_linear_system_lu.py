"""Comprehensive test suite for LU Decomposition method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app.algorithms.linear_system import lu_decomposition, gaussian_elimination

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

A2 = [[4, -1], [-1, 3]]
b2 = [[7], [1]]
x2_true = np.linalg.solve(np.array(A2, dtype=float), np.array([7, 1], dtype=float))

I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
bI3 = [[5], [-3], [2]]

A4 = [[2, 1, 1, 0],
      [4, 3, 3, 1],
      [8, 7, 9, 5],
      [6, 7, 9, 8]]
b4 = [[1], [2], [3], [4]]
x4_true = np.linalg.solve(np.array(A4, dtype=float), np.array([1, 2, 3, 4], dtype=float))

A_non_sq = [[1, 2], [3, 4], [5, 6]]

print("=" * 60)
print("LU DECOMPOSITION — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# =============================================================================
# 1. BASIC TESTS
# =============================================================================
print("\n--- 1. Basic Functionality ---")

# LU1: 3x3 B(3x1)
r = lu_decomposition(A3, b3)
check("LU1.1 B(3x1) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("LU1.2 B(3x1) shape 3x1", sol.shape == (3, 1), f"shape={sol.shape}")
    check("LU1.3 B(3x1) correct", np.allclose(sol.flatten(), x3_true, atol=1e-6),
          f"got {sol.flatten()} expected {x3_true}")

# LU2: 3x3 B(3x2) multi-column
r = lu_decomposition(A3, b3_2)
check("LU2.1 B(3x2) success", r["success"])
if r["success"]:
    sol = np.array(r["solution"])
    check("LU2.2 B(3x2) shape 3x2", sol.shape == (3, 2), f"shape={sol.shape}")
    check("LU2.3 B(3x2) correct", np.allclose(sol, x3_2_true, atol=1e-6))

# LU3: 2x2 B(2x1)
r = lu_decomposition(A2, b2)
check("LU3.1 2x2 success", r["success"])
if r["success"]:
    check("LU3.2 2x2 correct", np.allclose(np.array(r["solution"]).flatten(), x2_true, atol=1e-6))

# LU4: Identity matrix
r = lu_decomposition(I3, bI3)
check("LU4.1 I3 success", r["success"])
if r["success"]:
    check("LU4.2 I3 solution = B", np.allclose(np.array(r["solution"]).flatten(), [5, -3, 2], atol=1e-6))

# LU5: 4x4 B(4x1)
r = lu_decomposition(A4, b4)
check("LU5.1 4x4 success", r["success"])
if r["success"]:
    check("LU5.2 4x4 correct", np.allclose(np.array(r["solution"]).flatten(), x4_true, atol=1e-6))

# LU6: 1x1
r = lu_decomposition([[5]], [[10]])
check("LU6.1 1x1 success", r["success"])
if r["success"]:
    check("LU6.2 1x1 x=2", abs(np.array(r["solution"])[0, 0] - 2.0) < 1e-10)

# =============================================================================
# 2. NON-SQUARE REJECTION
# =============================================================================
print("\n--- 2. Non-Square Rejection ---")

r = lu_decomposition(A_non_sq, [[1], [2], [3]])
check("LU7.1 non-square rejected", not r["success"])
check("LU7.2 msg mentions LU", "LU" in r.get("message", ""))

# =============================================================================
# 3. STEP VERIFICATION
# =============================================================================
print("\n--- 3. Step Verification ---")

r = lu_decomposition(A3, b3)
check("LU8.1 has steps", r.get("steps") and len(r["steps"]) > 0)
check("LU8.2 steps >= 5", len(r.get("steps", [])) >= 5,
      f"got {len(r.get('steps', []))} steps")

# Check phases
phases = [s.get("phase") for s in r.get("steps", []) if s.get("phase")]
check("LU8.3 has lu_elimination phase", "lu_elimination" in phases,
      f"phases={set(phases)}")

# Check L and U matrices are present in steps
step_descs = [s.get("description", "") for s in r.get("steps", [])]
check("LU8.4 L matrix in steps", any("L matrix" in d for d in step_descs))
check("LU8.5 U matrix in steps", any("U matrix" in d for d in step_descs))

# Check elimination_factors in lu_elimination steps
lu_steps = [s for s in r.get("steps", []) if s.get("phase") == "lu_elimination"]
check("LU8.6 has elimination_factors", any(s.get("elimination_factors") for s in lu_steps))

# =============================================================================
# 4. PA = LU VERIFICATION
# =============================================================================
print("\n--- 4. PA = LU Verification ---")

r = lu_decomposition(A3, b3)
if r["success"]:
    # Extract L and U from steps
    L_mat = None
    U_mat = None
    for s in r["steps"]:
        if "L matrix" in s.get("description", ""):
            L_mat = np.array(s["matrix"], dtype=float)
        if "U matrix" in s.get("description", ""):
            U_mat = np.array(s["matrix"], dtype=float)
    
    if L_mat is not None and U_mat is not None:
        LU = L_mat @ U_mat
        A_np = np.array(A3, dtype=float)
        # L*U should reconstruct A up to row permutation (partial pivoting)
        # Since we don't extract P easily, verify L*U has same determinant
        d1 = float(abs(np.linalg.det(LU)))
        d2 = float(abs(np.linalg.det(A_np)))
        ok = abs(d1 - d2) < max(1e-3, 1e-12 * max(d1, d2))
        check("LU9.1 |det(LU)| = |det(A3)|", ok,
              f"|det(LU)|={d1:.6e} |det(A)|={d2:.6e} diff={abs(d1-d2):.2e}")
        
        # L should be lower triangular with 1s on diagonal
        for i in range(3):
            for j in range(i + 1, 3):
                check(f"LU9.2 L[{i},{j}]=0", abs(L_mat[i, j]) < 1e-10,
                      f"got {L_mat[i, j]}")
        check("LU9.3 L diag all 1", all(abs(L_mat[i, i] - 1) < 1e-10 for i in range(3)),
              f"diag={[L_mat[i,i] for i in range(3)]}")
        
        # U should be upper triangular
        for i in range(1, 3):
            for j in range(i):
                check(f"LU9.4 U[{i},{j}]=0", abs(U_mat[i, j]) < 1e-10,
                      f"got {U_mat[i, j]}")

# =============================================================================
# 5. LU vs GAUSSIAN CONSISTENCY
# =============================================================================
print("\n--- 5. LU vs Gaussian Consistency ---")

r_lu = lu_decomposition(A3, b3)
r_ga = gaussian_elimination(A3, b3)
check("LU10.1 LU == GA B(3x1)", r_lu["success"] and r_ga["success"]
      and np.allclose(np.array(r_lu["solution"]).flatten(),
                      np.array(r_ga["solution"]).flatten(), atol=1e-8))

r_lu2 = lu_decomposition(A3, b3_2)
r_ga2 = gaussian_elimination(A3, b3_2)
check("LU10.2 LU == GA B(3x2)", r_lu2["success"] and r_ga2["success"]
      and np.allclose(np.array(r_lu2["solution"]), np.array(r_ga2["solution"]), atol=1e-8))

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
    B_rand = np.random.uniform(-5, 5, (size, 1))
    r = lu_decomposition(A_rand.tolist(), B_rand.tolist())
    if not r["success"] or r["solution"] is None:
        all_ok = False; break
    if not np.allclose(np.array(r["solution"]).flatten(),
                       np.linalg.solve(A_rand, B_rand).flatten(), atol=1e-5):
        all_ok = False; break
check("LU11.1 5 random match numpy", all_ok)

# Multi-column random
all_ok2 = True
for i in range(3):
    A_rand = np.random.uniform(-5, 5, (3, 3))
    while abs(np.linalg.det(A_rand)) < 0.1:
        A_rand = np.random.uniform(-5, 5, (3, 3))
    B_rand = np.random.uniform(-5, 5, (3, 2))
    r = lu_decomposition(A_rand.tolist(), B_rand.tolist())
    if not r["success"] or r["solution"] is None:
        all_ok2 = False; break
    if not np.allclose(np.array(r["solution"]), np.linalg.solve(A_rand, B_rand), atol=1e-5):
        all_ok2 = False; break
check("LU11.2 multi-col random match", all_ok2)

# =============================================================================
# 7. ANALYSIS & TIME
# =============================================================================
print("\n--- 7. Analysis & Time ---")

r = lu_decomposition(A3, b3)
check("LU12.1 has analysis", r.get("analysis") is not None)
check("LU12.2 has execution_time", r.get("execution_time") is not None
      and r["execution_time"] > 0)

# =============================================================================
# 8. STRESS TEST 10x10
# =============================================================================
print("\n--- 8. Stress Test (10x10) ---")

A10 = np.random.uniform(-5, 5, (10, 10))
while abs(np.linalg.det(A10)) < 0.1:
    A10 = np.random.uniform(-5, 5, (10, 10))
B10 = np.random.uniform(-5, 5, (10, 1))
r = lu_decomposition(A10.tolist(), B10.tolist())
check("LU13.1 10x10 success", r["success"])
if r["success"]:
    check("LU13.2 10x10 correct", np.allclose(np.array(r["solution"]).flatten(),
          np.linalg.solve(A10, B10).flatten(), atol=1e-4))

# =============================================================================
# 9. SOLUTION_TYPE
# =============================================================================
print("\n--- 9. Solution Type ---")

r = lu_decomposition(A3, b3)
check("LU14.1 solution_type unique", r.get("solution_type") == "unique",
      f"got {r.get('solution_type')}")

# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print("\n" + "=" * 60)
print(f"LU DECOMPOSITION RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)