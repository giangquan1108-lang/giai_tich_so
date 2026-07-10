"""Comprehensive test suite for Cholesky Decomposition method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app.algorithms.linear_system import cholesky_decomposition, gaussian_elimination

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

A_spd = [[4, 1, 1], [1, 3, -1], [1, -1, 2]]
b_spd = [[6], [2], [3]]
b_spd2 = [[6, 0], [2, 1], [3, -1]]
x_spd = np.linalg.solve(np.array(A_spd, dtype=float), np.array([6, 2, 3], dtype=float))
x_spd2 = np.linalg.solve(np.array(A_spd, dtype=float), np.array(b_spd2, dtype=float))

A_spd2 = [[4, 1], [1, 3]]
b_spd2_small = [[7], [1]]
x_spd2_small = np.linalg.solve(np.array(A_spd2, dtype=float), np.array([7, 1], dtype=float))

A_non_sym = [[1, 2], [3, 4]]
A_non_pd = [[1, 2], [2, 1]]

print("=" * 60)
print("CHOLESKY DECOMPOSITION — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# 1. BASIC
print("\n--- 1. Basic Functionality ---")

r = cholesky_decomposition(A_spd, b_spd)
check("CH1.1 B(3x1) success", r["success"])
if r["success"]:
    check("CH1.2 B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x_spd, atol=1e-6))

r = cholesky_decomposition(A_spd, b_spd2)
check("CH2.1 B(3x2) success", r["success"])
if r["success"]:
    check("CH2.2 B(3x2) shape 3x2", np.array(r["solution"]).shape == (3, 2))
    check("CH2.3 B(3x2) correct", np.allclose(np.array(r["solution"]), x_spd2, atol=1e-6))

r = cholesky_decomposition(A_spd2, b_spd2_small)
check("CH3.1 2x2 success", r["success"])
if r["success"]:
    check("CH3.2 2x2 correct", np.allclose(np.array(r["solution"]).flatten(), x_spd2_small, atol=1e-6))

r = cholesky_decomposition([[9]], [[3]])
check("CH4.1 1x1 success", r["success"])
if r["success"]:
    check("CH4.2 1x1 x=1/3", abs(np.array(r["solution"])[0, 0] - 1/3) < 1e-10)

# 2. REJECTIONS
print("\n--- 2. Rejections ---")

r = cholesky_decomposition(A_non_sym, [[1], [2]])
check("CH5.1 non-symmetric rejected", not r["success"])

r = cholesky_decomposition([[1, 2], [3, 4], [5, 6]], [[1], [2], [3]])
check("CH5.2 non-square rejected", not r["success"])

r = cholesky_decomposition(A_non_pd, [[1], [2]])
check("CH5.3 non-PD rejected", not r["success"])

# 3. STEP VERIFICATION
print("\n--- 3. Step Verification ---")

r = cholesky_decomposition(A_spd, b_spd)
check("CH6.1 has steps", r.get("steps") and len(r["steps"]) > 0)
check("CH6.2 steps >= 6", len(r.get("steps", [])) >= 6)

phases = [s.get("phase") for s in r.get("steps", []) if s.get("phase")]
check("CH6.3 condition_analysis", "condition_analysis" in phases)
check("CH6.4 cholesky_check", "cholesky_check" in phases)
check("CH6.5 cholesky_compute", "cholesky_compute" in phases)
check("CH6.6 cholesky_L", "cholesky_L" in phases)
check("CH6.7 cholesky_verify", "cholesky_verify" in phases)
check("CH6.8 cholesky_forward", "cholesky_forward" in phases)
check("CH6.9 cholesky_back", "cholesky_back" in phases)
check("CH6.10 solution", "solution" in phases)

# 4. L VERIFICATION
print("\n--- 4. L Matrix Verification ---")

r = cholesky_decomposition(A_spd, b_spd)
L_mat = None
for s in r["steps"]:
    if s.get("phase") == "cholesky_L" and s.get("matrix"):
        L_mat = np.array(s["matrix"], dtype=float)

if L_mat is not None:
    check("CH7.1 L*L^T = A", np.allclose(L_mat @ L_mat.T, np.array(A_spd, dtype=float), atol=1e-6))
    for i in range(3):
        for j in range(i + 1, 3):
            check(f"CH7.2 L[{i},{j}]=0", abs(L_mat[i, j]) < 1e-10)
    check("CH7.3 L diag > 0", all(L_mat[i, i] > 0 for i in range(3)))

# 5. CHOLESKY vs GAUSSIAN
print("\n--- 5. Cholesky vs Gaussian ---")

r_ch = cholesky_decomposition(A_spd, b_spd)
r_ga = gaussian_elimination(A_spd, b_spd)
check("CH8.1 CH == GA", r_ch["success"] and r_ga["success"]
      and np.allclose(np.array(r_ch["solution"]).flatten(),
                      np.array(r_ga["solution"]).flatten(), atol=1e-8))

# 6. RANDOM SPD
print("\n--- 6. Random SPD Comparison ---")

np.random.seed(42)
all_ok = True
for i in range(5):
    size = np.random.randint(2, 5)
    A_rand = np.random.uniform(-2, 2, (size, size))
    A_rand = A_rand @ A_rand.T + size * np.eye(size)
    B_rand = np.random.uniform(-5, 5, (size, 1))
    r = cholesky_decomposition(A_rand.tolist(), B_rand.tolist())
    if not r["success"] or r["solution"] is None:
        all_ok = False; break
    if not np.allclose(np.array(r["solution"]).flatten(),
                       np.linalg.solve(A_rand, B_rand).flatten(), atol=1e-5):
        all_ok = False; break
check("CH9.1 5 random SPD match", all_ok)

# 7. ANALYSIS & TIME
print("\n--- 7. Analysis & Time ---")

r = cholesky_decomposition(A_spd, b_spd)
check("CH10.1 has analysis", r.get("analysis") is not None)
check("CH10.2 has execution_time", r.get("execution_time", 0) > 0)

# 8. STRESS 10x10 SPD
print("\n--- 8. Stress 10x10 SPD ---")

A10 = np.random.uniform(-2, 2, (10, 10))
A10 = A10 @ A10.T + 10 * np.eye(10)
B10 = np.random.uniform(-5, 5, (10, 1))
r = cholesky_decomposition(A10.tolist(), B10.tolist())
check("CH11.1 10x10 success", r["success"])
if r["success"]:
    check("CH11.2 10x10 correct", np.allclose(np.array(r["solution"]).flatten(),
          np.linalg.solve(A10, B10).flatten(), atol=1e-4))

# SUMMARY
total = passed + failed
print("\n" + "=" * 60)
print(f"CHOLESKY RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)