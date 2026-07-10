"""Comprehensive test suite for Gauss-Seidel Iterative Method."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import gauss_seidel, jacobi, gaussian_elimination

passed = failed = 0; errors = []
def check(name, condition, detail=""):
    global passed, failed
    if condition: passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1
        try: sd = str(detail).encode('ascii','replace').decode('ascii')
        except: sd = "(detail unavailable)"
        msg = f"  [FAIL] {name} — {sd}"; print(msg); errors.append(msg)

A3 = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
b3 = [[6], [25], [-11]]
x3 = np.linalg.solve(np.array(A3,dtype=float), np.array([6,25,-11],dtype=float))

print("=" * 60)
print("GAUSS-SEIDEL — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# 1. BASIC
print("\n--- 1. Basic ---")
r = gauss_seidel(A3, b3, epsilon=1e-10, max_iterations=100)
check("GS1.1 success", r["success"])
if r["success"]:
    check("GS1.2 correct", np.allclose(np.array(r["solution"]).flatten(), x3, atol=1e-4))
    check("GS1.3 has iterations", r.get("iterations_count", 0) > 0)

# 2. REJECTIONS
print("\n--- 2. Rejections ---")
r = gauss_seidel(A3, [[6,1],[25,2],[-11,3]])
check("GS2.1 B multi-col rejected", not r["success"])
r = gauss_seidel([[1,2],[3,4],[5,6]], [[1],[2],[3]])
check("GS2.2 non-square rejected", not r["success"])

# 3. EPSILON
print("\n--- 3. Epsilon ---")
r = gauss_seidel(A3, b3, epsilon=1e-3, max_iterations=100)
check("GS3.1 big eps", r["success"])
r = gauss_seidel(A3, b3, epsilon=1e-12, max_iterations=200)
check("GS3.2 small eps", r["success"])

# 4. MAX_ITERATIONS
print("\n--- 4. Max Iterations ---")
r = gauss_seidel(A3, b3, epsilon=1e-15, max_iterations=5)
check("GS4.1 limited iters handled", isinstance(r, dict))

# 5. GS vs JACOBI vs GAUSSIAN
print("\n--- 5. GS vs Jacobi vs Gaussian ---")
r_gs = gauss_seidel(A3, b3, epsilon=1e-10, max_iterations=100)
r_ga = gaussian_elimination(A3, b3)
r_ja = jacobi(A3, b3, epsilon=1e-10, max_iterations=100)
check("GS5.1 GS == GA", r_gs["success"] and np.allclose(np.array(r_gs["solution"]).flatten(), np.array(r_ga["solution"]).flatten(), atol=1e-4))
check("GS5.2 GS == Jacobi", r_gs["success"] and np.allclose(np.array(r_gs["solution"]).flatten(), np.array(r_ja["solution"]).flatten(), atol=1e-4))
if r_gs["success"] and r_ja["success"]:
    check("GS5.3 GS <= Jacobi iterations", r_gs["iterations_count"] <= r_ja["iterations_count"],
          f"GS={r_gs['iterations_count']} Jacobi={r_ja['iterations_count']}")

# 6. 1x1
print("\n--- 6. 1x1 ---")
r = gauss_seidel([[5]], [[10]])
check("GS6.1 1x1 success", r["success"])
if r["success"]:
    check("GS6.2 1x1 x=2", abs(np.array(r["solution"])[0,0] - 2) < 1e-10)

# SUMMARY
total = passed + failed
print("\n" + "=" * 60)
print(f"GAUSS-SEIDEL RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors: print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)