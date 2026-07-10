"""Test suite for Seidel Iteration Method."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import seidel

passed = failed = 0; errors = []
def check(name, condition, detail=""):
    global passed, failed
    if condition: passed += 1; print(f"  [PASS] {name}")
    else:
        failed += 1; msg = f"  [FAIL] {name} — {detail}"; print(msg); errors.append(msg)

# Test: B = [[0.1, 0.2], [0.2, 0.1]], d = [1, 2] → x = Bx + d
B = [[0.1, 0.2], [0.2, 0.1]]
d_vec = [1, 2]
x_true = np.linalg.solve(np.eye(2) - np.array(B, dtype=float), np.array([1, 2], dtype=float))

print("=" * 60)
print("SEIDEL ITERATION — TEST SUITE")
print("=" * 60)

print("\n--- 1. Basic convergence ---")
r = seidel(B, d_vec, epsilon=1e-10, max_iterations=100)
check("SE1.1 success", r["success"])
if r["success"]:
    check("SE1.2 correct", np.allclose(np.array(r["solution"]).flatten(), x_true, atol=1e-4))
    check("SE1.3 has steps", r.get("steps") is not None)

print("\n--- 2. Seidel faster than Simple Iteration ---")
from app.algorithms.linear_system import simple_iteration
r_si = simple_iteration(B, d_vec, epsilon=1e-10, max_iterations=100)
check("SE2.1 Seidel <= SI iterations", r.get("iterations_count", 999) <= r_si.get("iterations_count", 1000),
      f"s={r.get('iterations_count')} si={r_si.get('iterations_count')}")

print("\n--- 3. Rejections ---")
r2 = seidel(B, [1, 2, 3])
check("SE3.1 d wrong size rejected", not r2["success"])

total = passed + failed
print("\n" + "=" * 60)
print(f"SEIDEL RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed: print(f"\nFAILURES: {errors}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)