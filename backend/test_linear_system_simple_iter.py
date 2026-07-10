"""Test suite for Simple Iteration Method."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import simple_iteration, jacobi

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
print("SIMPLE ITERATION — TEST SUITE")
print("=" * 60)

print("\n--- 1. Basic convergence ---")
r = simple_iteration(B, d_vec, epsilon=1e-10, max_iterations=100)
check("SI1.1 success", r["success"])
if r["success"]:
    check("SI1.2 correct", np.allclose(np.array(r["solution"]).flatten(), x_true, atol=1e-4))
    check("SI1.3 has steps", r.get("steps") is not None)

print("\n--- 2. Rejections ---")
r2 = simple_iteration(B, [1, 2, 3])
check("SI2.1 d wrong size rejected", not r2["success"])

print("\n--- 3. Non-square B ---")
r3 = simple_iteration([[1,2]], [1])
check("SI3.1 non-square rejected", not r3["success"])

total = passed + failed
print("\n" + "=" * 60)
print(f"SIMPLE ITERATION RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed: print(f"\nFAILURES: {errors}")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)