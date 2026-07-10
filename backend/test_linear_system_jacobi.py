"""Comprehensive test suite for Jacobi Iterative Method."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import jacobi, gaussian_elimination

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
print("JACOBI METHOD — COMPREHENSIVE TEST SUITE")
print("=" * 60)

# 1. BASIC
print("\n--- 1. Basic Functionality ---")
r = jacobi(A3, b3, epsilon=1e-10, max_iterations=100)
check("J1.1 success", r["success"])
if r["success"]:
    check("J1.2 correct", np.allclose(np.array(r["solution"]).flatten(), x3, atol=1e-4))
    check("J1.3 has iterations", r.get("iterations_count", 0) > 0)
    check("J1.4 has convergence_analysis", r.get("convergence_analysis") is not None)
    if r.get("convergence_analysis"):
        ca = r["convergence_analysis"]
        check("J1.5 is_diagonally_dominant", ca.get("is_diagonally_dominant") == True)
        check("J1.6 has initial_guess", "initial_guess" in ca)

# 2. REJECTIONS
print("\n--- 2. Rejections ---")
r = jacobi(A3, [[6, 1], [25, 2], [-11, 3]])
check("J2.1 B(3x2) rejected", not r["success"])
r = jacobi([[1,2],[3,4],[5,6]], [[1],[2],[3]])
check("J2.2 non-square rejected", not r["success"])

# 3. EPSILON
print("\n--- 3. Epsilon ---")
r = jacobi(A3, b3, epsilon=1e-3, max_iterations=100)
check("J3.1 big eps success", r["success"])
r = jacobi(A3, b3, epsilon=1e-12, max_iterations=200)
check("J3.2 small eps success", r["success"])

# 4. MAX_ITERATIONS
print("\n--- 4. Max Iterations ---")
r = jacobi(A3, b3, epsilon=1e-15, max_iterations=5)
check("J4.1 not converged", not r["success"])

# 5. INITIAL GUESS
print("\n--- 5. Initial Guess ---")
r = jacobi(A3, b3, initial_guess=[1, 2, -1], epsilon=1e-10, max_iterations=100)
check("J5.1 with guess success", r["success"])
if r["success"]:
    check("J5.2 correct", np.allclose(np.array(r["solution"]).flatten(), x3, atol=1e-4))

# 6. NON-DIAGONALLY DOMINANT
print("\n--- 6. Non-Diagonally Dominant ---")
A_non_dd = [[1, 5], [5, 1]]
b_ndd = [[7], [7]]
r = jacobi(A_non_dd, b_ndd, epsilon=1e-10, max_iterations=100)
check("J6.1 non-DD handled", isinstance(r, dict))
if r.get("convergence_analysis"):
    check("J6.2 non-DD detected", r["convergence_analysis"].get("is_diagonally_dominant") == False)

# 7. JACOBI vs GAUSSIAN
print("\n--- 7. Jacobi vs Gaussian ---")
r_j = jacobi(A3, b3, epsilon=1e-10, max_iterations=100)
r_g = gaussian_elimination(A3, b3)
check("J7.1 Jacobi == Gaussian", r_j["success"] and np.allclose(np.array(r_j["solution"]).flatten(), np.array(r_g["solution"]).flatten(), atol=1e-4))

# 8. 1x1
print("\n--- 8. 1x1 ---")
r = jacobi([[5]], [[10]])
check("J8.1 1x1 success", r["success"])
if r["success"]:
    check("J8.2 1x1 x=2", abs(np.array(r["solution"])[0,0] - 2) < 1e-10)

# 9. ITERATIONS STRUCTURE
print("\n--- 9. Iterations Structure ---")
r = jacobi(A3, b3, epsilon=1e-10, max_iterations=100)
if r.get("iterations"):
    it = r["iterations"][0]
    check("J9.1 has k", "k" in it)
    check("J9.2 has x", "x" in it)
    check("J9.3 has error", "error" in it)

# SUMMARY
total = passed + failed
print("\n" + "=" * 60)
print(f"JACOBI RESULTS: {passed}/{total} passed ({100*passed//total if total else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors: print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)