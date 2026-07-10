"""Test all 8 linear system methods with a 7×7 ugly-number matrix."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import (
    gaussian_elimination, gauss_jordan, lu_decomposition,
    cholesky_decomposition, jacobi, gauss_seidel,
    simple_iteration, seidel
)

# Generate a 7×7 SPD matrix with ugly numbers (still diagonally dominant for iterative methods)
np.random.seed(42)
M = np.random.uniform(-1, 1, (7, 7))
# Make it strictly diagonally dominant
for i in range(7):
    M[i, i] = sum(abs(M[i, j]) for j in range(7)) + abs(np.random.uniform(1, 10))
# Also make it symmetric positive definite for Cholesky
M = M @ M.T  # ensures SPD but numbers get uglier

A = M.tolist()
b_vec = np.random.uniform(-10, 10, 7).tolist()
B = [[x] for x in b_vec]  # 7×1 column

print("A (7×7) condition number:", round(np.linalg.cond(A), 2))
print("A[0]:", [round(v, 10) for v in A[0]])
print("b:", [round(v, 6) for v in b_vec])
print()

methods = [
    ("Gaussian", lambda: gaussian_elimination(A, B)),
    ("Gauss-Jordan", lambda: gauss_jordan(A, B)),
    ("LU", lambda: lu_decomposition(A, B)),
    ("Cholesky", lambda: cholesky_decomposition(A, B)),
    ("Jacobi", lambda: jacobi(A, B, epsilon=1e-10, max_iterations=200)),
    ("Gauss-Seidel", lambda: gauss_seidel(A, B, epsilon=1e-10, max_iterations=200)),
]

# For Simple Iteration and Seidel, we need B_matrix and d_vec from A
# x = Bx + d where B = I - D⁻¹A, d = D⁻¹b ... but let's use a convergent B directly:
# Use B = 0.5*I (contractive), d = [1,2,3,4,5,6,7]
B_si = [[0.5 if i == j else 0.1 / (abs(i-j) + 1) for j in range(7)] for i in range(7)]
d_si = [float(i + 1) for i in range(7)]
x_si_true = np.linalg.solve(np.eye(7) - np.array(B_si), np.array(d_si))
methods.append(("Simple Iteration", lambda: simple_iteration(B_si, d_si, epsilon=1e-10, max_iterations=200)))
methods.append(("Seidel", lambda: seidel(B_si, d_si, epsilon=1e-10, max_iterations=200)))

passed = failed = 0
for name, fn in methods:
    try:
        r = fn()
        if r.get("success"):
            sol = np.array(r["solution"])
            has_nan = np.any(~np.isfinite(sol))
            has_inf = np.any(np.isinf(sol))
            if has_nan or has_inf:
                print(f"[FAIL] {name}: solution contains NaN/Inf")
                failed += 1
            else:
                # Verify against numpy for A-based methods
                if name not in ("Simple Iteration", "Seidel"):
                    x_true = np.linalg.solve(A, b_vec)
                    ok = np.allclose(sol.flatten(), x_true, atol=1e-2)
                else:
                    ok = np.allclose(sol.flatten(), x_si_true, atol=1e-2)
                status = "PASS" if ok else "WRONG"
                print(f"[{status}] {name}: iterations={r.get('iterations_count','N/A')}, error={r.get('final_error','N/A')}")
                if ok: passed += 1
                else:
                    failed += 1
                    print(f"   expected: {x_true[:3]}...")
                    print(f"   got:      {sol.flatten()[:3]}...")
        else:
            print(f"[FAIL] {name}: not successful — {r.get('message', 'unknown')}")
            failed += 1
    except Exception as e:
        print(f"[CRASH] {name}: {e}")
        failed += 1

print(f"\nRESULTS: {passed}/{passed+failed} passed")
sys.exit(0 if failed == 0 else 1)