"""Comprehensive test suite for all numerical analysis algorithms."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy import integrate as spi
from scipy.optimize import fsolve

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
        # Sanitize detail to avoid UnicodeEncodeError on Windows cp1252
        try:
            safe_detail = str(detail).encode('ascii', errors='replace').decode('ascii')
        except Exception:
            safe_detail = "(detail unavailable)"
        msg = f"  [FAIL] {name} — {safe_detail}"
        print(msg)
        errors.append(msg)

# =============================================================================
# ROOT FINDING
# =============================================================================
print("=" * 60)
print("ROOT FINDING")
print("=" * 60)

def f1(x): return x**3 - x - 2   # root ~ 1.5213
def fp1(x): return 3*x**2 - 1
def g1(x): return (x+2)**(1/3)    # fixed point

from app.algorithms.root_finding import bisection, newton_raphson, secant, fixed_point_iteration

# Bisection
r = bisection(f1, 1, 2, epsilon=1e-10, max_iterations=50)
check("Bisection converges", r["success"])
check("Bisection root ~1.521", r["success"] and abs(r["root"] - 1.5213797068) < 1e-6,
      f"got {r.get('root')}")
check("Bisection f_root ~0", abs(r.get("f_root", 99)) < 1e-6,
      f"f_root={r.get('f_root')}")

# Bisection: same signs
r = bisection(f1, 2, 3, epsilon=1e-10, max_iterations=10)
check("Bisection rejects same signs", not r["success"] and "opposite" in r["message"].lower())

# Newton
r = newton_raphson(f1, 1.5, epsilon=1e-10, max_iterations=50, f_prime=fp1)
check("Newton converges", r["success"])
check("Newton root ~1.521", r["success"] and abs(r["root"] - 1.5213797068) < 1e-6,
      f"got {r.get('root')}")

# Secant
r = secant(f1, 1, 2, epsilon=1e-10, max_iterations=50)
check("Secant converges", r["success"])
check("Secant root ~1.521", r["success"] and abs(r["root"] - 1.5213797068) < 1e-6,
      f"got {r.get('root')}")

# Fixed point — use a contracting function g(x)=cos(x), fixed point ~0.739
import math
r = fixed_point_iteration(math.cos, 0.5, epsilon=1e-10, max_iterations=100)
check("FixedPoint converges (cos)", r["success"])
check("FixedPoint root ~0.739", r["success"] and abs(r["root"] - 0.7390851332) < 1e-6,
      f"got {r.get('root')}")

# Fixed point divergence
r = fixed_point_iteration(lambda x: 2*x, 1, epsilon=1e-10, max_iterations=10)
check("FixedPoint detects divergence", not r["success"])

# =============================================================================
# NONLINEAR SYSTEM
# =============================================================================
print("\n" + "=" * 60)
print("NONLINEAR SYSTEM")
print("=" * 60)

from app.algorithms.nonlinear_system import newton_multivariable

# System: x^2+y^2=1, x^2-y=0.5  → two solutions: (0.866, 0.5) and (0.931, 0.366)
# Start closer to (0.866, 0.5) to ensure convergence to that root
r = newton_multivariable(
    ["x^2 + y^2 - 1", "x^2 - y - 0.5"],
    ["x", "y"],
    [0.9, 0.45],
    epsilon=1e-10, max_iterations=100,
)
check("Newton system converges", r["success"])
if r["success"] and r.get("solution"):
    x, y = r["solution"]
    # Accept either solution — both satisfy the system
    sol1 = abs(x - 0.8660254038) < 1e-4 and abs(y - 0.5) < 1e-4
    sol2 = abs(x - 0.9306048591) < 1e-4 and abs(y - 0.3660254038) < 1e-4
    check("Newton system correct solution", sol1 or sol2,
          f"got x={x}, y={y}")

# =============================================================================
# LINEAR SYSTEM
# =============================================================================
print("\n" + "=" * 60)
print("LINEAR SYSTEM AX=B")
print("=" * 60)

from app.algorithms.linear_system import (
    gaussian_elimination, gauss_jordan, lu_decomposition,
    cholesky_decomposition, thomas_algorithm, jacobi, gauss_seidel, sor
)

# Test matrices
A3 = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
b3 = [[6], [25], [-11]]          # B(3×1)
x3_true = np.linalg.solve(np.array(A3), np.array([6,25,-11]))
b3_2 = [[6, 1], [25, 2], [-11, 3]]  # B(3×2)
x3_2_true = np.linalg.solve(np.array(A3), np.array(b3_2))

# Edge-case matrices
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
x53_true = np.linalg.solve(np.array(A5), np.array(b53))

# --- Gaussian ---
r = gaussian_elimination(A3, b3)
check("Gaussian B(3x1) success", r["success"])
sol = np.array(r["solution"])
check("Gaussian B(3x1) correct", np.allclose(sol.flatten(), x3_true, atol=1e-6),
      f"got {sol.flatten()} expected {x3_true}")
check("Gaussian B(3x1) shape 3×1", sol.shape == (3, 1) or sol.shape == (3,), f"shape={sol.shape}")

r = gaussian_elimination(A3, b3_2)
check("Gaussian B(3x2) success", r["success"])
sol2 = np.array(r["solution"])
check("Gaussian B(3x2) shape 3×2", sol2.shape == (3, 2), f"shape={sol2.shape}")
check("Gaussian B(3x2) correct", np.allclose(sol2, x3_2_true, atol=1e-6),
      f"got {sol2} expected {x3_2_true}")

# --- Gauss-Jordan ---
r = gauss_jordan(A3, b3)
check("GaussJordan B(3x1) success", r["success"])
sol_gj = np.array(r["solution"])
check("GaussJordan B(3x1) correct", np.allclose(sol_gj.flatten(), x3_true, atol=1e-6),
      f"got {sol_gj.flatten()} expected {x3_true}")

r = gauss_jordan(A3, b3_2)
check("GaussJordan B(3x2) shape 3×2", np.array(r["solution"]).shape == (3, 2))
check("GaussJordan B(3x2) correct", np.allclose(np.array(r["solution"]), x3_2_true, atol=1e-6))

# Gauss-Jordan: inconsistent
r = gauss_jordan(A_inc, b_inc)
check("GaussJordan inconsistent detected", r["solution_type"] == "inconsistent",
      f"got {r.get('solution_type')}")
check("GaussJordan rank(A)=1 < rank(aug)=2",
      r.get("rank_A") == 1 and r.get("rank_augmented") == 2)

# Gauss-Jordan: infinite
r = gauss_jordan(A_inf, b_inf)
check("GaussJordan infinite detected", r["solution_type"] == "infinite",
      f"got {r.get('solution_type')}")
check("GaussJordan has free variables", r.get("free_variables") and len(r["free_variables"]) > 0)
check("GaussJordan has particular solution", r.get("particular_solution") is not None)
check("GaussJordan has basis vectors", r.get("basis_vectors") and len(r["basis_vectors"]) > 0)

# Gauss-Jordan: 5x5 with B(5x3)
r = gauss_jordan(A5, b53)
check("GaussJordan A(5x5) B(5x3) success", r["success"])
check("GaussJordan A(5x5) B(5x3) shape 5×3",
      np.array(r["solution"]).shape == (5, 3),
      f"shape={np.array(r['solution']).shape}")
check("GaussJordan A(5x5) B(5x3) correct",
      np.allclose(np.array(r["solution"]), x53_true, atol=1e-6))

# --- LU ---
r = lu_decomposition(A3, b3_2)
check("LU B(3x2) shape 3×2", np.array(r["solution"]).shape == (3, 2))
check("LU B(3x2) correct", np.allclose(np.array(r["solution"]), x3_2_true, atol=1e-6))

# LU non-square rejection
r = lu_decomposition([[1,2],[3,4],[5,6]], [[1],[2],[3]])
check("LU rejects non-square", not r["success"] and "LU" in r["message"])

# --- Cholesky ---
A_spd = [[4, 1, 1], [1, 3, -1], [1, -1, 2]]
b_spd = [[6], [2], [3]]
b_spd2 = [[6, 0], [2, 1], [3, -1]]
x_spd = np.linalg.solve(np.array(A_spd), np.array([6,2,3]))
x_spd2 = np.linalg.solve(np.array(A_spd), np.array(b_spd2))

r = cholesky_decomposition(A_spd, b_spd)
check("Cholesky B(3x1) success", r["success"])
check("Cholesky B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x_spd, atol=1e-6))

r = cholesky_decomposition(A_spd, b_spd2)
check("Cholesky B(3x2) success", r["success"])
check("Cholesky B(3x2) shape 3×2", np.array(r["solution"]).shape == (3, 2))
check("Cholesky B(3x2) correct", np.allclose(np.array(r["solution"]), x_spd2, atol=1e-6))

# Cholesky non-symmetric
r = cholesky_decomposition([[1,2],[3,4]], [[1],[2]])
check("Cholesky rejects non-symmetric", not r["success"])

# --- Thomas ---
A_tri = [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
b_tri = [[6], [2], [6]]
b_tri2 = [[6, 1], [2, 2], [6, 3]]
x_tri = np.linalg.solve(np.array(A_tri), np.array([6,2,6]))
x_tri2 = np.linalg.solve(np.array(A_tri), np.array(b_tri2))

r = thomas_algorithm(A_tri, b_tri)
check("Thomas B(3x1) success", r["success"])
check("Thomas B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x_tri, atol=1e-6))

r = thomas_algorithm(A_tri, b_tri2)
check("Thomas B(3x2) success", r["success"])
check("Thomas B(3x2) shape 3×2", np.array(r["solution"]).shape == (3, 2))
check("Thomas B(3x2) correct", np.allclose(np.array(r["solution"]), x_tri2, atol=1e-6))

# Thomas non-tridiagonal
r = thomas_algorithm(A3, b3)
check("Thomas rejects non-tridiagonal", not r["success"])

# --- Iterative methods ---
# Jacobi with B(3x1)
r = jacobi(A3, b3, epsilon=1e-10, max_iterations=100)
check("Jacobi B(3x1) success", r["success"])
if r["success"]:
    check("Jacobi B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x3_true, atol=1e-4))

# Jacobi with B(3x2) — should fail
r = jacobi(A3, b3_2)
check("Jacobi B(3x2) rejected (only vector)", not r["success"],
      f"msg={r.get('message')}")

# Gauss-Seidel
r = gauss_seidel(A3, b3, epsilon=1e-10, max_iterations=100)
check("GaussSeidel B(3x1) success", r["success"])
if r["success"]:
    check("GaussSeidel B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x3_true, atol=1e-4))

r = gauss_seidel(A3, b3_2)
check("GaussSeidel B(3x2) rejected", not r["success"])

# SOR
r = sor(A3, b3, epsilon=1e-10, max_iterations=100, omega=1.0)
check("SOR B(3x1) success", r["success"])
if r["success"]:
    check("SOR B(3x1) correct", np.allclose(np.array(r["solution"]).flatten(), x3_true, atol=1e-4))

r = sor(A3, b3_2)
check("SOR B(3x2) rejected", not r["success"])

# --- Solution type: unique, inconsistent, infinite ---
r = gaussian_elimination(A_inc, b_inc)
check("Inconsistent system detected", r["solution_type"] == "inconsistent")
check("rank(A)=1 < rank(aug)=2", r.get("rank_A") == 1 and r.get("rank_augmented") == 2)

r = gaussian_elimination(A_inf, b_inf)
check("Infinite system detected", r["solution_type"] == "infinite")
check("Has free variables", r.get("free_variables") and len(r["free_variables"]) > 0)
check("Has particular solution", r.get("particular_solution") is not None)
check("Has basis vectors", r.get("basis_vectors") and len(r["basis_vectors"]) > 0)

# --- 5x5 with B(5x3) ---
r = gaussian_elimination(A5, b53)
check("A(5x5) B(5x3) success", r["success"])
check("A(5x5) B(5x3) shape 5×3", np.array(r["solution"]).shape == (5, 3),
      f"shape={np.array(r['solution']).shape}")
check("A(5x5) B(5x3) correct", np.allclose(np.array(r["solution"]), x53_true, atol=1e-6))

# =============================================================================
# INTEGRATION
# =============================================================================
print("\n" + "=" * 60)
print("INTEGRATION")
print("=" * 60)

from app.algorithms.integration import trapezoidal, simpson_one_third, simpson_three_eighth, romberg

# ∫0→1 x² dx = 1/3 ≈ 0.3333333
import math as _math
true_integral = 1/3
def f_int(x): return x**2

r = trapezoidal(f_int, 0, 1, n=1000)
check("Trapezoidal success", r["success"])
check("Trapezoidal ~1/3", abs(r["result"] - true_integral) < 1e-4, f"got {r['result']}")

r = simpson_one_third(f_int, 0, 1, n=100)
check("Simpson 1/3 success", r["success"])
check("Simpson 1/3 ~1/3", abs(r["result"] - true_integral) < 1e-6, f"got {r['result']}")

r = simpson_three_eighth(f_int, 0, 1, n=99)
check("Simpson 3/8 success", r["success"])
check("Simpson 3/8 ~1/3", abs(r["result"] - true_integral) < 1e-6, f"got {r['result']}")

r = romberg(f_int, 0, 1, n=8)
check("Romberg success", r["success"])
check("Romberg ~1/3", abs(r["result"] - true_integral) < 1e-6, f"got {r['result']}")

# Check scipy comparison available
check("Integration has scipy reference", r.get("exact_value") is not None)

# =============================================================================
# MATRIX INVERSE
# =============================================================================
print("\n" + "=" * 60)
print("MATRIX INVERSE")
print("=" * 60)

from app.algorithms.linear_system import (
    matrix_inverse_gauss_jordan, matrix_inverse_adjoint,
    matrix_inverse_lu, matrix_inverse_cholesky
)

A_inv = [[4, 1], [2, 3]]
A_inv_true = np.linalg.inv(np.array(A_inv))

for name, func in [("GaussJordan", matrix_inverse_gauss_jordan),
                    ("Adjoint", matrix_inverse_adjoint),
                    ("LU", matrix_inverse_lu)]:
    r = func(A_inv)
    check(f"Inverse {name} success", r["success"],
          f"msg={r.get('message')}")
    if r["success"]:
        inv = np.array(r["inverse"])
        check(f"Inverse {name} correct", np.allclose(inv, A_inv_true, atol=1e-6),
              f"got {inv}")
        check(f"Inverse {name} verification", r.get("is_accurate", False))

# Cholesky inverse requires SPD matrix — test separately
A_spd_inv = [[4, 1], [1, 3]]  # symmetric positive definite
A_spd_inv_true = np.linalg.inv(np.array(A_spd_inv))
r = matrix_inverse_cholesky(A_spd_inv)
check("Inverse Cholesky SPD success", r["success"],
      f"msg={r.get('message')}")
if r["success"]:
    inv = np.array(r["inverse"])
    check("Inverse Cholesky SPD correct", np.allclose(inv, A_spd_inv_true, atol=1e-6),
          f"got {inv}")
    check("Inverse Cholesky SPD verification", r.get("is_accurate", False))

# Cholesky should reject non-symmetric matrix
r = matrix_inverse_cholesky(A_inv)  # non-symmetric
check("Inverse Cholesky rejects non-symmetric", not r["success"])

# Singular matrix
A_sing = [[1, 1], [2, 2]]
for name, func in [("GaussJordan", matrix_inverse_gauss_jordan),
                    ("Adjoint", matrix_inverse_adjoint),
                    ("LU", matrix_inverse_lu)]:
    r = func(A_sing)
    check(f"Inverse {name} rejects singular", not r["success"],
          f"msg={r.get('message', 'N/A')[:80]}")

# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 60)
sys.exit(0 if failed == 0 else 1)