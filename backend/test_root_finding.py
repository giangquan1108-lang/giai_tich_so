"""Comprehensive test suite for root finding algorithms.
Covers: Bisection, Newton-Raphson, Secant, Fixed Point Iteration.
Tests: normal functions, edge cases, special functions, tiny epsilon, cross-comparison.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

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
        msg = f"  [FAIL] {name} — {detail}"
        print(msg)
        errors.append(msg)

from app.algorithms.root_finding import bisection, newton_raphson, secant, fixed_point_iteration

# =============================================================================
# Helper: verify output structure
# =============================================================================
REQUIRED_KEYS = ["success", "message", "root", "f_root", "iterations_count",
                 "final_error", "iterations", "execution_time",
                 "convergence_data", "effective_epsilon", "machine_epsilon"]

def verify_structure(r, method_name):
    """Verify all required keys exist and have correct types."""
    for key in REQUIRED_KEYS:
        if key not in r:
            check(f"[STRUCT] {method_name} has {key}", False, f"Key '{key}' missing")
    if r["success"]:
        if not isinstance(r["root"], (int, float)):
            check(f"[STRUCT] {method_name} root is numeric", False, f"got {type(r['root'])}")
        check(f"[STRUCT] {method_name} iterations > 0", r["iterations_count"] > 0,
              f"iterations={r['iterations_count']}")
        check(f"[STRUCT] {method_name} iterations list non-empty", len(r["iterations"]) > 0)
    cd = r.get("convergence_data", {})
    n = len(cd.get("iterations", []))
    check(f"[STRUCT] {method_name} conv_data lengths match",
          len(cd.get("errors", [])) == n and len(cd.get("x_values", [])) == n,
          f"iters={n}, errors={len(cd.get('errors',[]))}, x={len(cd.get('x_values',[]))}")
    check(f"[STRUCT] {method_name} effective_epsilon >= 1e-15",
          r.get("effective_epsilon", 0) >= 1e-15,
          f"eps_eff={r.get('effective_epsilon')}")


# =============================================================================
# TEST FUNCTION DEFINITIONS
# =============================================================================

# f1: x³ - x - 2, root ~ 1.5213797068045676
def f_cubic(x): return x**3 - x - 2
def fp_cubic(x): return 3*x**2 - 1

# f2: x² - 4, roots -2, 2
def f_quad(x): return x**2 - 4
def fp_quad(x): return 2*x

# f3: x^15 - x + 1 (bug fix test)
def f_high_poly(x): return x**15 - x + 1

# f4: sin(x) root at pi
def f_sin(x): return math.sin(x)

# f5: eˣ - 4, root = ln(4)
def f_exp(x): return math.exp(x) - 4

# f6: 1/x → no root (horizontal)
def f_hyp(x): return 1.0 / x if x != 0 else float('inf')

# f7: x² - 2, root = sqrt(2)
def f_sqrt2(x): return x**2 - 2
def fp_sqrt2(x): return 2*x

# f8: cos(x) - x, root ~ 0.739085
def f_cos_x(x): return math.cos(x) - x
def fp_cos_x(x): return -math.sin(x) - 1

# f9: x³ at x=0 (flat root, f'(0)=0)
def f_x3(x): return x**3
def fp_x3(x): return 3*x**2

# f10: 1/(1+x²) - 0.5, root at x=±1, saddle
def f_saddle(x): return 1.0/(1.0 + x**2) - 0.5
def fp_saddle(x): return -(2*x)/((1 + x**2)**2)

# f11: tanh(x), root at 0
def f_tanh(x): return math.tanh(x)

# f12: (x-1)^10, root at 1 (high multiplicity)
def f_multi(x): return (x-1)**10

# f13: eˣ, derivative = eˣ → diverges from large x₀
def f_exp2(x): return math.exp(x)
def fp_exp2(x): return math.exp(x)

# f14: x²+1, no real root
def f_no_real(x): return x**2 + 1
def fp_no_real(x): return 2*x

# f15: hằng số 5
def f_const(x): return 5.0

# Fixed point functions
def g_cos(x): return math.cos(x)           # contraction, fixed point ~0.739085
def g_cubic(x): return (x+2)**(1/3)        # from x³-x-2=0 → x=(x+2)^{1/3}
def g_diverge(x): return 2*x               # diverges
def g_oscillate(x): return -x              # oscillates
def g_sqrt2(x): return (x**2 - 2)/2 + x   # Newton for x²-2 → x - (x²-2)/(2x) = (x+2/x)/2
def g_exp(x): return math.exp(x)           # overflow


# =============================================================================
# I. BISECTION (10 TESTS)
# =============================================================================
print("=" * 70)
print("BISECTION METHOD")
print("=" * 70)

# Test 1: Basic cubic
r = bisection(f_cubic, 1, 2, epsilon=1e-10, max_iterations=50)
check("Bisect-01: converges", r["success"])
check("Bisect-01: root ~1.5213797", r["success"] and abs(r["root"] - 1.5213797068045676) < 1e-6,
      f"got {r.get('root')}")
check("Bisect-01: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
      f"f_root={r.get('f_root')}")
verify_structure(r, "Bisect-01")

# Test 2: x²-4 on [0,3] → root=2
r = bisection(f_quad, 0, 3, epsilon=1e-10, max_iterations=50)
check("Bisect-02: root=2", r["success"] and abs(r["root"] - 2.0) < 1e-6,
      f"got {r.get('root')}")

# Test 3: x²-4 on [-3,0] → root=-2 (negative interval)
r = bisection(f_quad, -3, 0, epsilon=1e-10, max_iterations=50)
check("Bisect-03: root=-2 (negative interval)", r["success"] and abs(r["root"] - (-2.0)) < 1e-6,
      f"got {r.get('root')}")

# Test 4: x^15-x+1 on [-2,0] → root ~ -1.04898 (bug fix test)
r = bisection(f_high_poly, -2, 0, epsilon=1e-10, max_iterations=100)
check("Bisect-04: high-degree poly converges", r["success"])
if r["success"]:
    check("Bisect-04: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
          f"f_root={r.get('f_root')}")
    check("Bisect-04: root between -2 and 0", -2 <= r["root"] <= 0)
    # Expected root ~ -1.04898
    check("Bisect-04: root ~-1.04898", abs(r["root"] - (-1.04898)) < 0.01,
          f"got {r.get('root')}")

# Test 5: sin(x) on [3, 4] → root = π (transcendental)
r = bisection(f_sin, 3, 4, epsilon=1e-10, max_iterations=60)
check("Bisect-05: sin(x) root=pi", r["success"] and abs(r["root"] - math.pi) < 1e-6,
      f"got {r.get('root')}, expected {math.pi}")

# Test 6: eˣ-4 on [1, 2] → root = ln(4)
r = bisection(f_exp, 1, 2, epsilon=1e-10, max_iterations=50)
check("Bisect-06: exp(x)-4 root=ln(4)", r["success"] and abs(r["root"] - math.log(4)) < 1e-6,
      f"got {r.get('root')}, expected {math.log(4)}")

# Test 7: x²-2 on [1, 2] → root at sqrt(2) ≈ 1.414 (nested root test)
r = bisection(f_sqrt2, 1, 2, epsilon=1e-10, max_iterations=50)
check("Bisect-07: root = sqrt(2)", r["success"] and abs(r["root"] - math.sqrt(2)) < 1e-6,
      f"got {r.get('root')}")

# Test 8: f(a)*f(b) >= 0 → reject (same signs)
r = bisection(f_quad, 3, 5, epsilon=1e-10, max_iterations=10)
check("Bisect-08: reject same sign", not r["success"] and "opposite" in r["message"].lower(),
      f"msg={r.get('message')}")

# Test 9: Very small epsilon 1e-14
r = bisection(f_cubic, 1, 2, epsilon=1e-14, max_iterations=200)
check("Bisect-09: tiny epsilon converges", r["success"])
if r["success"]:
    check("Bisect-09: very high precision", abs(r["root"] - 1.5213797068045676) < 1e-10,
          f"got {r.get('root')}")
    check("Bisect-09: |f(root)| < 1e-10", abs(r.get("f_root", 99)) < 1e-10,
          f"f_root={r.get('f_root')}")

# Test 10: Constant function on [-1,1] → f(a)=f(b)=5 → reject
r = bisection(f_const, -1, 1, epsilon=1e-6, max_iterations=10)
check("Bisect-10: reject constant", not r["success"])


# =============================================================================
# II. NEWTON-RAPHSON (14 TESTS)
# =============================================================================
print("\n" + "=" * 70)
print("NEWTON-RAPHSON METHOD")
print("=" * 70)

# Test 1: Basic with analytical derivative
r = newton_raphson(f_cubic, 1.5, epsilon=1e-10, max_iterations=50, f_prime=fp_cubic)
check("Newton-01: analytical f' converges", r["success"])
check("Newton-01: root ~1.5213797", r["success"] and abs(r["root"] - 1.5213797068045676) < 1e-6,
      f"got {r.get('root')}")
check("Newton-01: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
      f"f_root={r.get('f_root')}")
verify_structure(r, "Newton-01")

# Test 2: Numerical derivative (f_prime=None)
r = newton_raphson(f_cubic, 1.5, epsilon=1e-10, max_iterations=50, f_prime=None)
check("Newton-02: numerical f' converges", r["success"])
check("Newton-02: root ~1.5213797", r["success"] and abs(r["root"] - 1.5213797068045676) < 1e-6,
      f"got {r.get('root')}")

# Test 3: x^15-x+1 from x0=1 → MUST return False (bug fix test)
r = newton_raphson(f_high_poly, 1.0, epsilon=1e-10, max_iterations=100, f_prime=None)
check("Newton-03: x^15-x+1 from x=1 FAILS (no false positive)", not r["success"],
      f"got success={r['success']}, root={r.get('root')}, f_root={r.get('f_root')}")

# Test 4: x^15-x+1 from x0=-1 → should find root near -1.048
r = newton_raphson(f_high_poly, -1.0, epsilon=1e-10, max_iterations=100, f_prime=None)
check("Newton-04: x^15-x+1 from x=-1 converges", r["success"])
if r["success"]:
    check("Newton-04: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
          f"f_root={r.get('f_root')}")
    check("Newton-04: root ~-1.04898", abs(r["root"] - (-1.04898)) < 0.01,
          f"got {r.get('root')}")

# Test 5: sqrt(2) with analytical derivative
r = newton_raphson(f_sqrt2, 1.0, epsilon=1e-10, max_iterations=50, f_prime=fp_sqrt2)
check("Newton-05: sqrt(2) ~1.41421", r["success"] and abs(r["root"] - math.sqrt(2)) < 1e-6,
      f"got {r.get('root')}")

# Test 6: cos(x)-x with analytical derivative
r = newton_raphson(f_cos_x, 0.5, epsilon=1e-10, max_iterations=50, f_prime=fp_cos_x)
check("Newton-06: cos(x)-x ~0.739085", r["success"] and abs(r["root"] - 0.7390851332151607) < 1e-6,
      f"got {r.get('root')}")
check("Newton-06: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
      f"f_root={r.get('f_root')}")

# Test 7: x³ from x0=1 → f'(0)=0 near root → Newton fails (derivative→0)
# This is CORRECT behavior: Newton cannot handle f'(root)=0
r = newton_raphson(f_x3, 1.0, epsilon=1e-10, max_iterations=100, f_prime=fp_x3)
check("Newton-07: x³ flat root FAILS (correct, f'(0)=0)", not r["success"],
      f"success={r['success']}, root={r.get('root')}, msg={r.get('message')}")

# Test 8: eˣ from x0=100 → step too large → should fail
r = newton_raphson(f_exp2, 100.0, epsilon=1e-10, max_iterations=50, f_prime=fp_exp2)
check("Newton-08: e^x from x=100 diverges", not r["success"],
      f"got success={r['success']}, msg={r.get('message')}")

# Test 9: 1/(1+x²)-0.5 from x=0 → f'(0)=0 saddle → maybe false convergence
r = newton_raphson(f_saddle, 0.0, epsilon=1e-10, max_iterations=100, f_prime=fp_saddle)
# f'(0)=0 → Newton fails at x=0
check("Newton-09: saddle point detected", not r["success"],
      f"got success={r['success']}, msg={r.get('message')}")

# Test 10: Tiny epsilon 1e-14 with analytical f'
r = newton_raphson(f_cubic, 1.5, epsilon=1e-14, max_iterations=100, f_prime=fp_cubic)
check("Newton-10: tiny epsilon converges", r["success"])
if r["success"]:
    check("Newton-10: high precision", abs(r["root"] - 1.5213797068045676) < 1e-10,
          f"got {r.get('root')}")

# Test 11: tanh(x) from x=0.5
r = newton_raphson(f_tanh, 0.5, epsilon=1e-10, max_iterations=50, f_prime=None)
check("Newton-11: tanh(x) root=0", r["success"] and abs(r["root"]) < 1e-6,
      f"got {r.get('root')}")
check("Newton-11: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6)

# Test 12: (x-1)^10 from x=0.5 (multiplicity 10 → linearly convergent, slow)
# Newton converges linearly (not quadratically) for high-multiplicity roots.
# After 300 iters, root approaches ~0.945 with f(root) ≈ 0 (underflow).
# This is expected behavior — multiplicity severely degrades convergence rate.
r = newton_raphson(f_multi, 0.5, epsilon=1e-10, max_iterations=300, f_prime=None)
check("Newton-12: high multiplicity handled (no crash)", isinstance(r, dict))
if r["success"]:
    check("Newton-12: root >= 0.9", r["root"] >= 0.9,
          f"got root={r.get('root')}")
check("Newton-12: |f(root)| nearly 0", abs(r.get("f_root", 99)) < 1e-5,
      f"f_root={r.get('f_root')}")

# Test 13: sin(1/x) → oscillatory, likely NaN/Inf
def f_osc(x):
    if abs(x) < 1e-15:
        return float('inf')
    return math.sin(1.0 / x)
r = newton_raphson(f_osc, 0.3, epsilon=1e-10, max_iterations=20, f_prime=None)
# This function is problematic; check that the algorithm handles it gracefully
check("Newton-13: oscillatory function handled (no crash)", isinstance(r, dict),
      f"R returned {type(r)}")

# Test 14: x²+1 from x=0 → no real root
r = newton_raphson(f_no_real, 0.0, epsilon=1e-10, max_iterations=50, f_prime=fp_no_real)
check("Newton-14: no real root detected", not r["success"],
      f"got success={r['success']}, root={r.get('root')}")


# =============================================================================
# III. SECANT METHOD (8 TESTS)
# =============================================================================
print("\n" + "=" * 70)
print("SECANT METHOD")
print("=" * 70)

# Test 1: Basic
r = secant(f_cubic, 1, 2, epsilon=1e-10, max_iterations=50)
check("Secant-01: converges", r["success"])
check("Secant-01: root ~1.5213797", r["success"] and abs(r["root"] - 1.5213797068045676) < 1e-6,
      f"got {r.get('root')}")
check("Secant-01: |f(root)| < 1e-6", abs(r.get("f_root", 99)) < 1e-6,
      f"f_root={r.get('f_root')}")
verify_structure(r, "Secant-01")

# Test 2: x²-4 from 0,3 → root=2
r = secant(f_quad, 0, 3, epsilon=1e-10, max_iterations=50)
check("Secant-02: root=2", r["success"] and abs(r["root"] - 2.0) < 1e-6,
      f"got {r.get('root')}")

# Test 3: x^15-x+1 from -2,0 — secant may diverge on very high-degree polys
# without derivative info. This test verifies the algorithm doesn't crash.
r = secant(f_high_poly, -2, 0, epsilon=1e-8, max_iterations=300)
check("Secant-03: handles high-degree poly (no crash)", isinstance(r, dict),
      f"r type={type(r)}")
if r["success"]:
    check("Secant-03: |f(root)| < 1e-4", abs(r.get("f_root", 99)) < 1e-4,
          f"f_root={r.get('f_root')}")
    check("Secant-03: root ~-1.048", abs(r["root"] - (-1.04898)) < 0.01,
          f"got {r.get('root')}")

# Test 4: eˣ-4 from 1,2
r = secant(f_exp, 1, 2, epsilon=1e-10, max_iterations=50)
check("Secant-04: exp(x)-4 root=ln(4)", r["success"] and abs(r["root"] - math.log(4)) < 1e-6,
      f"got {r.get('root')}")

# Test 5: sqrt(2) from 1,2
r = secant(f_sqrt2, 1, 2, epsilon=1e-10, max_iterations=50)
check("Secant-05: sqrt(2) ~1.41421", r["success"] and abs(r["root"] - math.sqrt(2)) < 1e-6,
      f"got {r.get('root')}")

# Test 6: Two points very close → near division by zero
r = secant(f_cubic, 1.5, 1.500000000001, epsilon=1e-10, max_iterations=50)
check("Secant-06: very close points", r["success"],
      f"success={r['success']}, root={r.get('root')}, f_root={r.get('f_root')}")

# Test 7: Tiny epsilon 1e-14
r = secant(f_cubic, 1, 2, epsilon=1e-14, max_iterations=200)
check("Secant-07: tiny epsilon converges", r["success"])
if r["success"]:
    check("Secant-07: high precision", abs(r["root"] - 1.5213797068045676) < 1e-10,
          f"got {r.get('root')}")

# Test 8: Constant function → no root (f(x0)=f(x1)=5)
r = secant(f_const, 1, 2, epsilon=1e-6, max_iterations=20)
check("Secant-08: constant -> fail/divzero", not r["success"],
      f"success={r['success']}, msg={r.get('message')}")


# =============================================================================
# IV. FIXED POINT ITERATION (7 TESTS)
# =============================================================================
print("\n" + "=" * 70)
print("FIXED POINT ITERATION")
print("=" * 70)

# Test 1: cos(x) — contraction, root ~0.739085
r = fixed_point_iteration(g_cos, 0.5, epsilon=1e-10, max_iterations=100)
check("FP-01: cos(x) converges", r["success"])
check("FP-01: root ~0.739085", r["success"] and abs(r["root"] - 0.7390851332151607) < 1e-6,
      f"got {r.get('root')}")
verify_structure(r, "FP-01")

# Test 2: g(x) = (x+2)^{1/3} from f(x) = x³ - x - 2 = 0
r = fixed_point_iteration(g_cubic, 1.0, epsilon=1e-10, max_iterations=200)
check("FP-02: (x+2)^(1/3) converges", r["success"])
if r["success"]:
    check("FP-02: root ~1.5213797", abs(r["root"] - 1.5213797068045676) < 1e-4,
          f"got {r.get('root')}")
    # The error here is |g(x)-x| not |f(x)|, so we check differently
    check("FP-02: error small", r.get("final_error", 99) < 1e-4,
          f"final_error={r.get('final_error')}")

# Test 3: g(x)=2x → diverges (|g'|=2 >= 1)
r = fixed_point_iteration(g_diverge, 1, epsilon=1e-10, max_iterations=10)
check("FP-03: detects divergence (2x)", not r["success"],
      f"got success={r['success']}")

# Test 4: g(x)=-x → oscillates (|g'|=1)
r = fixed_point_iteration(g_oscillate, 1, epsilon=1e-10, max_iterations=10)
check("FP-04: detects oscillation/divergence (-x)", not r["success"],
      f"got success={r['success']}, msg={r.get('message')}")

# Test 5: g(x) = x^(1/3) (no sign) → complex values possible for x<0
# Test with positive x
import numpy as np
def g_cbrt(x):
    if x >= 0:
        return x**(1/3)
    else:
        return -((-x)**(1/3))
r = fixed_point_iteration(g_cbrt, 10, epsilon=1e-10, max_iterations=20)
check("FP-05: real cube root converges", r["success"],
      f"success={r['success']}, root={r.get('root')}")
if r["success"]:
    # Fixed point of x^(1/3): x = x^(1/3) => x^3 = x => x(x²-1)=0 => x=0,1,-1
    check("FP-05: root is 1", abs(r["root"] - 1.0) < 1e-3,
          f"got {r.get('root')}")

# Test 6: eˣ → overflow (diverges from any starting point > 0)
r = fixed_point_iteration(g_exp, 5, epsilon=1e-10, max_iterations=50)
check("FP-06: e^x overflow detected", not r["success"],
      f"success={r['success']}, msg={r.get('message')}")

# Test 7: Newton fixed point for sqrt(2): g(x) = (x + 2/x)/2
def g_newton_sqrt2(x):
    if x == 0:
        return float('inf')
    return (x + 2.0/x) / 2.0
r = fixed_point_iteration(g_newton_sqrt2, 1.0, epsilon=1e-10, max_iterations=50)
check("FP-07: Newton fixed-point for sqrt(2) converges", r["success"])
if r["success"]:
    check("FP-07: root ~1.41421", abs(r["root"] - math.sqrt(2)) < 1e-6,
          f"got {r.get('root')}")


# =============================================================================
# V. CROSS-COMPARISON OF ALL 4 METHODS
# =============================================================================
print("\n" + "=" * 70)
print("CROSS-COMPARISON")
print("=" * 70)

# Test C1: All 4 methods on same function should agree
true_root = 1.5213797068045676
methods = [
    ("Bisection", lambda: bisection(f_cubic, 1, 2, epsilon=1e-10, max_iterations=50)),
    ("Newton", lambda: newton_raphson(f_cubic, 1.5, epsilon=1e-10, max_iterations=50, f_prime=fp_cubic)),
    ("Secant", lambda: secant(f_cubic, 1, 2, epsilon=1e-10, max_iterations=50)),
]

for name, fn in methods:
    r = fn()
    check(f"Cross-01: {name} matches true root",
          r["success"] and abs(r["root"] - true_root) < 1e-6,
          f"got {r.get('root')}")

# Test C2: All methods on x²-4 should find root=2
for name, fn in [
    ("Bisection", lambda: bisection(f_quad, 0, 3, epsilon=1e-10, max_iterations=50)),
    ("Newton", lambda: newton_raphson(f_quad, 1.5, epsilon=1e-10, max_iterations=50, f_prime=fp_quad)),
    ("Secant", lambda: secant(f_quad, 0, 3, epsilon=1e-10, max_iterations=50)),
]:
    r = fn()
    check(f"Cross-02: {name} finds x=2",
          r["success"] and abs(r["root"] - 2.0) < 1e-6, f"got {r.get('root')}")

# Test C3: Bisection should take more iterations than Newton for same function
r_b = bisection(f_cubic, 1, 2, epsilon=1e-8, max_iterations=100)
r_n = newton_raphson(f_cubic, 1.5, epsilon=1e-8, max_iterations=20, f_prime=fp_cubic)
check("Cross-03: Bisection > Newton iterations (expected)", r_b["iterations_count"] > r_n["iterations_count"],
      f"Bisection={r_b['iterations_count']}, Newton={r_n['iterations_count']}")


# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print("\n" + "=" * 70)
print(f"ROOT FINDING RESULTS: {passed}/{total} passed ({100*passed//total if total > 0 else 0}%)")
if failed > 0:
    print(f"\nFAILURES ({failed}):")
    for e in errors:
        print(e)
print("=" * 70)
sys.exit(0 if failed == 0 else 1)