"""End-to-end test: A(5x5) B(5x2)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from app.algorithms.linear_system import gaussian_elimination

print("=== TEST 1: Unique A(5x5) B(5x2) ===")
A = [[10,0,1,0,0],[1,9,0,2,0],[0,2,8,0,1],[0,0,3,7,0],[1,0,0,0,12]]
B = [[1,100],[2,200],[3,300],[4,400],[5,500]]
r = gaussian_elimination(A, B)
s = r['solution']
print(f"  Success: {r['success']}")
print(f"  Type: {r['solution_type']}")
print(f"  Shape: {len(s)}x{len(s[0])}")
assert len(s) == 5, f"Expected 5 rows, got {len(s)}"
assert len(s[0]) == 2, f"Expected 2 cols, got {len(s[0])}"
assert np.allclose(np.array(s), np.linalg.solve(np.array(A), np.array(B)), atol=1e-6)
print("  PASS: Shape 5x2, matches numpy")

print("\n=== TEST 2: Infinite A(3x5) B(3x2) ===")
A2 = [[1,1,1,1,1],[2,2,2,2,2],[1,2,3,4,5]]
B2 = [[5,50],[10,100],[15,150]]
r = gaussian_elimination(A2, B2)
print(f"  Success: {r['success']}")
print(f"  Type: {r['solution_type']}")
ps = r.get('particular_solution')
if ps:
    print(f"  x_p shape: {len(ps)}x{len(ps[0])}")
    print(f"  x_p: {ps}")
    assert len(ps) == 5, f"Expected 5 rows, got {len(ps)}"
    assert len(ps[0]) == 2, f"Expected 2 cols, got {len(ps[0])}"
    print("  PASS: x_p shape 5x2 for infinite solution")
else:
    print("  FAIL: No particular_solution returned")

print(f"\n  free_vars: {r.get('free_variables')}")
print(f"  basis count: {len(r.get('basis_vectors',[]))}")
print("\n=== ALL E2E TESTS PASSED ===")