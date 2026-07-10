"""Test Gauss-Seidel iterative inverse method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.matrix_inverse import matrix_inverse_gauss_seidel


def test_gs_3x3():
    A = [
        [10, 1, 1],
        [1, 10, 1],
        [1, 1, 10],
    ]
    result = matrix_inverse_gauss_seidel(A, epsilon=1e-6, max_iterations=300)
    assert result["success"], f"Gauss-Seidel failed: {result.get('message')}"
    inv = result["inverse"]
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = sum(A[i][k] * inv[k][j] for k in range(n))
            expected = 1.0 if i == j else 0.0
            assert abs(s - expected) < 1e-5, f"A*A^-1[{i}][{j}] = {s}, expected {expected}"
    print(f"[PASS] Gauss-Seidel 3x3 ({result['iterations_count']} iterations)")


def test_gs_singular():
    A = [
        [1, 2],
        [2, 4],
    ]
    result = matrix_inverse_gauss_seidel(A)
    assert not result["success"], "Should reject singular matrix"
    print("[PASS] Gauss-Seidel singular (correctly rejected)")


if __name__ == "__main__":
    test_gs_3x3()
    test_gs_singular()
    print("\n=== All Gauss-Seidel inverse tests passed! ===")