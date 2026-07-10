"""Test Newton-Schulz iterative inverse method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.matrix_inverse import matrix_inverse_newton


def test_newton_3x3():
    A = [
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2],
    ]
    result = matrix_inverse_newton(A, epsilon=1e-6, max_iterations=50)
    assert result["success"], f"Newton failed: {result.get('message')}"
    inv = result["inverse"]
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = sum(A[i][k] * inv[k][j] for k in range(n))
            expected = 1.0 if i == j else 0.0
            assert abs(s - expected) < 1e-5, f"A*A^-1[{i}][{j}] = {s}, expected {expected}"
    print(f"[PASS] Newton-Schulz 3x3 ({result['iterations_count']} iterations)")


def test_newton_singular():
    A = [
        [1, 2],
        [2, 4],
    ]
    result = matrix_inverse_newton(A)
    assert not result["success"], "Should reject singular matrix"
    print("[PASS] Newton-Schulz singular (correctly rejected)")


if __name__ == "__main__":
    test_newton_3x3()
    test_newton_singular()
    print("\n=== All Newton-Schulz inverse tests passed! ===")