"""Test Jacobi iterative inverse method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.matrix_inverse import matrix_inverse_jacobi


def test_jacobi_3x3():
    A = [
        [10, 1, 1],
        [1, 10, 1],
        [1, 1, 10],
    ]
    result = matrix_inverse_jacobi(A, epsilon=1e-6, max_iterations=500)
    assert result["success"], f"Jacobi failed: {result.get('message')}"
    inv = result["inverse"]
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = sum(A[i][k] * inv[k][j] for k in range(n))
            expected = 1.0 if i == j else 0.0
            assert abs(s - expected) < 1e-5, f"A*A^-1[{i}][{j}] = {s}, expected {expected}"
    print(f"[PASS] Jacobi 3x3 ({result['iterations_count']} iterations)")


def test_jacobi_not_diagonally_dominant():
    A = [
        [1, 5],
        [5, 1],
    ]
    result = matrix_inverse_jacobi(A, epsilon=1e-6, max_iterations=100)
    assert "message" in result
    rho = result.get('spectral_radius', 'N/A')
    print(f"[PASS] Jacobi non-DD: success={result['success']}, rho={rho}")


def test_jacobi_singular():
    A = [
        [1, 2],
        [2, 4],
    ]
    result = matrix_inverse_jacobi(A)
    assert not result["success"], "Should reject singular matrix"
    print("[PASS] Jacobi singular (correctly rejected)")


if __name__ == "__main__":
    test_jacobi_3x3()
    test_jacobi_not_diagonally_dominant()
    test_jacobi_singular()
    print("\n=== All Jacobi inverse tests passed! ===")