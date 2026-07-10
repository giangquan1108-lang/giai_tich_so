"""Test Bordering (Vien quanh) inverse method."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.matrix_inverse import matrix_inverse_bordering


def test_bordering_3x3():
    A = [
        [4, 1, 1],
        [1, 3, 0],
        [1, 0, 2],
    ]
    result = matrix_inverse_bordering(A)
    assert result["success"], f"Bordering failed: {result.get('message')}"
    inv = result["inverse"]
    # Verify A*A^-1 ~ I
    n = len(A)
    for i in range(n):
        for j in range(n):
            s = sum(A[i][k] * inv[k][j] for k in range(n))
            expected = 1.0 if i == j else 0.0
            assert abs(s - expected) < 1e-8, f"A*A^-1[{i}][{j}] = {s}, expected {expected}"
    print("[PASS] Bordering 3x3")


def test_bordering_2x2():
    A = [
        [2, 1],
        [1, 3],
    ]
    result = matrix_inverse_bordering(A)
    assert result["success"], f"Bordering 2x2 failed: {result.get('message')}"
    inv = result["inverse"]
    # Known inverse: [[0.6, -0.2], [-0.2, 0.4]]
    assert abs(inv[0][0] - 0.6) < 1e-6
    assert abs(inv[0][1] - (-0.2)) < 1e-6
    assert abs(inv[1][0] - (-0.2)) < 1e-6
    assert abs(inv[1][1] - 0.4) < 1e-6
    print("[PASS] Bordering 2x2")


def test_bordering_singular():
    A = [
        [1, 2],
        [2, 4],
    ]
    result = matrix_inverse_bordering(A)
    assert not result["success"], "Should reject singular matrix, but got success"
    print("[PASS] Bordering singular (correctly rejected)")


if __name__ == "__main__":
    test_bordering_2x2()
    test_bordering_3x3()
    test_bordering_singular()
    print("\n=== All Bordering tests passed! ===")