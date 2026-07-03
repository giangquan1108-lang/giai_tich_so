"""Thomas Algorithm validation tests."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import thomas_algorithm


def test_valid_tridiagonal():
    """Test Case 1: valid tridiagonal matrix"""
    A = [[10, -1, 0], [-1, 11, -1], [0, -1, 10]]
    B = [[9], [9], [9]]
    result = thomas_algorithm(A, B)
    assert result["success"], f"Expected success, got: {result['message']}"
    sol = result["solution"]
    assert sol is not None
    import numpy as np
    Ax = np.dot(np.array(A, dtype=float), np.array(sol))
    assert np.allclose(Ax, B, atol=1e-6), f"Ax != B"
    print(f"  [OK] Test 1: Valid tridiagonal -> x={sol}")

def test_non_tridiagonal():
    """Test Case 2: non-tridiagonal matrix rejected"""
    A = [[10, -1, 2], [-1, 11, -1], [2, -1, 10]]
    B = [[9], [9], [9]]
    result = thomas_algorithm(A, B)
    assert not result["success"], "Should fail for non-tridiagonal"
    msg = result["message"].lower()
    assert any(word in msg for word in ["ba duong cheo", "ba đường chéo"]), f"Bad msg: {result['message']}"
    assert result.get("solution") is None
    print("  [OK] Test 2: Non-tridiagonal correctly rejected")

def test_non_square():
    """Test Case 3: non-square matrix rejected"""
    A = [[10, -1, 0, 0], [-1, 11, -1, 0], [0, -1, 10, 0]]
    B = [[9], [9], [9]]
    result = thomas_algorithm(A, B)
    assert not result["success"], "Should fail for non-square"
    assert "Thomas" in result["message"]
    print("  [OK] Test 3: Non-square correctly rejected")

def test_zero_diagonal():
    """Edge case: zero on main diagonal before forward sweep"""
    A = [[0, -1, 0], [-1, 11, -1], [0, -1, 10]]
    B = [[9], [9], [9]]
    result = thomas_algorithm(A, B)
    assert not result["success"], "Should fail for zero main diagonal"
    print("  [OK] Test 4: Zero diagonal correctly rejected")

def test_zero_pivot_after_sweep():
    """Edge case: matrix that becomes singular during sweep"""
    A = [[1, 2, 0], [2, 4, 3], [0, 3, 5]]
    B = [[3], [7], [8]]
    result = thomas_algorithm(A, B)
    assert not result["success"], "Should detect singularity during sweep"
    print("  [OK] Test 5: Pivot collapse during forward sweep detected")


if __name__ == "__main__":
    print("=== Thomas Algorithm Validation ===")
    print()
    test_valid_tridiagonal()
    test_non_tridiagonal()
    test_non_square()
    test_zero_diagonal()
    test_zero_pivot_after_sweep()
    print()
    print("=" * 40)
    print("  ALL THOMAS TESTS PASSED")
    print("=" * 40)