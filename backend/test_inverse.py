"""Matrix inverse validation tests."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app.algorithms.linear_system import (
    matrix_inverse_gauss_jordan,
    matrix_inverse_adjoint,
    matrix_inverse_lu,
    matrix_inverse_cholesky,
)

INVERSE_METHODS = [
    ("gauss_jordan", matrix_inverse_gauss_jordan),
    ("adjoint", matrix_inverse_adjoint),
    ("lu", matrix_inverse_lu),
    ("cholesky", matrix_inverse_cholesky),
]


# =============================================================================
# Test 1: Invertible matrix
# =============================================================================
def test_invertible_matrix():
    """All 4 methods should successfully invert [[1,0,0],[0,2,0],[0,0,3]] (det=6)."""
    A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    for name, func in INVERSE_METHODS:
        result = func(A)
        assert result["success"], f"[{name}] Expected success, got: {result['message']}"
        assert result["inverse"] is not None
        assert result["determinant"] is not None and abs(abs(result["determinant"]) - 6.0) < 1e-6, \
            f"[{name}] |det| should be 6, got {result['determinant']}"
        assert result["rank"] == 3, f"[{name}] rank should be 3, got {result['rank']}"

        # Verify A * A^{-1} ≈ I
        inv = result["inverse"]
        import numpy as np
        A_np = np.array(A, dtype=float)
        inv_np = np.array(inv, dtype=float)
        verify = np.dot(A_np, inv_np)
        identity = np.eye(3)
        assert np.allclose(verify, identity, atol=1e-8), \
            f"[{name}] A * A^{-1} != I:\n{verify}"
        assert result["is_accurate"] == True, f"[{name}] is_accurate should be True"

        # Steps
        assert result["steps"] is not None and len(result["steps"]) > 0, \
            f"[{name}] Should have steps"

        print(f"  [OK] Test 1 (invertible) — {name} passed")


# =============================================================================
# Test 2: Singular matrix
# =============================================================================
def test_singular_matrix():
    """All methods should reject [[1,2],[2,4]] (det=0)."""
    A = [[1, 2], [2, 4]]
    for name, func in INVERSE_METHODS:
        result = func(A)
        assert not result["success"], f"[{name}] Should fail for singular matrix"
        msg = result["message"].lower()
        assert any(w in msg for w in ["suy biến", "singular", "singular"]), \
            f"[{name}] Bad message: {result['message']}"
        print(f"  [OK] Test 2 (singular) — {name} correctly rejected")


# =============================================================================
# Test 3: Non-square matrix
# =============================================================================
def test_non_square_matrix():
    """All methods should reject a 2×3 matrix."""
    A = [[1, 2, 3], [4, 5, 6]]
    for name, func in INVERSE_METHODS:
        result = func(A)
        assert not result["success"], f"[{name}] Should fail for non-square"
        msg = result["message"].lower()
        assert "không phải" in msg or "vuông" in msg, \
            f"[{name}] Bad message: {result['message']}"
        print(f"  [OK] Test 3 (non-square) — {name} correctly rejected")


# =============================================================================
# Test 4: Symmetric positive definite
# =============================================================================
def test_spd_matrix():
    """Cholesky should succeed on SPD. Other methods should also succeed."""
    # Hilbert 3×3 is symmetric positive definite
    A = [[1.0, 1/2, 1/3], [1/2, 1/3, 1/4], [1/3, 1/4, 1/5]]
    for name, func in INVERSE_METHODS:
        result = func(A)
        assert result["success"], f"[{name}] Should succeed on SPD matrix, got: {result['message']}"
        assert result["inverse"] is not None
        assert result["rank"] == 3, f"[{name}] rank should be 3"

        # Check steps include L for Cholesky
        if name == "cholesky":
            assert any("Ma trận L" in str(s) for s in result.get("steps", [])), \
                "Cholesky should have L matrix step"

        # Verify
        import numpy as np
        A_np = np.array(A, dtype=float)
        inv_np = np.array(result["inverse"], dtype=float)
        verify = np.dot(A_np, inv_np)
        identity = np.eye(3)
        assert np.allclose(verify, identity, atol=1e-6), \
            f"[{name}] Verification failed on SPD"
        print(f"  [OK] Test 4 (SPD) — {name} passed")


# =============================================================================
# Test 5: Tridiagonal matrix
# =============================================================================
def test_tridiagonal_matrix():
    """All methods should handle a tridiagonal matrix."""
    A = [[10, -1, 0], [-1, 11, -1], [0, -1, 10]]
    for name, func in INVERSE_METHODS:
        result = func(A)
        assert result["success"], f"[{name}] Should succeed on tridiagonal, got: {result['message']}"
        assert result["inverse"] is not None

        # Verify
        import numpy as np
        A_np = np.array(A, dtype=float)
        inv_np = np.array(result["inverse"], dtype=float)
        verify = np.dot(A_np, inv_np)
        identity = np.eye(3)
        assert np.allclose(verify, identity, atol=1e-8), \
            f"[{name}] Verification failed on tridiagonal"
        print(f"  [OK] Test 5 (tridiagonal) — {name} passed")


if __name__ == "__main__":
    print("=== Matrix Inverse Validation ===")
    print()
    test_invertible_matrix()
    print()
    test_singular_matrix()
    print()
    test_non_square_matrix()
    print()
    test_spd_matrix()
    print()
    test_tridiagonal_matrix()
    print()
    print("=" * 40)
    print("  ALL INVERSE TESTS PASSED")
    print("=" * 40)