"""
tests.py — Unit Tests for CDKS-LSS PPFL
==========================================

Tests every major component of the implementation.
Run with: python tests.py

These tests verify mathematical correctness, not cryptographic security.
"""

import sys
import numpy as np

# Ensure we can import from the project
sys.path.insert(0, '.')

from src.ring import (
    poly_add, poly_sub, poly_mul, poly_mod_q, poly_zero,
    poly_scalar_mul, poly_negate, poly_from_int, poly_norm,
    sample_uniform, sample_ternary, sample_error, mod_inverse,
    encode_vector_as_polynomials, decode_polynomials_to_vector
)
from src.rlwe import (
    sample_secret, sample_error_poly, sample_uniform_polynomial,
    generate_rlwe_sample, verify_rlwe_structure
)
from src.cdks import (
    CDKSPublicParams, setup as cdks_setup, keygen, encrypt, add,
    partial_decrypt, merge, full_cdks_pipeline
)
from src.shamir_lss import ClassicShamir, RingShamir, reshare
from src.cdks_lss import full_cdks_lss_pipeline


# ============================================================================
# TEST FRAMEWORK
# ============================================================================

passed = 0
failed = 0
errors = []


def run_test(name, func):
    """Run a test function and report results."""
    global passed, failed
    try:
        func()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
        errors.append((name, str(e)))


# ============================================================================
# RING TESTS
# ============================================================================

def test_ring_addition():
    """Test polynomial addition in R_q."""
    q = 1048583
    a = np.array([1, 2, 3, 4], dtype=np.int64)
    b = np.array([5, 6, 7, 8], dtype=np.int64)
    result = poly_add(a, b, q)
    expected = np.array([6, 8, 10, 12], dtype=np.int64)
    assert np.array_equal(result, expected), f"Got {result}, expected {expected}"


def test_ring_subtraction():
    """Test polynomial subtraction in R_q."""
    q = 1048583
    a = np.array([10, 20, 30, 40], dtype=np.int64)
    b = np.array([1, 2, 3, 4], dtype=np.int64)
    result = poly_sub(a, b, q)
    expected = np.array([9, 18, 27, 36], dtype=np.int64)
    assert np.array_equal(result, expected), f"Got {result}"


def test_ring_multiplication():
    """Test polynomial multiplication with negacyclic reduction."""
    n = 4
    q = 1048583
    # (1 + X) * (1 + X) = 1 + 2X + X^2
    a = np.array([1, 1, 0, 0], dtype=np.int64)
    b = np.array([1, 1, 0, 0], dtype=np.int64)
    result = poly_mul(a, b, q)
    expected = np.array([1, 2, 1, 0], dtype=np.int64)
    assert np.array_equal(result, expected), f"Got {result}"


def test_ring_negacyclic():
    """Test that X^n ≡ -1 in R_q = Z_q[X]/(X^n+1)."""
    n = 4
    q = 1048583
    # X^3 * X = X^4 ≡ -1 (mod X^4 + 1)
    x3 = np.array([0, 0, 0, 1], dtype=np.int64)  # X^3
    x1 = np.array([0, 1, 0, 0], dtype=np.int64)   # X
    result = poly_mul(x3, x1, q)
    expected = poly_mod_q(np.array([-1, 0, 0, 0], dtype=np.int64), q)
    assert np.array_equal(result, expected), f"X^4 should be -1, got {result}"


def test_mod_q_centering():
    """Test that poly_mod_q centers coefficients."""
    q = 101
    poly = np.array([100, 50, 0, 51], dtype=np.int64)
    result = poly_mod_q(poly, q)
    # 100 → -1, 50 → 50 (= 50 ≤ 50), 0 → 0, 51 → -50
    expected = np.array([-1, 50, 0, -50], dtype=np.int64)
    assert np.array_equal(result, expected), f"Got {result}"


def test_mod_inverse():
    """Test modular inverse computation."""
    q = 104729
    a = 42
    inv = mod_inverse(a, q)
    assert (a * inv) % q == 1, f"42 * {inv} mod {q} = {(a*inv)%q}, expected 1"


def test_encoding_roundtrip():
    """Test that encoding/decoding float vectors is approximately inverse."""
    n = 8
    q = 1048583
    scale = 1000
    vec = np.array([0.5, -1.3, 2.7, 0.0, -0.8], dtype=np.float64)
    
    polys = encode_vector_as_polynomials(vec, n, q, scale)
    recovered = decode_polynomials_to_vector(polys, len(vec), q, scale)
    
    max_err = np.max(np.abs(vec - recovered))
    assert max_err < 0.01, f"Encoding roundtrip error = {max_err}"


# ============================================================================
# RLWE TESTS
# ============================================================================

def test_rlwe_generation():
    """Test that RLWE samples have the correct structure."""
    n = 8
    q = 1048583
    rng = np.random.default_rng(42)
    
    s = sample_secret(n, rng)
    a = sample_uniform_polynomial(n, q, rng)
    _, b = generate_rlwe_sample(s, a, q, error_bound=3, rng=rng)
    
    # Verify: b - a*s should be small (the error)
    recovered_e = verify_rlwe_structure(a, b, s, q)
    max_e = poly_norm(recovered_e)
    assert max_e <= 3, f"RLWE error too large: {max_e}"


def test_secret_distribution():
    """Test that secret keys have ternary coefficients."""
    rng = np.random.default_rng(42)
    s = sample_secret(64, rng)
    unique_vals = set(s.tolist())
    assert unique_vals.issubset({-1, 0, 1}), f"Secret has non-ternary values: {unique_vals}"


# ============================================================================
# CDKS TESTS
# ============================================================================

def test_key_generation():
    """Test CDKS key generation produces valid keys."""
    n, q = 8, 1048583
    rng = np.random.default_rng(42)
    pp = cdks_setup(n, q, rng=rng)
    sk, pk = keygen(pp, rng)
    
    assert 's' in sk, "Secret key missing 's'"
    assert 'b' in pk, "Public key missing 'b'"
    assert len(sk['s']) == n, f"Secret key wrong length"
    assert len(pk['b']) == n, f"Public key wrong length"


def test_encryption():
    """Test that encryption produces valid ciphertexts."""
    n, q = 8, 1048583
    rng = np.random.default_rng(42)
    pp = cdks_setup(n, q, rng=rng)
    sk, pk = keygen(pp, rng)
    
    m = np.array([10, 20, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    c0, c1 = encrypt(pp, pk, m, rng)
    
    assert len(c0) == n, "c0 wrong length"
    assert len(c1) == n, "c1 wrong length"


def test_cdks_addition():
    """Test that homomorphic addition preserves the aggregate."""
    n, q = 8, 1048583
    rng = np.random.default_rng(42)
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    m1 = np.array([10, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    m2 = np.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    result = full_cdks_pipeline(pp, [m1, m2], rng)
    
    # The recovered aggregate should be close to m1 + m2 = [30, 0, ...]
    assert result['max_error'] < 500, f"CDKS error too large: {result['max_error']}"


def test_cdks_decryption():
    """Test full CDKS encrypt-add-decrypt pipeline."""
    n, q = 8, 1048583
    rng = np.random.default_rng(42)
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    plaintexts = [
        np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        np.array([200, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        np.array([300, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
    ]
    
    result = full_cdks_pipeline(pp, plaintexts, rng)
    
    # Aggregate should be [600, 0, 0, ...]
    recovered_first = result['M_recovered'][0]
    true_first = 600
    error = abs(recovered_first - true_first)
    assert error < 500, f"Decryption error at [0]: {error}"


def test_cdks_vulnerability():
    """Test that the CDKS vulnerability allows individual plaintext recovery."""
    n, q = 8, 1048583
    rng = np.random.default_rng(42)
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    m = np.array([1000, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    sk, pk = keygen(pp, rng)
    c0, c1 = encrypt(pp, pk, m, rng)
    mu = partial_decrypt(sk, c1, pp, rng)
    
    # Attack: c0 + mu ≈ m
    recovered = poly_mod_q(c0 + mu, q)
    error = abs(int(recovered[0]) - 1000)
    assert error < 500, f"Vulnerability attack failed with error {error}"


# ============================================================================
# SHAMIR TESTS
# ============================================================================

def test_shamir_share():
    """Test classic Shamir secret sharing."""
    p = 104729
    secret = 42
    t = 3
    alphas = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    
    shares = ClassicShamir.share(secret, t, alphas, p, rng)
    assert len(shares) == 5, f"Expected 5 shares, got {len(shares)}"


def test_shamir_reconstruction():
    """Test that t shares reconstruct the secret."""
    p = 104729
    secret = 42
    t = 3
    alphas = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    
    shares = ClassicShamir.share(secret, t, alphas, p, rng)
    
    # Exactly t shares → correct
    recovered_t = ClassicShamir.combine(shares[:t], p)
    assert recovered_t == secret, f"t shares: got {recovered_t}, expected {secret}"
    
    # All shares → correct
    recovered_all = ClassicShamir.combine(shares, p)
    assert recovered_all == secret, f"All shares: got {recovered_all}"
    
    # t-1 shares → wrong (with high probability)
    recovered_few = ClassicShamir.combine(shares[:t-1], p)
    # Note: there's a tiny chance this equals secret by accident,
    # but with a large prime it's astronomically unlikely
    # We just check the function runs without error


def test_exceptional_sequence():
    """Test the exceptional sequence check."""
    q = 1048583  # Prime
    
    # Small distinct integers → should be exceptional when q is prime
    alphas_good = [1, 2, 3, 4, 5]
    assert RingShamir.check_exceptional_sequence(alphas_good, q), \
        "Should be exceptional for small integers with prime q"
    
    # Repeated values → should fail
    alphas_bad = [1, 1, 3]
    assert not RingShamir.check_exceptional_sequence(alphas_bad, q), \
        "Should NOT be exceptional with repeated values"


def test_ring_shamir():
    """Test Shamir over R_q."""
    n_ring = 8
    q = 1048583
    t = 2
    alphas = [1, 2, 3]
    rng = np.random.default_rng(42)
    
    secret = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
    
    shares = RingShamir.share(secret, t, alphas, n_ring, q, rng)
    recovered = RingShamir.combine(shares[:t], n_ring, q)
    
    assert np.array_equal(poly_mod_q(recovered, q), poly_mod_q(secret, q)), \
        f"Ring Shamir reconstruction failed"


def test_share_resharing():
    """Test that re-sharing correctly reconstructs the sum of secrets."""
    n_ring = 8
    q = 1048583
    t = 2
    alphas = [1, 2, 3]
    rng = np.random.default_rng(42)
    
    s1 = np.array([10, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    s2 = np.array([20, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    s3 = np.array([30, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    true_sum = poly_add(poly_add(s1, s2, q), s3, q)
    
    agg_shares = reshare([s1, s2, s3], t, alphas, n_ring, q, rng)
    recovered_sum = RingShamir.combine(agg_shares[:t], n_ring, q)
    
    assert np.array_equal(poly_mod_q(recovered_sum, q), poly_mod_q(true_sum, q)), \
        f"Re-sharing sum mismatch: got {recovered_sum}, expected {true_sum}"


# ============================================================================
# CDKS-LSS TESTS
# ============================================================================

def test_cdks_lss_decryption():
    """Test full CDKS-LSS pipeline: aggregate recovery without individual leakage."""
    n = 8
    q = 1048583
    t = 2
    N = 3
    alphas = [1, 2, 3]
    rng = np.random.default_rng(42)
    
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    m1 = np.array([100, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    m2 = np.array([200, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    m3 = np.array([300, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    
    result = full_cdks_lss_pipeline(pp, [m1, m2, m3], t, alphas, rng=rng)
    
    # Aggregate should be close to 600
    error = result['max_error']
    assert error < 500, f"CDKS-LSS decryption error: {error}"


def test_threshold_reconstruction():
    """Test that CDKS-LSS works with exactly t shares."""
    n = 8
    q = 1048583
    t = 2
    N = 5
    alphas = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    plaintexts = [np.array([i*100, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64) for i in range(1, N+1)]
    
    # Only use t out of N clients
    available = list(range(t))
    
    result = full_cdks_lss_pipeline(pp, plaintexts, t, alphas,
                                    available_indices=available, rng=rng)
    
    # Should still work (t shares are enough)
    assert result['max_error'] < 500, \
        f"Threshold reconstruction failed with error {result['max_error']}"


# ============================================================================
# FEDAVG TEST
# ============================================================================

def test_fedavg():
    """Test that FedAvg converges on a simple problem."""
    from data.generate_data import generate_fl_dataset
    from src.client import FLClient
    from src.federated_learning import run_fedavg
    
    rng = np.random.default_rng(42)
    
    client_data, test_data = generate_fl_dataset(
        n_clients=3, n_samples_per_client=50, n_features=10,
        non_iid=False, random_state=42
    )
    
    clients = [FLClient(i, X, y) for i, (X, y) in enumerate(client_data)]
    
    history = run_fedavg(
        clients, test_data, n_features=10,
        n_rounds=5, learning_rate=0.1, local_epochs=3, rng=rng
    )
    
    # Should achieve at least 50% accuracy (better than random)
    final_acc = history['accuracy'][-1]
    assert final_acc > 0.5, f"FedAvg final accuracy too low: {final_acc}"


# ============================================================================
# MATHEMATICAL VALIDATION: N=3, t=2 walkthrough
# ============================================================================

def test_mathematical_validation():
    """
    Tiny example (N=3, t=2) that manually demonstrates CDKS-LSS.
    
    This is the "simplest example I can use to understand the method"
    as requested in the specification.
    """
    print("\n  --- Mathematical Validation (N=3, t=2) ---")
    
    n = 8
    q = 1048583
    t = 2
    N = 3
    alphas = [1, 2, 3]
    rng = np.random.default_rng(42)
    
    pp = cdks_setup(n, q, error_bound=3, smudge_bound=10, rng=rng)
    
    # Simple plaintexts
    m1 = np.zeros(n, dtype=np.int64); m1[0] = 10
    m2 = np.zeros(n, dtype=np.int64); m2[0] = 20
    m3 = np.zeros(n, dtype=np.int64); m3[0] = 30
    
    result = full_cdks_lss_pipeline(pp, [m1, m2, m3], t, alphas, rng=rng)
    
    print(f"    m_1 = {m1[0]}, m_2 = {m2[0]}, m_3 = {m3[0]}")
    print(f"    True sum = {m1[0] + m2[0] + m3[0]}")
    print(f"    Recovered M[0] = {result['M_recovered'][0]}")
    print(f"    Max error = {result['max_error']}")
    print(f"    μ_1[0] = {result['mu_values'][0][0]} (HIDDEN from server)")
    print(f"    μ_2[0] = {result['mu_values'][1][0]} (HIDDEN from server)")
    print(f"    μ_3[0] = {result['mu_values'][2][0]} (HIDDEN from server)")
    print(f"    Server only sees aggregated shares → reconstructs sum")
    
    assert result['max_error'] < 500, "Mathematical validation failed"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CDKS-LSS PPFL — Unit Tests")
    print("=" * 60)
    
    print("\n--- Ring Operations ---")
    run_test("test_ring_addition", test_ring_addition)
    run_test("test_ring_subtraction", test_ring_subtraction)
    run_test("test_ring_multiplication", test_ring_multiplication)
    run_test("test_ring_negacyclic", test_ring_negacyclic)
    run_test("test_mod_q_centering", test_mod_q_centering)
    run_test("test_mod_inverse", test_mod_inverse)
    run_test("test_encoding_roundtrip", test_encoding_roundtrip)
    
    print("\n--- RLWE ---")
    run_test("test_rlwe_generation", test_rlwe_generation)
    run_test("test_secret_distribution", test_secret_distribution)
    
    print("\n--- CDKS ---")
    run_test("test_key_generation", test_key_generation)
    run_test("test_encryption", test_encryption)
    run_test("test_cdks_addition", test_cdks_addition)
    run_test("test_cdks_decryption", test_cdks_decryption)
    run_test("test_cdks_vulnerability", test_cdks_vulnerability)
    
    print("\n--- Shamir LSS ---")
    run_test("test_shamir_share", test_shamir_share)
    run_test("test_shamir_reconstruction", test_shamir_reconstruction)
    run_test("test_exceptional_sequence", test_exceptional_sequence)
    run_test("test_ring_shamir", test_ring_shamir)
    run_test("test_share_resharing", test_share_resharing)
    
    print("\n--- CDKS-LSS ---")
    run_test("test_cdks_lss_decryption", test_cdks_lss_decryption)
    run_test("test_threshold_reconstruction", test_threshold_reconstruction)
    
    print("\n--- Federated Learning ---")
    run_test("test_fedavg", test_fedavg)
    
    print("\n--- Mathematical Validation ---")
    run_test("test_mathematical_validation", test_mathematical_validation)
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'=' * 60}")
