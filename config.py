"""
config.py — Central Configuration for CDKS-LSS PPFL
=====================================================

This file contains all configurable parameters for the implementation of:

    "Privacy-preserving federated learning via secret sharing 
     and multi-key homomorphic encryption"
    by Yuntao Wang, Fumiya Inoue, Yujie Gu, Xun Shen, Mingwu Zhang
    Information Sciences, 2026

IMPORTANT — EDUCATIONAL IMPLEMENTATION:
    All parameters are chosen for clarity, NOT for cryptographic security.
    A production implementation would use n >= 4096, q ~ 2^100+, etc.
    
References:
    - Section 2.2: RLWE parameters
    - Section 2.4: CDKS scheme parameters
    - Section 2.5: Shamir LSS parameters
    - Section 4:   CDKS-LSS parameters
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display window needed)

import numpy as np

# ============================================================================
# 1. RING PARAMETERS  (Section 2.2)
# ============================================================================
# The ring is R_q = Z_q[X] / (X^n + 1)
#
# n : polynomial degree (must be a power of 2 for the cyclotomic ring)
# q : ciphertext modulus (must be a prime for modular inverse to exist)
#
# Educational defaults:
#   n = 64   → small enough to trace computations
#   q = 2^20 + 7 = 1048583  → a prime near 2^20, large enough for small noise
# ============================================================================

RING_N = 64                    # Polynomial degree (power of 2)
RING_Q = 1048583               # Ciphertext modulus (prime)

# ============================================================================
# 2. NOISE / SAMPLING PARAMETERS  (Section 2.2)
# ============================================================================
# χ : secret key distribution  — ternary {-1, 0, 1}
# ψ : error distribution       — small bounded integers in [-ERROR_BOUND, ERROR_BOUND]
# φ : smudging noise (for partial decryption) — larger than ψ
#
# The paper uses discrete Gaussian for ψ. For simplicity, we use a 
# bounded uniform distribution, which is a standard educational substitute.
# ============================================================================

ERROR_BOUND = 3                # Coefficients of error polynomials in [-3, 3]
SMUDGE_BOUND = 50              # Smudging noise bound for partial decryption (φ)

# ============================================================================
# 3. CDKS SCHEME PARAMETERS  (Section 2.4)
# ============================================================================
# The CDKS scheme uses the ring parameters above.
# Plaintext space: we encode integer model weights directly as polynomial 
# coefficients. The plaintext must satisfy |m| << q/2 for correct decryption.
#
# DELTA : scaling factor for encoding (set to 1 for integer encoding)
# ============================================================================

DELTA = 1                      # Plaintext scaling factor

# ============================================================================
# 4. FEDERATED LEARNING PARAMETERS
# ============================================================================
# N : number of FL participants (clients)
# t : secret sharing threshold  (t <= N, need t shares to reconstruct)
#
# Paper experiments use various N; we default to small values for clarity.
# ============================================================================

NUM_CLIENTS = 5                # N = number of participants
THRESHOLD = 3                  # t = LSS reconstruction threshold

# ============================================================================
# 5. MACHINE LEARNING PARAMETERS
# ============================================================================
# We use logistic regression for simplicity.
# The model weights are the main "payload" that gets encrypted.
# ============================================================================

NUM_FEATURES = 20              # Dimensionality of feature space
NUM_SAMPLES_PER_CLIENT = 100   # Local dataset size per client
NUM_CLASSES = 2                # Binary classification
LEARNING_RATE = 0.1            # SGD learning rate
LOCAL_EPOCHS = 5               # Local training epochs per FL round
FL_ROUNDS = 20                 # Number of federated learning rounds
RANDOM_SEED = 42               # For reproducibility

# ============================================================================
# 6. ENCODING PARAMETERS
# ============================================================================
# To encrypt float model weights as ring elements, we quantize them:
#   quantized_weight = round(weight * WEIGHT_SCALE)
#
# The quantized values must satisfy |quantized_weight| << q/2.
# After decryption and aggregation, we divide by WEIGHT_SCALE to recover floats.
# ============================================================================

WEIGHT_SCALE = 1000            # Quantization scale for float → int

# ============================================================================
# 7. SHAMIR LSS PARAMETERS  (Section 2.5)
# ============================================================================
# For classic Shamir over Z_p, we need a prime p.
# For Shamir over R_q, the evaluation points must form an "exceptional sequence"
# (Definition 2): α_i - α_j must be a unit in R_q for all i ≠ j.
#
# When α_i are small distinct integers embedded as constants in R_q,
# (α_i - α_j) is a nonzero integer. It is a unit in Z_q when q is prime
# and |α_i - α_j| < q, which holds for our parameters.
# ============================================================================

SHAMIR_PRIME = 104729          # Prime for classic Shamir demo over Z_p

# Evaluation points for ring Shamir (small distinct positive integers)
# These are embedded as constant polynomials in R_q.
def get_evaluation_points(n_clients):
    """
    Generate evaluation points α_1, ..., α_n for Shamir LSS.
    
    We use 1, 2, ..., n_clients as evaluation points.
    Since q is prime and all differences |α_i - α_j| < q,
    every difference is invertible in Z_q, satisfying the
    exceptional sequence requirement (Definition 2 in the paper).
    """
    return list(range(1, n_clients + 1))

# ============================================================================
# 8. EXPERIMENT PARAMETERS
# ============================================================================

DROPOUT_COUNTS = [0, 1, 2, 3, 4]   # Number of clients that drop out
RESULTS_DIR = "results"             # Directory for output plots/tables

# ============================================================================
# 9. DISPLAY PARAMETERS
# ============================================================================

VERBOSE = True                 # Print intermediate values for educational purposes
np.set_printoptions(linewidth=120, precision=4)
