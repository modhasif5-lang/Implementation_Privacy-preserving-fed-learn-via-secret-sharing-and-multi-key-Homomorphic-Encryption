# Implementation_Privacy-preserving-fed-learn-via-secret-sharing-and-multi-key-Homomorphic-Encryption

# CDKS-LSS Privacy-Preserving Federated Learning

**Implementation of:**

> "Privacy-preserving federated learning via secret sharing and multi-key homomorphic encryption"
> Yuntao Wang, Fumiya Inoue, Yujie Gu, Xun Shen, Mingwu Zhang
> *Information Sciences*, 2026

---

## Educational Implementation

This is an **academic research prototype** designed for understanding the paper's algorithms.
It is **NOT** a production cryptographic library.

- Cryptographic parameters (n=64, q≈2^20) are chosen for clarity, not security
- Error distributions use bounded uniform instead of discrete Gaussian
- Ring arithmetic uses naive O(n²) multiplication instead of NTT
- The implementation demonstrates mathematical structure, not production security

---

## 1. Paper Summary

### Problem
In Federated Learning (FL), clients train models locally and send updates to a server.
Standard FL exposes raw model updates, enabling gradient inversion attacks.

### Existing Approach: CDKS Multi-Key HE
The CDKS scheme allows encrypted aggregation where each client encrypts with their own key.
However, during **partial decryption**, the server can recover individual plaintexts:

```
c_{i,0} + μ_i ≈ m_i   ← VULNERABILITY!
```

### Proposed Solution: CDKS-LSS
Instead of sending partial decryptions μ_i directly, clients **secret-share** μ_i
among themselves using Shamir's Linear Secret Sharing. The server receives only
**aggregated shares** that reconstruct the sum Σμ_i, not individual values.

### Key Advantages
| Feature              | FedAvg | xMK-CKKS | CDKS-LSS |
|----------------------|--------|----------|----------|
| Encrypted Updates    | ✗      | ✓       | ✓        |
| Individual Privacy   | ✗      | ✗       | ✓        |
| Dropout Tolerance    | ✓*     | ✗       | ✓        |

*FedAvg has no crypto, so dropout is trivial but privacy is absent.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CDKS-LSS Protocol                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────┐  global model   ┌──────────┐             │
│   │  Server  │ ──────────────► │ Client i │             │
│   └──────────┘                 └──────────┘             │
│        ▲                            │                   │
│        │                      local training            │
│        │                            │                   │
│        │                      encrypt update            │
│        │                    (CDKS.Enc with pk_i)        │
│        │                            │                   │
│   ┌────┴─────┐  send c_{i,0}  ┌────▼─────┐              │
│   │ Aggregate │ ◄──────────── │ compute  │              │
│   │  Σc_{i,0} │               │   μ_i    │              │
│   └──────────┘                └──────────┘              │
│        │                            │                   │
│        │                    secret-share μ_i            │
│        │                   (Shamir over R_q)            │
│        │                            │                   │
│        │                  exchange shares with          │
│        │                    other clients               │
│        │                            │                   │
│        │                    aggregate shares            │
│        │                     s̃_j = Σ f_i(α_j)           │
│        │                            │                   │
│   ┌────┴─────┐  send s̃_j    ┌─────▼────┐                │
│   │ Lagrange │ ◄──────────── │  Client  │               │
│   │  interp  │               │   j      │               │
│   │  → Σμ_i  │               └──────────┘               │
│   └──────────┘                                          │
│        │                                                │
│   M = c_0 + Σμ_i ≈ Σm_i                                 │
│   w_G = M / N                                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### Ring
```
R_q = Z_q[X] / (X^n + 1)
```

### CDKS Key Generation
```
s_i ← χ (ternary)
e_i ← ψ (small error)
b_i = -a·s_i + e_i
sk_i = (1, s_i),  pk_i = (b_i, a)
```

### CDKS Encryption
```
c_{i,0} = v_i·b_i + m_i + e_{i,0}  (mod q)
c_{i,1} = v_i·a + e_{i,1}          (mod q)
```

### CDKS-LSS Decryption
```
1. μ_i = c_{i,1} · s_i
2. f_i(X): degree t-1 polynomial with f_i(0) = μ_i
3. Exchange: f_i(α_j) sent to participant j
4. Aggregate: s̃_j = Σ_i f_i(α_j)
5. Server: Lagrange interpolation of t shares → Σμ_i
6. M = c_0 + Σμ_i ≈ Σm_i
```

---

## 4. Installation

```bash
cd cdks_lss_ppfl
pip install -r requirements.txt
```

Requirements: Python 3.8+, NumPy, scikit-learn, matplotlib, pandas, tabulate

---

## 5. Running the Experiments

### Run everything
```bash
python main.py
```

### Run unit tests
```bash
python tests.py
```

### Run specific experiments
```bash
python main.py --demo           # Component demonstrations
python main.py --exp 1          # Experiment 1: FedAvg baseline
python main.py --exp 2          # Experiment 2: CDKS + vulnerability
python main.py --exp 3          # Experiment 3: CDKS-LSS FL
python main.py --exp 4          # Experiment 4: xMK-CKKS comparison
python main.py --exp 5          # Experiment 5: Security demonstration
python main.py --exp 6          # Experiment 6: Dropout tolerance
python main.py --exp 7          # Experiment 7: Communication overhead
```

### Run individual experiments
```bash
python experiments/experiment_fedavg.py
python experiments/experiment_cdks.py
python experiments/experiment_cdks_lss.py
python experiments/experiment_xmk_ckks.py
python experiments/experiment_security.py
python experiments/experiment_dropout.py
python experiments/experiment_overhead.py
```

---

## 6. Expected Results

| Experiment    | Expected Output                                   |
|---------------|---------------------------------------------------|
| FedAvg        | Accuracy converges to ~80-90%                     |
| CDKS FL       | Similar accuracy, vulnerability demonstrated      |
| CDKS-LSS FL   | Similar accuracy, individual privacy protected    |
| xMK-CKKS      | Similar accuracy, fails with dropout              |
| Security      | Attack succeeds on CDKS, fails on CDKS-LSS        |
| Dropout       | CDKS-LSS works with ≥t clients; xMK-CKKS needs all|
| Overhead      | CDKS-LSS has higher client communication          |

Plots are saved to `results/`.

---

## 7. Project Structure

```
cdks_lss_ppfl/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── config.py                  # All parameters
├── main.py                    # Entry point
├── tests.py                   # Unit tests
│
├── src/
│   ├── ring.py                # R_q arithmetic
│   ├── rlwe.py                # RLWE sampling
│   ├── cdks.py                # CDKS multi-key HE
│   ├── shamir_lss.py          # Shamir LSS (classic + ring)
│   ├── cdks_lss.py            # CDKS-LSS combined scheme
│   ├── xmk_ckks.py            # xMK-CKKS baseline
│   ├── client.py              # FL client
│   ├── server.py              # FL server
│   ├── federated_learning.py  # FL orchestration
│   ├── attacks.py             # Vulnerability demonstrations
│   ├── metrics.py             # Accuracy, loss, overhead
│   └── utils.py               # Encoding, timing
│
├── models/
│   └── logistic_regression.py # Simple ML model
│
├── data/
│   └── generate_data.py       # Synthetic dataset
│
├── experiments/               # Individual experiments
│   ├── experiment_fedavg.py
│   ├── experiment_cdks.py
│   ├── experiment_cdks_lss.py
│   ├── experiment_xmk_ckks.py
│   ├── experiment_security.py
│   ├── experiment_dropout.py
│   └── experiment_overhead.py
│
└── results/                   # Output plots and tables
```

---

## 8. Implementation Assumptions and Deviations from the Paper

### Parameter Choices
- **Ring dimension n=64**: The paper does not specify exact parameters. We use n=64 for educational clarity. Production would use n≥4096.
- **Modulus q=1048583**: A prime near 2^20. Production would use q~2^100+.
- **Error distribution**: We use bounded uniform [-3,3] instead of discrete Gaussian. Both provide "small noise" behavior.

### Simplifications
- **Integer evaluation points**: The paper's exceptional sequence (Definition 2) requires differences α_i-α_j to be units in R_q. We use small integers 1,2,...,N, which are units in Z_q when q is prime.
- **Scalar multiplication for Lagrange**: Since evaluation points are integers (constant polynomials), Lagrange coefficients are also integers, simplifying the ring interpolation.
- **No NTT**: Ring multiplication uses naive negacyclic convolution O(n²) instead of NTT O(n log n).
- **Weight encoding**: Float weights are quantized via round(w×scale) and packed as polynomial coefficients.

### What This Implementation Does NOT Prove
- It does NOT prove cryptographic security (that requires the formal analysis in Section 5)
- It does NOT demonstrate security against real-world attackers
- The educational parameters provide zero actual security
- The bounded uniform error distribution differs from the paper's discrete Gaussian

### Missing from the Paper
- Exact parameter selection guidelines
- Concrete error bounds for correctness
- Detailed encoding scheme for model weights
- Specific exceptional sequence construction for non-integer evaluation points

---
