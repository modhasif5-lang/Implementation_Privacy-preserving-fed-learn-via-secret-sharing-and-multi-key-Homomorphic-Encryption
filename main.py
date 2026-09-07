"""
main.py — CDKS-LSS Privacy-Preserving Federated Learning
==========================================================

Main entry point that runs all experiments from the paper:

    "Privacy-preserving federated learning via secret sharing 
     and multi-key homomorphic encryption"
    by Yuntao Wang, Fumiya Inoue, Yujie Gu, Xun Shen, Mingwu Zhang
    Information Sciences, 2026

Usage:
    python main.py              — Run all experiments
    python main.py --test       — Run unit tests only
    python main.py --demo       — Run small demonstrations only
    python main.py --exp N      — Run experiment N (1-7)

EDUCATIONAL NOTE:
    This is an academic research prototype, NOT a production system.
    Cryptographic parameters are chosen for clarity, not security.
"""

import sys
import os
import argparse
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR


def run_demos():
    """Run small demonstrations of each component."""
    print("\n" + "=" * 70)
    print("   COMPONENT DEMONSTRATIONS")
    print("=" * 70)
    
    from src.rlwe import demo_rlwe
    from src.cdks import demo_cdks
    from src.shamir_lss import demo_classic_shamir, demo_ring_shamir, demo_resharing
    from src.cdks_lss import demo_cdks_lss
    
    demo_rlwe()
    print()
    demo_cdks()
    print()
    demo_classic_shamir()
    print()
    demo_ring_shamir()
    print()
    demo_resharing()
    print()
    demo_cdks_lss()


def run_experiments(experiment_num=None):
    """Run experiments."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    experiments = {
        1: ("FedAvg Baseline", "experiments.experiment_fedavg"),
        2: ("CDKS + Vulnerability", "experiments.experiment_cdks"),
        3: ("CDKS-LSS FL", "experiments.experiment_cdks_lss"),
        4: ("xMK-CKKS Comparison", "experiments.experiment_xmk_ckks"),
        5: ("Security Demonstration", "experiments.experiment_security"),
        6: ("Dropout Tolerance", "experiments.experiment_dropout"),
        7: ("Communication Overhead", "experiments.experiment_overhead"),
    }
    
    if experiment_num is not None:
        if experiment_num in experiments:
            name, module = experiments[experiment_num]
            print(f"\nRunning Experiment {experiment_num}: {name}")
            mod = __import__(module, fromlist=['run'])
            mod.run()
        else:
            print(f"Unknown experiment: {experiment_num}. Available: {list(experiments.keys())}")
        return
    
    # Run all experiments
    print("\n" + "=" * 70)
    print("   CDKS-LSS PPFL — ALL EXPERIMENTS")
    print("=" * 70)
    
    total_start = time.time()
    
    for num, (name, module) in experiments.items():
        try:
            print(f"\n{'='*60}")
            print(f"  Running Experiment {num}: {name}")
            print(f"{'='*60}")
            mod = __import__(module, fromlist=['run'])
            mod.run()
        except Exception as e:
            print(f"  ERROR in experiment {num}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  All experiments completed in {total_time:.1f}s")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'='*70}")


def run_tests():
    """Run unit tests."""
    import tests
    # The tests module runs automatically when imported via __main__
    exec(open("tests.py").read())


def main():
    parser = argparse.ArgumentParser(
        description="CDKS-LSS Privacy-Preserving Federated Learning"
    )
    parser.add_argument('--test', action='store_true', help='Run unit tests')
    parser.add_argument('--demo', action='store_true', help='Run component demos')
    parser.add_argument('--exp', type=int, help='Run specific experiment (1-7)')
    parser.add_argument('--all', action='store_true', help='Run everything')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  CDKS-LSS Privacy-Preserving Federated Learning")
    print("  Paper: Wang, Inoue, Gu, Shen, Zhang")
    print("  Information Sciences, 2026")
    print("=" * 70)
    
    if args.test:
        run_tests()
    elif args.demo:
        run_demos()
    elif args.exp is not None:
        run_experiments(args.exp)
    elif args.all:
        run_demos()
        run_experiments()
    else:
        # Default: run demos and all experiments
        run_demos()
        run_experiments()


if __name__ == "__main__":
    main()
