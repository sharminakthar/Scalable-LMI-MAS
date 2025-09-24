"""
Analysis Runner for Multi-Agent Lur'e Systems.

This module contains the main analysis execution script that runs the computational
efficiency comparison between brute-force and scalable LMI methods.
"""

import sys
import os
import numpy as np
import cvxpy as cp

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.graph_utils import build_laplacian
from algorithms.stability_solvers import (
    solve_brute_force_circle, 
    solve_brute_force_popov,
    solve_scalable_circle, 
    solve_scalable_popov
)


def run_analysis_experiment(Nset, Gtype=1, alpha=0.09, solver=cp.SCS):
    """Run the analysis experiment from the notebook."""
    results = []
    for N in Nset:
        L = build_laplacian(N, Gtype)
        A = np.array([[-2, 1, 0],
                      [1, -1, 1],
                      [0, -2, 0]], dtype=float)
        B = np.array([[0], [0], [3]], dtype=float)
        C = np.array([[1, 0, 0]], dtype=float)
        
        out_circ = solve_brute_force_circle(A, B, C, L, alpha, solver=solver)
        out_pop = solve_brute_force_popov(A, B, C, L, alpha, solver=solver)
        out_circ_scale = solve_scalable_circle(A, B, C, L, alpha, solver=solver)
        out_pop_scale = solve_scalable_popov(A, B, C, L, alpha, solver=solver)
        
        results.append({
            "N": N,
            "circ": out_circ,
            "pop": out_pop,
            "circ_scale": out_circ_scale,
            "pop_scale": out_pop_scale
        })
    return results


def print_results_table(results):
    """Print the results table from the notebook."""
    print("\nSummary:")
    header = (
        f"{'N':>3} | {'lambda_min':>10} {'lambda_max':>10} | "
        f"{'circ_time':>10} {'circ_feas':>10} | "
        f"{'pop_time':>10} {'pop_feas':>10} | "
        f"{'circScale_time':>14} {'circScale_feas':>14} | "
        f"{'popScale_time':>13} {'popScale_feas':>13}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        lam_min = r['circ_scale']['lambda_min']
        lam_max = r['circ_scale']['lambda_max']
        print(
            f"{r['N']:>3} | {lam_min:>10.3f} {lam_max:>10.3f} | "
            f"{r['circ']['time']:>10.3f} {str(r['circ']['feasible']):>10} | "
            f"{r['pop']['time']:>10.3f} {str(r['pop']['feasible']):>10} | "
            f"{r['circ_scale']['time']:>14.3f} {str(r['circ_scale']['feasible']):>14} | "
            f"{r['pop_scale']['time']:>13.3f} {str(r['pop_scale']['feasible']):>13}"
        )


def main():
    """Main analysis execution function."""
    # Run the analysis from the notebook
    Nset = [4, 8, 12, 32, 64, 100]
    print("Running experiment for N =", Nset)
    results = run_analysis_experiment(Nset, Gtype=1, alpha=0.09, solver=cp.SCS)
    print_results_table(results)


if __name__ == "__main__":
    main()
