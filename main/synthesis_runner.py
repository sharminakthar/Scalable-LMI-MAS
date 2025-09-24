"""
Synthesis Runner for Multi-Agent Lur'e Systems.

This module contains the main synthesis execution script that designs state-feedback
controllers and simulates the closed-loop system.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.graph_utils import build_laplacian
from algorithms.synthesis_solvers import lqr_lure_scale


def run_synthesis_example():
    """Run the synthesis example from the notebook."""
    # Setup for synthesis
    Gtype = 3  # ring
    N = 32     # Number of agents

    # Build Laplacian
    L = build_laplacian(N, Gtype)

    # System definition for a single agent
    P1 = {
        'a': np.array([[2, 1, 0], [1, -1, 1], [0, -2, 0]]),
        'b1': np.array([[0], [0], [3]]),
        'b2': np.array([[0], [0], [1]]),
        'c': np.array([[1, 0, 0]])
    }
    n, m = P1['b2'].shape

    # Synthesis algorithm
    alpha = 1.0
    Q = 1 * np.eye(n)
    R = 1

    print(f"Running synthesis for a {Gtype}-graph with {N} agents...")
    data = lqr_lure_scale(P1, L, alpha, Q, R)

    if 'error' in data:
        print(f"Synthesis failed: {data['error']}")
        return None
    else:
        # Print results from solver
        print("\n--- Synthesis Results ---")
        print(f"Solver status: {data['status']}")
        print(f"Minimized eta: {data['eta']:.4f}")
        
        K = data['K']
        print(f"Optimal Controller Gain (K):\n{K}")
        print("-------------------------\n")
        
        # Construct aggregate closed-loop system
        A_cl = np.kron(np.eye(N), P1['a'] + P1['b2'] @ K)
        
        # Create state-space model representing closed-loop system
        sys_closed_loop = ct.ss(A_cl, np.zeros((n * N, 1)), np.eye(n * N), 0)

        # Plot sample results
        x0 = np.random.rand(n * N)
        T = np.linspace(0, 15, 1000)  # Simulate for 15 seconds

        print("Simulating system response...")
        # Simulate the response of the system to the initial condition x0
        time, yout = ct.initial_response(sys_closed_loop, T=T, X0=x0)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time, yout.T)
        plt.xlabel('Time [sec]')
        plt.ylabel('States [units]')
        plt.title(f'State Trajectories for {N} Agents')
        plt.grid(True)
        plt.show()
        
        return data


def main():
    """Main synthesis execution function."""
    # Run the synthesis from the notebook
    run_synthesis_example()


if __name__ == "__main__":
    main()
