"""
Controller Synthesis Solvers for Multi-Agent Lur'e Systems.

This module contains LQR-based controller synthesis algorithms for multi-agent systems.
"""

import numpy as np
import cvxpy as cp
import time as systime
from typing import Dict, Any


def lqr_lure_scale(syst, L_laplacian, alpha, Q, R):
    """
    LQR-Lur'e scalable synthesis solver.
    
    Parameters:
    syst (dict): System matrices 'a', 'b1', 'b2', 'c'
    L_laplacian (np.ndarray): Laplacian matrix
    alpha (float): Sector bound parameter
    Q (np.ndarray): State weighting matrix
    R (np.ndarray): Input weighting matrix
    
    Returns:
    dict: Synthesis results
    """
    # Parameters
    A = syst['a']
    B1 = syst['b1']
    B2 = syst['b2']
    C = syst['c']

    # n = state dimension, m = input dimension
    n, m = B2.shape  
    
    # Ensure R is a matrix for inversion
    if isinstance(R, (int, float)):
        R = np.array([[R]])

    # Eigenvalues of the Laplacian
    Lambda = np.linalg.eigvals(L_laplacian)
    lam_min = np.min(Lambda)
    lam_max = np.max(Lambda)

    tstart = systime.time()
    
    # LMI variable definitions using CVXPY
    Y = cp.Variable((n, n), symmetric=True)
    U1 = cp.Variable()
    U2 = cp.Variable()
    L = cp.Variable((m, n))
    eta = cp.Variable()  # Scalar to be minimized

    # Prepare constant matrices
    R_inv = np.linalg.inv(R)
    Q_inv = np.linalg.inv(Q)

    # LMI Constraints
    constraints = []

    # First LMI (with lam_min)
    M1 = cp.bmat([
        [A @ Y + Y @ A.T + B2 @ L + L.T @ B2.T, B1 * U1 + alpha * lam_min * Y @ C.T, L.T, Y],
        [(B1 * U1 + alpha * lam_min * Y @ C.T).T, -2 * U1 * np.eye(m), np.zeros((m, m)), np.zeros((m, n))],
        [L, np.zeros((m, m)), -R_inv, np.zeros((m, n))],
        [Y, np.zeros((n, m)), np.zeros((n, m)), -Q_inv]
    ])
    constraints.append(M1 << 0)

    # Second LMI (with lam_max)
    M2 = cp.bmat([
        [A @ Y + Y @ A.T + B2 @ L + L.T @ B2.T, B1 * U2 + alpha * lam_max * Y @ C.T, L.T, Y],
        [(B1 * U2 + alpha * lam_max * Y @ C.T).T, -2 * U2 * np.eye(m), np.zeros((m, m)), np.zeros((m, n))],
        [L, np.zeros((m, m)), -R_inv, np.zeros((m, n))],
        [Y, np.zeros((n, m)), np.zeros((n, m)), -Q_inv]
    ])
    constraints.append(M2 << 0)
    
    # Third LMI: Y > 0
    constraints.append(Y >> 1e-9 * np.eye(n))

    # Fourth LMI: Minimize eta s.t. [ -eta*I, I; I, -Y ] <= 0
    M4 = cp.bmat([
        [-eta * np.eye(n), np.eye(n)],
        [np.eye(n), -Y]
    ])
    constraints.append(M4 << 0)

    # Solve the problem
    objective = cp.Minimize(eta)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False) 
    
    telapsed = systime.time() - tstart
    print(f"LMI solved in {telapsed:.4f} seconds. Status: {problem.status}")

    # Return solution
    if problem.status in ["optimal", "optimal_inaccurate"]:
        Y_val = Y.value
        L_val = L.value
        
        K_val = L_val @ np.linalg.inv(Y_val)

        data = {
            'Y': Y_val,
            'U1': U1.value,
            'U2': U2.value,
            'L': L_val,
            'K': K_val,
            'eta': eta.value,
            'status': problem.status
        }
        return data
    else:
        return {'status': problem.status, 'error': 'Solver failed or the problem is infeasible.'}
