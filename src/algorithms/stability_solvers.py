"""
Stability Analysis Solvers for Multi-Agent Lur'e Systems.

This module contains LMI-based stability analysis methods including both
brute-force and scalable approaches using Circle and Popov criteria.
"""

import numpy as np
import cvxpy as cp
import time
from typing import Dict, Any


def solve_brute_force_circle(A, B, C, L, alpha, eps=1e-6, solver=cp.SCS, solver_opts=None) -> Dict[str, Any]:
    """Brute-force Circle LMI solver."""
    N = L.shape[0]
    n = A.shape[0]
    ny = C.shape[0]
    eigs = np.linalg.eigvals(L)
    lam_min = np.min(np.real(eigs))
    lam_max = np.max(np.real(eigs))

    Aagg = np.kron(np.eye(N), A)
    Bagg = np.kron(np.eye(N), B)
    Cagg = np.kron(L, C)

    P = cp.Variable((N*n, N*n), symmetric=True)
    W = cp.Variable((N*ny, N*ny), symmetric=True)

    top_left = Aagg.T @ P + P @ Aagg
    top_right = P @ Bagg + Cagg.T @ (alpha * W)
    bottom_right = -2.0 * W

    M = cp.bmat([[top_left, top_right],
                 [top_right.T, bottom_right]])

    solver_opts = solver_opts or {}
    constraints = [
        P >> eps * np.eye(N*n),
        W >> eps * np.eye(N*ny),
        -M >> eps * np.eye(N*n + N*ny)
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)

    t0 = time.time()
    prob.solve(solver=solver, verbose=False, **solver_opts)
    t_elapsed = time.time() - t0

    feasible = prob.status in ('optimal', 'optimal_inaccurate')
    return {"feasible": feasible, 
            "status": prob.status, 
            "time": t_elapsed, 
            "P": P.value if feasible else None, 
            "W": W.value if feasible else None,
            "lambda_min": lam_min,
            "lambda_max": lam_max
    }


def solve_brute_force_popov(A, B, C, L, alpha, eps=1e-6, solver=cp.SCS, solver_opts=None) -> Dict[str, Any]:
    """Brute-force Popov LMI solver."""
    N = L.shape[0]
    n = A.shape[0]
    ny = C.shape[0]

    eigs = np.linalg.eigvals(L)
    lam_min = np.min(np.real(eigs))
    lam_max = np.max(np.real(eigs))

    Aagg = np.kron(np.eye(N), A)
    Bagg = np.kron(np.eye(N), B)
    Cagg = np.kron(L, C)

    P = cp.Variable((N*n, N*n), symmetric=True)
    W = cp.Variable((N*ny, N*ny), symmetric=True)
    Npop = cp.Variable((N*ny, N*ny), symmetric=True)

    top_left = Aagg.T @ P + P @ Aagg
    AC_term = (Aagg.T @ Cagg.T) @ Npop
    top_right = P @ Bagg + Cagg.T @ (alpha * W) + AC_term
    bottom_right = -2.0 * W + Npop @ (Cagg @ Bagg) + (Bagg.T @ Cagg.T) @ Npop

    M = cp.bmat([[top_left, top_right],
                 [top_right.T, bottom_right]])

    solver_opts = solver_opts or {}
    constraints = [
        P >> eps * np.eye(N*n),
        W >> eps * np.eye(N*ny),
        Npop >> eps * np.eye(N*ny),
        -M >> eps * np.eye(N*n + N*ny)
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)

    t0 = time.time()
    prob.solve(solver=solver, verbose=False, **solver_opts)
    t_elapsed = time.time() - t0

    feasible = prob.status in ('optimal', 'optimal_inaccurate')
    return {"feasible": feasible, 
            "status": prob.status, 
            "time": t_elapsed, 
            "P": P.value if feasible else None, 
            "W": W.value if feasible else None, 
            "Npop": Npop.value if feasible else None,
            "lambda_min": lam_min,
            "lambda_max": lam_max
    }


def solve_scalable_circle(A, B, C, L, alpha, eps=1e-6, solver=cp.SCS, solver_opts=None) -> Dict[str, Any]:
    """Scalable Circle LMI solver."""
    eigs = np.linalg.eigvals(L)
    lam_min = np.min(np.real(eigs))
    lam_max = np.max(np.real(eigs))
    n = A.shape[0]
    ny = C.shape[0]

    P = cp.Variable((n, n), symmetric=True)
    W1 = cp.Variable((ny, ny), symmetric=True)
    W2 = cp.Variable((ny, ny), symmetric=True)

    TL1 = A.T @ P + P @ A
    TR1 = P @ B + lam_max * alpha * (C.T @ W1)
    BR1 = -2.0 * W1
    M1 = cp.bmat([[TL1, TR1], [TR1.T, BR1]])

    TL2 = A.T @ P + P @ A
    TR2 = P @ B + lam_min * alpha * (C.T @ W2)
    BR2 = -2.0 * W2
    M2 = cp.bmat([[TL2, TR2], [TR2.T, BR2]])

    solver_opts = solver_opts or {}
    constraints = [
        P >> eps * np.eye(n),
        W1 >> eps * np.eye(ny),
        W2 >> eps * np.eye(ny),
        -M1 >> eps * np.eye(n + ny),
        -M2 >> eps * np.eye(n + ny)
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    t0 = time.time()
    prob.solve(solver=solver, verbose=False, **solver_opts)
    t_elapsed = time.time() - t0
    feasible = prob.status in ('optimal', 'optimal_inaccurate')
    return {"feasible": feasible, 
            "status": prob.status, 
            "time": t_elapsed, 
            "P": P.value if feasible else None, 
            "W1": W1.value if feasible else None, 
            "W2": W2.value if feasible else None,
            "lambda_min": lam_min,
            "lambda_max": lam_max
    }


def solve_scalable_popov(A, B, C, L, alpha, eps=1e-6, solver=cp.SCS, solver_opts=None) -> Dict[str, Any]:
    """Scalable Popov LMI solver."""
    eigs = np.linalg.eigvals(L)
    lam_min = np.min(np.real(eigs))
    lam_max = np.max(np.real(eigs))
    n = A.shape[0]
    ny = C.shape[0]

    P = cp.Variable((n, n), symmetric=True)
    W1 = cp.Variable((ny, ny), symmetric=True)
    W2 = cp.Variable((ny, ny), symmetric=True)
    eta = cp.Variable(nonneg=True)

    def build_M(lam, W):
        TL = A.T @ P + P @ A
        TR = P @ B + lam * (alpha * C.T @ W + eta * (A.T @ C.T))
        BR = -2.0 * W + 2.0 * eta * lam * (C @ B + (B.T @ C.T))
        return cp.bmat([[TL, TR], [TR.T, BR]])

    M1 = build_M(lam_min, W1)
    M2 = build_M(lam_max, W2)

    solver_opts = solver_opts or {}
    constraints = [
        P >> eps * np.eye(n),
        W1 >> eps * np.eye(ny),
        W2 >> eps * np.eye(ny),
        eta >= 0,
        -M1 >> eps * np.eye(n + ny),
        -M2 >> eps * np.eye(n + ny)
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    t0 = time.time()
    prob.solve(solver=solver, verbose=False, **solver_opts)
    t_elapsed = time.time() - t0
    feasible = prob.status in ('optimal', 'optimal_inaccurate')
    return {"feasible": feasible, 
            "status": prob.status,
            "time": t_elapsed, 
            "P": P.value if feasible else None, 
            "W1": W1.value if feasible else None, 
            "W2": W2.value if feasible else None, 
            "eta": eta.value if feasible else None,
            "lambda_min": lam_min,
            "lambda_max": lam_max
    }
