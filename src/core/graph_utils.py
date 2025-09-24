"""
Graph utilities for Multi-Agent Lur'e Systems.

This module contains functions for generating different types of communication
topologies commonly used in multi-agent systems research.
"""

import numpy as np


def build_laplacian(N: int, gtype: int) -> np.ndarray:
    """Build Laplacian matrix for different graph topologies.
    
    Args:
        N (int): Number of agents in the network
        gtype (int): Graph type: 1=line, 2=star, 3=ring, 4=complete
        
    Returns:
        np.ndarray: Laplacian matrix of size N x N
        
    Raises:
        ValueError: If gtype is not in {1, 2, 3, 4}
    """
    if gtype == 1:  # line
        A = np.zeros((N, N))
        for i in range(N-1):
            A[i, i+1] = 1
            A[i+1, i] = 1
    elif gtype == 2:  # star
        A = np.zeros((N, N))
        for j in range(1, N):
            A[0, j] = 1
            A[j, 0] = 1
    elif gtype == 3:  # ring
        A = np.zeros((N, N))
        for j in range(N):
            A[j, (j+1) % N] = 1
            A[(j+1) % N, j] = 1
    elif gtype == 4:  # complete
        A = np.ones((N, N)) - np.eye(N)
    else:
        raise ValueError("unknown gtype")
    
    deg = np.sum(A, axis=1)
    L = np.diag(deg) - A
    return L
