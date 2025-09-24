"""
Algorithms for Multi-Agent Lur'e Systems.

This module contains LMI-based stability analysis and controller synthesis algorithms.
"""

from .stability_solvers import (
    solve_brute_force_circle,
    solve_brute_force_popov,
    solve_scalable_circle,
    solve_scalable_popov
)
from .synthesis_solvers import lqr_lure_scale

__all__ = [
    'solve_brute_force_circle',
    'solve_brute_force_popov', 
    'solve_scalable_circle',
    'solve_scalable_popov',
    'lqr_lure_scale'
]
