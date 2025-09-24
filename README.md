# Multi-Agent Lur'e Systems Simulations

Numerical simulations for the paper "A Scalable LMI Approach for Absolute Stability of Weakly Non-Linear Multi-Agent Systems" by Sharmin Akthar and Matthew C. Turner.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Analysis (Section A)

Run the LMI analysis comparison:

```bash
python run_analysis.py
```

This runs the computational efficiency comparison between brute-force and scalable LMI methods.

### Synthesis (Section B)

Run the controller synthesis:

```bash
python run_synthesis.py
```

This designs a state-feedback controller and simulates the closed-loop system.

## Project Structure

```
scalable_lmi|_lure/
├── NumericalSimulation.ipynb  # Runnable all-in-one notebook for analysis and synthesis
├── run_analysis.py           # Analysis entry point
├── run_synthesis.py          # Synthesis entry point
├── main/                     # Main execution scripts
│   ├── analysis_runner.py    # Analysis runner
│   └── synthesis_runner.py   # Synthesis runner
├── src/                      # Source code modules
│   ├── core/                 # Core utilities
│   │   ├── __init__.py
│   │   └── graph_utils.py    # Graph topology generation
│   └── algorithms/           # LMI algorithms
│       ├── __init__.py
│       ├── stability_solvers.py  # Analysis LMI solvers
│       └── synthesis_solvers.py  # Synthesis LMI solver
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Modules

- **`src/core/`** - Core utilities (graph generation)
- **`src/algorithms/`** - LMI-based algorithms (stability analysis and controller synthesis)
- **`main/`** - Main execution scripts and experiment runners
- **`run_*.py`** - Convenient entry point scripts
