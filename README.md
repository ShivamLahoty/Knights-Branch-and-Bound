<div align="center">

<img src="https://img.shields.io/badge/♞-Branch%20%26%20Bound-000000?style=for-the-badge&logoColor=white" alt="Knights B&B Banner"/>

# ♞ Knights on the Chessboard
### Branch & Bound Solver — Minimum Domination Cover

*A from-scratch Branch & Bound algorithm that finds the minimum number of knights needed to occupy or threaten every square of an n × n chessboard — with full strategy comparison and interactive visualisation.*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![SciPy](https://img.shields.io/badge/SciPy-LP_Relaxation-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Results_Analysis-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualisation-11557C?style=flat-square)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

## 📌 What is this?

This project solves a combinatorial optimisation problem on a chessboard:

> **Find the minimum set of knights such that every square is either occupied *and* threatened by another knight, or threatened by at least one knight.**

The twist: an occupied square is **not** automatically covered — it must *also* be attacked by a different knight. This makes it a strict variant of the [dominating set problem](https://en.wikipedia.org/wiki/Dominating_set) on a knight-move graph.

The solver is a **hand-written Branch & Bound algorithm** built on top of LP relaxation (SciPy). No commercial solver (Gurobi, CPLEX) is used for integer decisions — only as a free LP oracle for bounding.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ♟️ **Pure B&B from scratch** | Full Branch & Bound implementation — no ILP solver black-box |
| 📐 **Tight lower bound** | Graph-theoretic bound `⌈n²/8⌉` beats the LP root relaxation and prunes aggressively |
| 🚀 **Greedy warm start** | Heuristic initial solution slashes the number of nodes explored |
| 🔀 **3 traversal strategies** | Best-first, depth-first, breadth-first — all benchmarked head-to-head |
| 🎯 **2 branching strategies** | Most-constrained variable vs first-fractional — statistically compared |
| 📊 **Interactive results window** | matplotlib charts open automatically after experiments run |
| 📁 **CSV + structured export** | Every run logged with solver, board size, nodes explored, solve time, placement |
| 🧪 **Mock solver for testing** | Full pipeline testable without any commercial licence |

---

## 🏗️ Architecture

```
n × n Chessboard
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                    main.py  (CLI)                    │
│          --mode solve / --mode experiment            │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌─────────────────────────┐
│   src/board.py   │     │  experiments/            │
│  Knight-move     │     │  run_experiments.py      │
│  graph builder   │     │  All strategy combos     │
└────────┬─────────┘     └──────────┬──────────────┘
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────────────┐
│               src/ilp_solver.py                      │
│   LP Relaxation  ←  SciPy linprog (free, no licence) │
│   Returns fractional lower bound at each B&B node    │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│            Branch & Bound Core  (bnb.py)             │
│                                                      │
│  ┌─────────────────┐   ┌──────────────────────────┐  │
│  │  Lower Bounds   │   │   Branching Strategies   │  │
│  │  • LP relax     │   │   • most_constrained     │  │
│  │  • ⌈n²/8⌉ bound │   │   • first_fractional     │  │
│  └─────────────────┘   └──────────────────────────┘  │
│  ┌─────────────────────────────────────────────────┐  │
│  │          Node Selection Strategies              │  │
│  │   best_first │ depth_first │ breadth_first      │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────┐
│                  src/utils.py                        │
│   save_results_csv()  ·  print_comparison_table()    │
└───────────────────────────┬──────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
  bnb_strategies_comparison.csv    Interactive matplotlib
  (structured results log)         charts window
```

---

## 🧠 How the Algorithm Works

### 1 · ILP Formulation

Each cell `(i, j)` gets a binary decision variable `x[i][j] ∈ {0, 1}`.

```
Minimise:   Σ x[i][j]   for all cells

Subject to: Σ x[k] ≥ 1  for every cell c
            where k ranges over all cells that attack c via knight moves

            x[i][j] ∈ {0, 1}
```

The constraint forces every cell to be **threatened by at least one knight** — simply placing a knight on a cell is *not* sufficient to cover it.

### 2 · Lower Bound (tighter than LP root)

The LP relaxation bound is computed at every node. But a global lower bound better than the first LP value is derived analytically:

```
Every knight attacks at most 8 squares
→  you need at least ⌈n² / 8⌉ knights to cover all n² squares
```

This bound is precomputed once at the start and applied throughout the tree for aggressive early pruning.

### 3 · Greedy Heuristic Warm Start

Before the B&B tree is explored, a greedy heuristic produces an initial **upper bound**:

> Repeatedly place a knight on the unoccupied cell that attacks the *largest number* of currently uncovered squares.

This warm start tightens the upper bound from the very first node, dramatically reducing the subtrees that need to be explored.

### 4 · Branching & Node Selection

```
Branching variable strategies:
  most_constrained  →  pick the fractional variable that covers
                        the most remaining unconstrained cells
  first_fractional  →  pick the first fractional LP variable
                        (simpler, cheaper per node)

Node selection strategies:
  best_first    →  priority queue on LP lower bound (fewest nodes)
  depth_first   →  LIFO stack (low memory, fast to feasibility)
  breadth_first →  FIFO queue (finds shallow optima first)
```

All six combinations are benchmarked and statistically compared.

---

## 📁 Project Structure

```text
knights-branch-and-bound/
│
├── main.py                      # CLI entry point  (--mode solve / experiment)
├── requirements.txt             # pip dependencies
├── README.md                    # Project documentation
├── .gitignore
│
├── src/                         # Core solver modules
│   ├── __pycache__/
│   ├── bnb.py                   # Branch and Bound core logic
│   ├── board.py                 # Board representation & knight-move graph
│   ├── heuristics.py            # Heuristic functions for warm start
│   ├── ilp_solver.py            # LP relaxation wrapper
│   ├── lower_bounds.py          # Bounding calculations
│   └── utils.py                 # Results helpers & formatting
│
├── experiments/                 # Experiment harness
│   ├── __pycache__/
│   ├── results/                 # Output directory for benchmarks and visualizations
│   │   ├── bnb_complete_dashboard.png
│   │   ├── bnb_strategies_comparison.csv
│   │   ├── bnb_strategy_comparison.jpg
│   │   ├── lp_heatmap.png
│   │   └── optimal_board.png
│   ├── board_visualization.py   # Scripts for rendering board states
│   └── run_experiments.py       # Benchmarks all strategy combos
│
├── tests/                       # Unit tests
│   └── test_board.py            # Tests for board structure validation
│
└── report/                      # Written analysis & findings
    └── Report_Knights_Branch_and_Bound.pdf

## 🚀 Getting Started

**Prerequisites:** Python 3.8+ · No commercial solver licence required

---

```bash
# 1. Clone
git clone https://github.com/swayamvirmani/knights-branch-and-bound.git
cd knights-branch-and-bound

# 2. (Optional but recommended) Virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Solve a single board

```bash
python main.py --mode solve --n 5 --strategy best_first --branch_var most_constrained
```

| Flag | Options | Default |
|---|---|---|
| `--mode` | `solve` · `experiment` | `solve` |
| `--n` | any integer board size | `5` |
| `--strategy` | `best_first` · `depth_first` · `breadth_first` | `best_first` |
| `--branch_var` | `most_constrained` · `first_fractional` | `most_constrained` |

### Run full strategy comparison

```bash
python experiments/run_experiments.py
# or via the CLI:
python main.py --mode experiment
```

This benchmarks **all 6 strategy combinations** across board sizes, then:
- Prints a formatted comparison table to the terminal
- Saves `experiments/results/bnb_strategies_comparison.csv`
- Opens an **interactive matplotlib window** with:
  - 📈 Solve Time vs Board Size
  - 🔢 Nodes Explored vs Board Size

---

## 📊 Sample Results

Results from an actual run (n = 4, LP + greedy warm start):

| Solver | n | Status | Knights | Solve Time | Nodes Explored |
|---|---|---|---|---|---|
| `BnB_best_first` | 4 | ✅ optimal | 4 | 1.09 s | **58** |
| `BnB_breadth_first` | 4 | ✅ optimal | 4 | 3.68 s | 518 |
| `BnB_depth_first` | 4 | ✅ optimal | 4 | 4.18 s | 654 |
| `BnB_best_first` | 5 | ✅ optimal | 5 | 0.73 s | **38** |
| `BnB_depth_first` | 5 | ✅ optimal | 5 | 3.56 s | 698 |

> **Best-first search consistently finds the optimal solution with the fewest nodes explored** — the tighter lower bound from `⌈n²/8⌉` allows it to prune large subtrees early.

Each row in the CSV includes: `solver`, `n`, `status`, `num_knights`, `placement`, `solve_time`, `nodes_explored`, `upper_bound`, `lower_bound`, `branch_var`, `timestamp`.

---

## 🧪 Testing Without Gurobi

A **mock solver** is bundled for pipeline validation when no commercial solver is available:

```bash
python experiments/mock_bnb_solver.py
```

It generates realistic fake results in exactly the format `utils.py` expects, so the full pipeline — CSV export, interactive charts, comparison table — can be verified on any machine.

> The real solver uses **SciPy `linprog`** as its LP oracle, which is free and open-source. Gurobi is **not required**.

---

## 🔬 Concepts Demonstrated

```
✔  Branch & Bound algorithm built from first principles
✔  LP relaxation as a bounding function (SciPy linprog)
✔  Graph-theoretic lower bound tighter than LP root relaxation
✔  Greedy heuristic warm start for upper bound initialisation
✔  3 node selection strategies with empirical benchmarking
✔  2 branching variable selection strategies
✔  Statistical strategy comparison across board sizes
✔  Clean modular architecture: solver core / experiment harness / CLI
✔  Interactive matplotlib visualisation pipeline
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `scipy` | LP relaxation via `linprog` |
| `numpy` | Matrix ops & board representation |
| `pandas` | Results aggregation, CSV I/O |
| `matplotlib` | Interactive comparison charts |

```bash
pip install -r requirements.txt
```

No commercial solver. No cloud. Runs fully offline.

---

<div align="center">

*If this project helped you understand Branch & Bound, consider leaving a ⭐*

</div>