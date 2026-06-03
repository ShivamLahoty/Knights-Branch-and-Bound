import time
import csv
import json
import os
from datetime import datetime


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = None
        self._end = None

    def start(self):
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        self._end = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """Return elapsed seconds. Can be called before stop() for a live reading."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def timed(fn):
    """
    Decorator that prints the execution time of a function.

    Usage:
        @timed
        def my_function(...): ...
    """
    def wrapper(*args, **kwargs):
        t = Timer().start()
        result = fn(*args, **kwargs)
        print(f"[timer] {fn.__name__} took {t.stop():.4f}s")
        return result
    return wrapper

# Pretty printing

SEPARATOR = "=" * 60

def print_header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}")


def print_result(result: dict):
    """
    Print a standardized result dictionary in a readable format.
    Works with results from ILPSolver, BnBSolver, etc.
    """
    print_header(f"Result — {result.get('solver', 'unknown')} | n={result.get('n', '?')}")
    for key, value in result.items():
        if key == 'placement':
            print(f"  {'placement':<20}: {value}")
        else:
            print(f"  {key:<20}: {value}")
    print()


def print_comparison_table(results: list[dict]):
    """
    Print a comparison table from a list of result dicts.
    Each dict should have at least: n, solver, num_knights, solve_time, status.

    Args:
        results: list of result dicts
    """
    if not results:
        print("No results to display.")
        return

    print(f"\n{'n':>4} | {'solver':<20} | {'knights':>8} | {'time (s)':>10} | {'status':>12} | {'nodes':>8}")
    print(f"{'-'*4}-+-{'-'*20}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")

    for r in results:
        n           = r.get('n', '?')
        solver      = r.get('solver', 'unknown')
        knights     = str(r.get('num_knights', 'N/A'))
        t           = r.get('solve_time', 0.0)
        status      = r.get('status', 'unknown')
        nodes       = str(r.get('nodes_explored', 'N/A'))
        print(f"{n:>4} | {solver:<20} | {knights:>8} | {t:>10.4f} | {status:>12} | {nodes:>8}")

    print()

# Result standardization

def make_result(
    solver: str,
    n: int,
    status: str,
    num_knights: int = None,
    placement: list = None,
    solve_time: float = 0.0,
    nodes_explored: int = None,
    upper_bound: int = None,
    lower_bound: int = None,
    extra: dict = None,
) -> dict:
    """
    Build a standardized result dictionary.
    All solvers (ILP, BnB variants) should return this format
    so Shivam's experiment runner can process them uniformly.

    Args:
        solver         : name string e.g. 'ILP', 'BnB_BestFirst', 'BnB_DepthFirst'
        n              : board size
        status         : 'optimal' | 'infeasible' | 'timeout' | 'error'
        num_knights    : number of knights in best solution found
        placement      : list of square indices
        solve_time     : wall-clock seconds
        nodes_explored : how many B&B nodes were processed
        upper_bound    : best feasible solution value found
        lower_bound    : best lower bound at termination
        extra          : any additional solver-specific info

    Returns:
        dict
    """
    result = {
        'solver'         : solver,
        'n'              : n,
        'status'         : status,
        'num_knights'    : num_knights,
        'placement'      : placement or [],
        'solve_time'     : round(solve_time, 6),
        'nodes_explored' : nodes_explored,
        'upper_bound'    : upper_bound,
        'lower_bound'    : lower_bound,
        'timestamp'      : datetime.now().isoformat(),
    }
    if extra:
        result.update(extra)
    return result

# Saving results

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "results"
)

def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results_csv(results: list[dict], filename: str):
    """
    Save a list of result dicts to a CSV file in experiments/results/.

    Args:
        results  : list of result dicts (from make_result)
        filename : e.g. 'bnb_comparison.csv'
    """
    ensure_results_dir()
    filepath = os.path.join(RESULTS_DIR, filename)

    if not results:
        print(f"[utils] No results to save.")
        return

    fieldnames = list(results[0].keys())

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # placement is a list — convert to string for CSV
            row = r.copy()
            row['placement'] = str(row.get('placement', []))
            writer.writerow(row)

    print(f"[utils] Saved {len(results)} results to {filepath}")


def save_results_json(results: list[dict], filename: str):
    """
    Save a list of result dicts to a JSON file in experiments/results/.

    Args:
        results  : list of result dicts
        filename : e.g. 'bnb_comparison.json'
    """
    ensure_results_dir()
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[utils] Saved {len(results)} results to {filepath}")


def load_results_json(filename: str) -> list[dict]:
    """Load results from a JSON file in experiments/results/."""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'r') as f:
        return json.load(f)


# Validation helper

def verify_against_ilp(bnb_results: list[dict], ilp_results: dict) -> bool:
    """
    Cross-check BnB results against known ILP ground truth.

    Args:
        bnb_results : list of result dicts from BnB runs
        ilp_results : dict mapping n -> num_knights (ground truth)

    Returns:
        True if all match, False if any mismatch found
    """
    all_correct = True
    print("\nVerifying BnB results against ILP ground truth:")
    for r in bnb_results:
        n = r.get('n')
        bnb_val = r.get('num_knights')
        ilp_val = ilp_results.get(n)

        if ilp_val is None:
            print(f"  n={n}: no ILP ground truth available")
            continue

        match = (bnb_val == ilp_val)
        symbol = "✓" if match else "✗"
        print(f"  n={n}: BnB={bnb_val}, ILP={ilp_val}  {symbol}")
        if not match:
            all_correct = False

    return all_correct

# Smoke test

if __name__ == "__main__":
    print_header("utils.py smoke test")

    # Timer test
    with Timer() as t:
        time.sleep(0.1)
    print(f"Timer test: {t.elapsed():.3f}s (should be ~0.1s)")

    # make_result test
    r = make_result(
        solver='ILP',
        n=5,
        status='optimal',
        num_knights=10,
        placement=[0, 3, 7, 12],
        solve_time=0.042,
        nodes_explored=None,
    )
    print_result(r)

    # Table test
    results = [
        make_result('ILP',            5, 'optimal', 10, solve_time=0.01),
        make_result('BnB_BestFirst',  5, 'optimal', 10, solve_time=0.45, nodes_explored=123),
        make_result('BnB_DepthFirst', 5, 'optimal', 10, solve_time=0.62, nodes_explored=201),
        make_result('ILP',            6, 'optimal', 14, solve_time=0.03),
    ]
    print_comparison_table(results)

    print("All smoke tests passed ✓")