
import os
import sys

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from utils import print_header, print_result, print_comparison_table, make_result, Timer
from board import Board
import argparse



def run_ilp(n: int, verbose: bool = False):
    try:
        from ilp_solver import ILPSolver
    except ImportError as e:
        print(f"[ERROR] Could not import ILPSolver: {e}")
        return

    solver = ILPSolver(n)
    solver.solve_and_display()

# Mode: BnB solve

def run_bnb(n, strategy, branch_var, branch_order="zero_first", verbose=False):
    try:
        from bnb import BnBSolver
    except ImportError:
        print("[ERROR] bnb.py not found yet!")
        print("        Run with --mode ilp for now.")
        return

    print_header(
        f"Branch and Bound — {n}x{n} | strategy={strategy} | branch_var={branch_var}")

    solver = BnBSolver(
        n=n,
        strategy=strategy,
        branch_var=branch_var,
        branch_order=branch_order,
        verbose=verbose,
    )

    with Timer() as t:
        result = solver.solve()

    result['solve_time'] = t.elapsed()
    print_result(result)

    board = Board(n)
    board.display(result.get('placement', []))


def run_benchmark(sizes: list[int]):
    try:
        from ilp_solver import run_benchmark as ilp_benchmark
    except ImportError as e:
        print(f"[ERROR] Could not import ilp_solver: {e}")
        return

    ilp_benchmark(sizes=sizes)

# Mode: Verify BnB against ILP

def run_verify(n: int):
    try:
        from ilp_solver import ILPSolver
        from bnb import BnBSolver
        from utils import verify_against_ilp
    except ImportError as e:
        print(f"[ERROR] {e}")
        return

    print_header(f"Verification — n={n}")

    # Ground truth from ILP
    ilp_solver = ILPSolver(n)
    ilp_result = ilp_solver.solve()

    if ilp_result['status'] != 'optimal':
        print(f"ILP could not solve n={n}: {ilp_result['status']}")
        return

    print(f"ILP ground truth: {ilp_result['num_knights']} knights")

    # BnB result (default strategy)
    bnb_solver = BnBSolver(n=n)
    bnb_result = bnb_solver.solve()

    # Compare
    ilp_ground_truth = {n: ilp_result['num_knights']}
    verify_against_ilp([bnb_result], ilp_ground_truth)

# Mode: Full experiment suite

def run_experiments():
    try:
        from experiments.run_experiments import main as exp_main
    except ImportError:
        print(
            "[ERROR] experiments/run_experiments.py not found yet!")
        return

    exp_main()

# Mode: Display board only

def run_display(n: int):
    board = Board(n)
    print_header(f"Empty {n}x{n} Board")
    board.display()
    print(f"Total squares  : {board.num_squares}")
    center = board.square_index(n // 2, n // 2)
    attackers = board.attackers(center)
    print(f"Center square  : {center} at {board.square_coords(center)}")
    print(f"Center attacked by {len(attackers)} squares: {attackers}")

# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knights on the Chessboard 2 — Branch and Bound",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        '--mode',
        choices=['ilp', 'lp', 'bnb', 'benchmark',
                 'verify', 'experiment', 'display'],
        default='ilp',
        help=(
            "ilp        : solve with ILP (ground truth)\n"
            "bnb        : solve with Branch and Bound\n"
            "benchmark  : run ILP across multiple sizes\n"
            "verify     : check BnB matches ILP\n"
            "experiment : run full experiment suite\n"
            "display    : just show the board\n"
        )
    )

    parser.add_argument('--n', type=int, default=5,
                        help='Board size n (default: 5)')

    parser.add_argument('--sizes', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8],
                        help='Board sizes for benchmark mode (default: 3 4 5 6 7 8)')

    parser.add_argument('--strategy', type=str, default='best_first',
                        choices=['best_first', 'depth_first', 'breadth_first'],
                        help='Node selection strategy for BnB (default: best_first)')

    parser.add_argument('--branch_var', type=str, default='most_constrained',
                        choices=['most_constrained', 'least_constrained',
                                 'first_fractional', 'most_coverage'],
                        help='Variable selection rule for branching (default: most_constrained)')

    parser.add_argument('--branch_order', type=str, default='zero_first',
                        choices=['zero_first', 'one_first',
                                 'lp_guided', 'three_way'],
                        help='Branch value order (default: zero_first)')

    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose solver output')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'ilp':
        run_ilp(args.n, args.verbose)

    elif args.mode == 'lp':
        from ilp_solver import ILPSolver

        solver = ILPSolver(args.n)

        result = solver.solve(
            verbose=args.verbose,
            relax=True
        )

        print("\nLP Relaxation Result")
        print("=" * 50)

        print(f"Objective Value : {result['obj_value']:.4f}")
        print(f"Solve Time      : {result['solve_time']:.4f}s")

        print("\nFractional Variables:")

        for i, val in enumerate(result['x_values']):
            if val > 1e-6:
                print(f"x[{i}] = {val:.4f}")

    elif args.mode == 'bnb':
        run_bnb(args.n, args.strategy, args.branch_var, args.branch_order, args.verbose)

    elif args.mode == 'benchmark':
        run_benchmark(args.sizes)

    elif args.mode == 'verify':
        run_verify(args.n)

    elif args.mode == 'experiment':
        run_experiments()

    elif args.mode == 'display':
        run_display(args.n)


if __name__ == "__main__":
    main()
