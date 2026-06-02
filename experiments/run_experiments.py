import os
import sys
# import random
# import time
import pandas as pd
import matplotlib.pyplot as plt
# from bnb import BnBSolver


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Point to the 'src' directory
src_dir = os.path.join(root_dir, 'src')

# Add 'src' to Python's system path so it can find utils.py and bnb.py
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
from bnb import BnBSolver


from utils import make_result, save_results_csv, print_comparison_table, print_header
import matplotlib.pyplot as plt
import pandas as pd
import time
import random


# Now we can safely import from utils


# def mock_bnb_solve(n: int, strategy: str, branch_var: str) -> dict:
#     """
#     A dummy solver to test the experiment pipeline without Gurobi or bnb.py.
#     This generates fake data in the exact format utils.py expects.
#     """
#     time.sleep(random.uniform(0.01, 0.05))  # Simulate compute time
    
#     # Fake a placement array
#     fake_placement = random.sample(range(n*n), k=n)
    
#     return make_result(
#         solver=f"BnB_{strategy}",
#         n=n,
#         status='optimal',
#         num_knights=len(fake_placement),
#         placement=fake_placement,
#         solve_time=random.uniform(0.1, 5.0),
#         nodes_explored=random.randint(10, 1000),
#         extra={'branch_var': branch_var}
#     )


def visualize_results(csv_path):
    try:
        df = pd.read_csv(csv_path)

        strategy_df = df[
            df['branch_var'] == 'most_constrained'
        ]

        branch_df = df[
            df['strategy'] == 'best_first'
        ]

    except FileNotFoundError:
        print("[ERROR] CSV not found.")
        return

    print("\nGenerating combined visualization dashboard...")

    graphs_dir = os.path.join(
        root_dir,
        'experiments',
        'results'
    )



    fig, axes = plt.subplots(
        3,
        2,
        figsize=(18, 12)
    )

    fig.suptitle(
        'Branch and Bound Strategy Evaluation',
        fontsize=18,
        fontweight='bold'
    )



    for solver in strategy_df['solver'].unique():

        subset = strategy_df[
            strategy_df['solver'] == solver
        ]

        avg_time = subset.groupby('n')[
            'solve_time'
        ].mean()

        axes[0, 0].plot(
            avg_time.index,
            avg_time.values,
            marker='o',
            linewidth=2,
            label=solver
        )

    axes[0, 0].set_title(
        'Strategy Comparison — Solve Time'
    )

    axes[0, 0].set_xlabel('Board Size (n)')
    axes[0, 0].set_ylabel('Solve Time (seconds)')

    axes[0, 0].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[0, 0].legend(fontsize=8)



    for solver in strategy_df['solver'].unique():

        subset = strategy_df[
            strategy_df['solver'] == solver
        ]

        avg_nodes = subset.groupby('n')[
            'nodes_explored'
        ].mean()

        axes[0, 1].plot(
            avg_nodes.index,
            avg_nodes.values,
            marker='s',
            linewidth=2,
            label=solver
        )

    axes[0, 1].set_title(
        'Strategy Comparison — Nodes Explored'
    )

    axes[0, 1].set_xlabel('Board Size (n)')
    axes[0, 1].set_ylabel('Nodes Explored')

    axes[0, 1].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[0, 1].legend(fontsize=8)



    for solver in branch_df['solver'].unique():

        subset = branch_df[
            branch_df['solver'] == solver
        ]

        avg_time = subset.groupby('n')[
            'solve_time'
        ].mean()

        axes[1, 0].plot(
            avg_time.index,
            avg_time.values,
            marker='o',
            linewidth=2,
            label=solver
        )

    axes[1, 0].set_title(
        'Branch Variable Comparison — Solve Time'
    )

    axes[1, 0].set_xlabel('Board Size (n)')
    axes[1, 0].set_ylabel('Solve Time (seconds)')

    axes[1, 0].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[1, 0].legend(fontsize=8)


    for solver in branch_df['solver'].unique():

        subset = branch_df[
            branch_df['solver'] == solver
        ]

        avg_nodes = subset.groupby('n')[
            'nodes_explored'
        ].mean()

        axes[1, 1].plot(
            avg_nodes.index,
            avg_nodes.values,
            marker='d',
            linewidth=2,
            label=solver
        )

    axes[1, 1].set_title(
        'Branch Variable Comparison — Nodes'
    )

    axes[1, 1].set_xlabel('Board Size (n)')
    axes[1, 1].set_ylabel('Nodes Explored')

    axes[1, 1].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[1, 1].legend(fontsize=8)


    for solver in df['solver'].unique():

        subset = df[
            df['solver'] == solver
        ]

        axes[2, 0].scatter(
            subset['nodes_explored'],
            subset['solve_time'],
            label=solver,
            alpha=0.7
        )

    axes[2, 0].set_title(
        'Runtime vs Nodes Explored'
    )

    axes[2, 0].set_xlabel('Nodes Explored')
    axes[2, 0].set_ylabel('Solve Time (seconds)')

    axes[2, 0].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[2, 0].legend(fontsize=7)


    lower_bounds = df.groupby('n')[
        'lower_bound'
    ].mean()

    optimal_vals = df.groupby('n')[
        'num_knights'
    ].mean()

    axes[2, 1].plot(
        lower_bounds.index,
        lower_bounds.values,
        marker='o',
        linewidth=2,
        label='LP Lower Bound'
    )

    axes[2, 1].plot(
        optimal_vals.index,
        optimal_vals.values,
        marker='s',
        linewidth=2,
        label='Optimal Solution'
    )

    axes[2, 1].set_title(
        'LP Relaxation Quality'
    )

    axes[2, 1].set_xlabel('Board Size (n)')
    axes[2, 1].set_ylabel('Knights')

    axes[2, 1].grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    axes[2, 1].legend(fontsize=8)



    plt.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    plt.savefig(
        os.path.join(
            graphs_dir,
            'bnb_complete_dashboard.png'
        ),
        dpi=300,
        bbox_inches='tight'
    )

    print("\nDashboard saved successfully!")

    plt.show()


def main():

    print_header("Running B&B Experiments")

    sizes = [4, 5, 6, 7]
    strategies = ['best_first', 'depth_first', 'breadth_first']
    branch_vars = ['most_constrained', 'first_fractional', 'least_constrained']

    results = []
    NUM_RUNS = 3

    for n in sizes:
        for strat in strategies:
            for b_var in branch_vars:
                for run_id in range(NUM_RUNS):

                    print(
                        f"\nRunning: n={n}, "
                        f"strategy={strat}, "
                        f"branch_var={b_var}, "
                        f"run={run_id+1}"
                    )

                    solver = BnBSolver(
                        n=n,
                        strategy=strat,
                        branch_var=b_var,
                        verbose=False,
                        time_limit=60
                    )

                    res = solver.solve()

                    res["run_id"] = run_id + 1

                    results.append(res)

    print_comparison_table(results)

    csv_filename = "bnb_strategies_comparison.csv"

    save_results_csv(results, csv_filename)

    results_path = os.path.join(
        root_dir,
        'experiments',
        'results',
        csv_filename
    )

    visualize_results(results_path)

if __name__ == "__main__":
    main()