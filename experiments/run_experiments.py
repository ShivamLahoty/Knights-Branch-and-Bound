import os
import sys
# --- PATH ADJUSTMENT ---
# Find the project root (one folder up from 'experiments')
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Point to the 'src' directory
src_dir = os.path.join(root_dir, 'src')

# Add 'src' to Python's system path so it can find utils.py and bnb.py
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import make_result, save_results_csv, print_comparison_table, print_header
import matplotlib.pyplot as plt
import pandas as pd
import time
import random


# Now we can safely import from utils


def mock_bnb_solve(n: int, strategy: str, branch_var: str) -> dict:
    """
    A dummy solver to test the experiment pipeline without Gurobi or bnb.py.
    This generates fake data in the exact format utils.py expects.
    """
    time.sleep(random.uniform(0.01, 0.05))  # Simulate compute time

    # Fake a placement array
    fake_placement = random.sample(range(n*n), k=n)

    return make_result(
        solver=f"BnB_{strategy}",
        n=n,
        status='optimal',
        num_knights=len(fake_placement),
        placement=fake_placement,
        solve_time=random.uniform(0.1, 5.0),
        nodes_explored=random.randint(10, 1000),
        extra={'branch_var': branch_var}
    )


def visualize_results(csv_path):
    """Reads the CSV, saves charts as JPG, then opens interactive window."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[ERROR] CSV not found. Run experiments first.")
        return

    df = df[df["status"] != "timeout"]
    df = df[(df["solve_time"] > 0) & (df["nodes_explored"] > 0)]
    print("\nGenerating charts...")

    # --- Color per node-selection strategy, line style per branching variable ---
    color_map = {
        'best_first':   'blue',
        'depth_first':  'red',
        'breadth_first': 'green',
    }
    # Only solid and two dash styles — no dotted
    style_map = {
        'most_constrained':  ('-',  'o'),
        'first_fractional':  ('--', 's'),
        'least_constrained': ('-.', '^'),
        'most_coverage':     ('--', 'D'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Branch & Bound Strategy Comparison',
                 fontsize=14, fontweight='bold')

    # Enable scroll-to-zoom and click-drag-to-pan on both axes
    def on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        scale = 0.85 if event.button == 'up' else 1.15
        xdata, ydata = event.xdata, event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([xdata - (xdata - xlim[0]) * scale,
                     xdata + (xlim[1] - xdata) * scale])
        ax.set_ylim([ydata - (ydata - ylim[0]) * scale,
                     ydata + (ylim[1] - ydata) * scale])
        fig.canvas.draw_idle()

    pan_state = {'active': False, 'x': None, 'y': None, 'ax': None}

    def on_press(event):
        if event.inaxes is None:
            return
        pan_state['active'] = True
        pan_state['x'] = event.xdata
        pan_state['y'] = event.ydata
        pan_state['ax'] = event.inaxes

    def on_release(event):
        pan_state['active'] = False

    def on_motion(event):
        if not pan_state['active'] or event.inaxes != pan_state['ax']:
            return
        if event.xdata is None or event.ydata is None:
            return
        ax = pan_state['ax']
        dx = pan_state['x'] - event.xdata
        dy = pan_state['y'] - event.ydata
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event',         on_scroll)
    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)

    # --- Plot both charts ---
    nodes_lines = {}   # track unique lines for the nodes chart legend

    for solver in df['solver'].unique():
        subset = df[df['solver'] == solver]

        # Parse solver name — format is "strategy_branchvar" e.g. best_first_most_constrained
        parts = solver.split('_')
        # strategy is first two words joined, branch_var is the rest
        # best_first / depth_first / breadth_first
        strategy = '_'.join(parts[:2])
        # most_constrained / first_fractional etc.
        branch_var = '_'.join(parts[2:])

        color = color_map.get(strategy,   'gray')
        linestyle, marker = style_map.get(branch_var, ('-', 'o'))

        avg_time = subset.groupby('n')['solve_time'].mean()
        axes[0].plot(avg_time.index, avg_time.values,
                     marker=marker, label=solver, linewidth=1.8,
                     color=color, linestyle=linestyle, alpha=0.85)

        avg_nodes = subset.groupby('n')['nodes_explored'].mean()
        line, = axes[1].plot(avg_nodes.index, avg_nodes.values,
                             marker=marker, linewidth=1.8,
                             color=color, linestyle=linestyle, alpha=0.85)

        # For the nodes chart legend, only keep one entry per strategy (color)
        if strategy not in nodes_lines:
            nodes_lines[strategy] = (line, strategy)

    # --- Chart 1: full legend ---
    axes[0].set_title('Average Solve Time by Board Size')
    axes[0].set_xlabel('Board Size (n)')
    axes[0].set_ylabel('Solve Time (seconds)')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Chart 2: condensed legend (one entry per search strategy) ---
    axes[1].set_title('Average Nodes Explored by Board Size')
    axes[1].set_xlabel('Board Size (n)')
    axes[1].set_ylabel('Nodes Explored (Tree Size)')
    handles = [v[0] for v in nodes_lines.values()]
    labels = [v[1] for v in nodes_lines.values()]
    axes[1].legend(handles, labels, fontsize=9, loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # --- Auto-save as JPG ---
    results_dir = os.path.dirname(csv_path)
    jpg_path = os.path.join(results_dir, 'bnb_strategy_comparison.jpg')
    plt.savefig(jpg_path, format='jpg', dpi=150, bbox_inches='tight')
    print(f"[INFO] Chart saved to: {jpg_path}")

    plt.show()


def main():
    print_header("Running B&B Experiments (Real Mode)")

    from bnb import BnBSolver
    import statistics

    sizes = [4, 5, 6, 7, 8]
    strategies = ['best_first', 'depth_first', 'breadth_first']
    branch_vars = ['most_constrained', 'first_fractional',
                   'least_constrained', 'most_coverage']
    branch_orders = ['zero_first', 'lp_guided', 'three_way']

    results = []
    NUM_RUNS = 1

    for n in sizes:
        for strat in strategies:
            for b_var in branch_vars:
                for b_order in branch_orders:
                    run_times = []
                    run_nodes = []
                    run_knights = []

                    for run_i in range(NUM_RUNS):
                        solver = BnBSolver(
                            n=n, strategy=strat, branch_var=b_var, branch_order=b_order,
                            time_limit=600.0)
                        res = solver.solve()
                        run_times.append(res['solve_time'])
                        run_nodes.append(res['nodes_explored'])
                        run_knights.append(res['num_knights'])

                    res['solve_time'] = statistics.mean(run_times)
                    res['nodes_explored'] = int(statistics.mean(run_nodes))
                    res['solve_time_std'] = statistics.stdev(
                        run_times) if NUM_RUNS > 1 else 0.0
                    res['nodes_std'] = statistics.stdev(
                        run_nodes) if NUM_RUNS > 1 else 0.0
                    if res.get("status") != "timeout":
                        results.append(res)
                    print(f"  Done: n={n} | {strat} | {b_var} | {b_order} → "
                          f"{res['num_knights']} knights "
                          f"(avg {res['solve_time']:.3f}s ± {res['solve_time_std']:.3f}s)")

    print_comparison_table(results)
    csv_filename = "bnb_strategies_comparison.csv"
    save_results_csv(results, csv_filename)

    results_path = os.path.join(
        root_dir, 'experiments', 'results', csv_filename)
    visualize_results(results_path)


if __name__ == "__main__":
    main()