import os
import sys
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# --- PATH ADJUSTMENT ---
# Find the project root (one folder up from 'experiments')
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Point to the 'src' directory
src_dir = os.path.join(root_dir, 'src')

# Add 'src' to Python's system path so it can find utils.py and bnb.py
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now we can safely import from utils
from utils import make_result, save_results_csv, print_comparison_table, print_header

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
    """Reads the CSV and opens an interactive matplotlib window."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("[ERROR] CSV not found. Run experiments first.")
        return

    print("\nOpening interactive visualization window...")
    
    # Create a layout with 2 side-by-side charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title('Branch & Bound Strategy Comparison')

    # Chart 1: Solve Time vs Board Size
    for solver in df['solver'].unique():
        subset = df[df['solver'] == solver]
        # Grouping by 'n' to calculate the average solve time for each board size
        avg_time = subset.groupby('n')['solve_time'].mean()
        axes[0].plot(avg_time.index, avg_time.values, marker='o', label=solver, linewidth=2)
        
    axes[0].set_title('Average Solve Time by Board Size')
    axes[0].set_xlabel('Board Size (n)')
    axes[0].set_ylabel('Solve Time (seconds)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Chart 2: Nodes Explored vs Board Size
    for solver in df['solver'].unique():
        subset = df[df['solver'] == solver]
        avg_nodes = subset.groupby('n')['nodes_explored'].mean()
        axes[1].plot(avg_nodes.index, avg_nodes.values, marker='s', label=solver, linewidth=2)
        
    axes[1].set_title('Average Nodes Explored by Board Size')
    axes[1].set_xlabel('Board Size (n)')
    axes[1].set_ylabel('Nodes Explored (Tree Size)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show() # This command actually opens the interactive window!

def main():
    print_header("Running B&B Experiments (Mock Mode)")
    
    # 1. Define the experiment grid
    sizes = [4, 5, 6, 7, 8, 9]
    strategies = ['best_first', 'depth_first', 'breadth_first']
    branch_vars = ['most_constrained', 'first_fractional', 'least_constrained']
    
    results = []
    
    # 2. Run the combinations
    for n in sizes:
        for strat in strategies:
            for b_var in branch_vars:
                
                # -------------------------------------------------------------
                # TODO: Replace this mock block with the real BnBSolver
                # -------------------------------------------------------------
                # from bnb import BnBSolver
                # solver = BnBSolver(n=n, strategy=strat, branch_var=b_var)
                # res = solver.solve()
                
                res = mock_bnb_solve(n, strat, b_var) 
                results.append(res)
                
    # 3. Output and save the results
    print_comparison_table(results)
    save_results_csv(results, "bnb_strategies_comparison.csv")
    # 3. Output and save the results
    print_comparison_table(results)
    csv_filename = "bnb_strategies_comparison.csv"
    save_results_csv(results, csv_filename)
    
    # 4. Open the interactive presentation window
    # Make sure to point to the correct folder!
    results_path = os.path.join(root_dir, 'experiments', 'results', csv_filename)
    visualize_results(results_path)
    
if __name__ == "__main__":
    main()