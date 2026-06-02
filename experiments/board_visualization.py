import os
import sys
import numpy as np
import matplotlib.pyplot as plt

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from board import Board
from ilp_solver import ILPSolver
from bnb import BnBSolver

RESULTS_DIR = os.path.join(root_dir, 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_chessboard_visualization(n=8):

    solver = BnBSolver(
        n=n,
        strategy='best_first',
        branch_var='most_constrained'
    )

    result = solver.solve()

    placement = result['placement']

    board = Board(n)

    threatened = set()

    for sq in placement:
        for attacked in board.attacks_from(sq):
            threatened.add(attacked)

    grid = np.zeros((n, n))

    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                grid[r][c] = 0.85
            else:
                grid[r][c] = 0.25

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(grid, cmap='Blues')

    for sq in threatened:
        r, c = board.square_coords(sq)

        ax.add_patch(
            plt.Rectangle(
                (c - 0.5, r - 0.5),
                1,
                1,
                color='limegreen',
                fill=True,
                alpha=0.25
            )
        )

    for sq in placement:
        r, c = board.square_coords(sq)

        ax.text(
            c,
            r,
            '♞',
            ha='center',
            va='center',
            fontsize=30,
            fontweight='bold',
            color='black'
        )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels([chr(ord('A') + i) for i in range(n)])
    ax.set_yticklabels([str(i + 1) for i in range(n)])

    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)

    ax.grid(which='minor', color='black', linewidth=1.5)

    ax.set_title(
        f'Optimal Knight Placement ({n}x{n}) — {result["num_knights"]} Knights',
        fontsize=24,
        fontweight='bold',
        pad=20
    )

    ax.text(
    0.5,
    1.02,
    f'Optimal Knights: {result["num_knights"]}   |   Nodes Explored: {result["nodes_explored"]}',
    transform=ax.transAxes,
    ha='center',
    fontsize=12
)

    plt.figtext(
        0.5,
        0.02,
        '♞ = Knight placement found using Branch and Bound',
        ha='center',
        fontsize=12
    )

    save_path = os.path.join(
        RESULTS_DIR,
        'optimal_board.png'
    )

    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches='tight'
    )

    print(f"\nSaved chessboard visualization:")
    print(save_path)

    plt.show()


def create_lp_heatmap(n=8):

    solver = ILPSolver(n)

    result = solver.solve(relax=True)

    values = result['x_values']

    heatmap = np.array(values).reshape((n, n))

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        heatmap,
        cmap='viridis'
    )

    for r in range(n):
        for c in range(n):

            val = heatmap[r][c]

            ax.text(
                c,
                r,
                f"{val:.2f}",
                ha='center',
                va='center',
                fontsize=10,
                color='white'
            )

    cbar = plt.colorbar(im)
    cbar.set_label('LP Variable Value', fontsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xticklabels([chr(ord('A') + i) for i in range(n)])
    ax.set_yticklabels([str(i + 1) for i in range(n)])

    ax.set_title(
        f'LP Relaxation Heatmap ({n}x{n})',
        fontsize=24,
        fontweight='bold',
        pad=35
    )

    plt.figtext(
        0.5,
        0.02,
        'Darker colors represent variables closer to 1 in the LP relaxation',
        ha='center',
        fontsize=12
    )

    save_path = os.path.join(
        RESULTS_DIR,
        'lp_heatmap.png'
    )

    plt.savefig(
        save_path,
        dpi=400,
        bbox_inches='tight'
    )

    print(f"\nSaved LP heatmap:")
    print(save_path)

    plt.show()


def main():

    create_chessboard_visualization(n=8)

    create_lp_heatmap(n=8)


if __name__ == "__main__":
    main()