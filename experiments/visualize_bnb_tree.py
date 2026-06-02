import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(12, 7))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

ax.axis('off')


def draw_node(x, y, text, color="#d9edf7"):
    ax.text(
        x,
        y,
        text,
        ha='center',
        va='center',
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=color,
            edgecolor='black'
        )
    )


def connect(x1, y1, x2, y2):
    ax.plot(
        [x1, x2],
        [y1, y2],
        color='black',
        linewidth=1.5
    )


# Root node
draw_node(
    5,
    9,
    "Root\nLP = 5.5\nUB = 7"
)

# Level 1
draw_node(
    3,
    6.5,
    "x12 = 0\nLP = 6.0"
)

draw_node(
    7,
    6.5,
    "x12 = 1\nLP = 5.8"
)

connect(5, 8.5, 3, 7)
connect(5, 8.5, 7, 7)

# Level 2 left
draw_node(
    2,
    4,
    "Integer\nKnights = 7",
    color="#dff0d8"
)

draw_node(
    4,
    4,
    "Pruned\nLP ≥ UB",
    color="#f2dede"
)

connect(3, 6, 2, 4.5)
connect(3, 6, 4, 4.5)

# Level 2 right
draw_node(
    6,
    4,
    "x7 = 0\nLP = 6.2"
)

draw_node(
    8,
    4,
    "Pruned\nInfeasible",
    color="#f2dede"
)

connect(7, 6, 6, 4.5)
connect(7, 6, 8, 4.5)

# Final solution
draw_node(
    6,
    1.5,
    "Optimal\nKnights = 7",
    color="#dff0d8"
)

connect(6, 3.5, 6, 2)

plt.title(
    "Example Branch-and-Bound Search Tree (n=5)",
    fontsize=18,
    weight='bold'
)

plt.figtext(
    0.5,
    0.02,
    "Green nodes indicate integer feasible solutions. "
    "Red nodes indicate pruned branches.",
    ha='center',
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "experiments/results/bnb_tree_visualization.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()