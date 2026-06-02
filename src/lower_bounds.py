
from __future__ import annotations
import math
from board import Board


def mis_lower_bound(board: Board) -> int:

    n_sq = board.num_squares
    if n_sq <= 1:
        return 0

    attacks = [set(board.attacks_from(sq)) for sq in range(n_sq)]

    # Greedy MIS: repeatedly pick minimum-degree node, add to MIS,
    # remove it and its neighbours from the candidate set.
    available = set(range(n_sq))
    mis: list[int] = []

    while available:
        best = min(available, key=lambda s: len(attacks[s] & available))
        mis.append(best)
        available -= {best} | (attacks[best] & available)

    if not mis:
        return 0

    mis_set = set(mis)
    max_coverage = max(len(attacks[sq] & mis_set) for sq in range(n_sq))

    if max_coverage == 0:
        return 0

    return math.ceil(len(mis) / max_coverage)


def node_lower_bound(
    board: Board,
    fixed_vars: dict[int, int],
) -> int:

    n_sq = board.num_squares
    placed = {sq for sq, val in fixed_vars.items() if val == 1}

    # Squares covered by already-placed knights
    covered: set[int] = set()
    for sq in placed:
        for t in board.attacks_from(sq):
            covered.add(t)

    uncovered_count = n_sq - len(covered)

    if uncovered_count <= 0:
        return len(placed)

    # Each knight covers at most 8 squares → ceil(uncovered / 8) more needed
    extra = math.ceil(uncovered_count / 8)
    return len(placed) + extra


if __name__ == "__main__":
    from board import Board

    print("=" * 60)
    print("  lower_bounds.py  —  smoke test")
    print("=" * 60)

    # Known optima (ILP ground truth for this problem variant)

    known = {
        4: 6,
        5: 7,
        6: 8,
        7: 10,
        8: 14
    }

    print(
        f"\n{'n':>4} | {'opt':>5} | {'mis_lb':>8} | {'node_lb':>9} | mis<=opt | node<=opt")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")

    all_ok = True
    for n, opt in known.items():
        b = Board(n)
        mis = mis_lower_bound(b)
        node = node_lower_bound(b, fixed_vars={})

        mis_ok = mis <= opt
        node_ok = node <= opt
        all_ok = all_ok and mis_ok and node_ok

        print(f"{n:>4} | {opt:>5} | {mis:>8} | {node:>9} | "
              f"{'OK' if mis_ok else 'FAIL':^8} | "
              f"{'OK' if node_ok else 'FAIL':^10}")

    print()
    assert all_ok, "One or more bounds exceeded known optimum — INVALID bound!"

    # ── node_lower_bound with partial fixings ────────────────────────
    print("--- node_lower_bound with partial fixed_vars ---")
    b5 = Board(5)

    fv1 = {2: 1, 0: 0, 1: 0}
    lb1 = node_lower_bound(b5, fv1)
    print(f"  n=5 fixed={{2:1, 0:0, 1:0}}  ->  node_lb = {lb1}")
    assert lb1 >= 1

    fv0 = {}
    lb0 = node_lower_bound(b5, fv0)
    print(f"  n=5 fixed={{}}               ->  node_lb = {lb0}")
    assert lb0 <= known[5], f"node_lb={lb0} exceeds opt={known[5]}"

   

    print("\nAll smoke tests passed!")
