"""
lower_bounds.py  —  Person 3: Heuristics & Strategy Scripting Developer
========================================================================
Provides lower bounds that AUGMENT bnb.py's existing LP-based pruning.

How this fits into bnb.py
--------------------------
bnb.py already uses LP relaxation bounds (via ILPSolver.solve(relax=True))
as its primary pruning signal.  Those bounds are tight but cost an LP
solve at every node.

This file adds two complementary bounds that require NO LP solves:

  mis_lower_bound(board)
      A global floor based on a Maximum Independent Set of the knight-
      attack graph.  Compute ONCE before the tree starts.

  node_lower_bound(board, fixed_vars)
      A cheap per-node bound (zero LP calls).  Use as a fast pre-filter:
      if this already >= best_val, prune without calling the LP at all.

Recommended integration into BnBSolver.solve() in bnb.py
---------------------------------------------------------

  # Once, before the main loop:
  from lower_bounds import mis_lower_bound, node_lower_bound
  global_lb = mis_lower_bound(self.board)

  # Inside the main loop, BEFORE calling _solve_lp():
  cheap_lb = node_lower_bound(self.board, node.fixed_vars)
  if cheap_lb >= best_val:
      stats["nodes_pruned_bound"] += 1
      continue   # skip the LP call entirely

  # After solving LP, tighten with global MIS floor:
  effective_lb = max(lp_obj, global_lb)
  if effective_lb >= best_val - 1e-6:
      stats["nodes_pruned_bound"] += 1
      continue

Public API
----------
  mis_lower_bound(board)               -> int
  node_lower_bound(board, fixed_vars)  -> int
"""

from __future__ import annotations
import math
from board import Board


# ======================================================================
# 1. MIS LOWER BOUND  (compute once, use globally)
# ======================================================================

def mis_lower_bound(board: Board) -> int:
    """
    Global lower bound derived from a Maximum Independent Set (MIS) of
    the knight-attack graph.

    Mathematical basis
    ------------------
    Let I be an independent set — squares where no two attack each other.
    Each knight can threaten at most K squares in I, where K is the
    maximum number of MIS squares any single attacker can reach.

    Therefore:   OPT  >=  ceil( |I| / K )

    We build I via a greedy minimum-degree heuristic (fast, O(n^2 log n)).

    Args:
        board : Board instance

    Returns:
        Integer lower bound, always <= true optimum (safe for pruning).
        Returns 0 for the degenerate n=1 case.
    """
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


# ======================================================================
# 2. NODE-LEVEL CHEAP BOUND  (call at every node, no LP)
# ======================================================================

def node_lower_bound(
    board: Board,
    fixed_vars: dict[int, int],
) -> int:
    """
    Fast per-node lower bound requiring zero LP solves.

    Mathematical basis
    ------------------
    Key fact: every knight attacks at most 8 squares.  Therefore to
    cover U uncovered squares you need AT LEAST ceil(U / 8) knights,
    no matter how optimally you place them.

    This gives us:

        node_lb  =  |placed|  +  ceil( |uncovered| / 8 )

    where:
      placed    = squares already fixed to 1 (committed knights)
      covered   = squares threatened by `placed`
      uncovered = all squares not yet in `covered`

    This bound is ALWAYS valid (never exceeds the true optimum) because:
      - We must keep every placed knight  →  cost >= |placed|
      - Each additional knight covers at most 8 new squares
        →  additional cost >= ceil(uncovered / 8)
      - The two costs are independent and additive

    Note: using a greedy estimate here would be WRONG — a greedy
    solution is an upper bound, not a lower bound, and would exceed
    the optimum, making pruning unsound.

    Args:
        board      : Board instance
        fixed_vars : {sq: 0_or_1} from the current B&B node

    Returns:
        Integer lower bound for this node.
    """
    n_sq   = board.num_squares
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


# ======================================================================
# SMOKE TEST
# ======================================================================

if __name__ == "__main__":
    from board import Board

    print("=" * 60)
    print("  lower_bounds.py  —  smoke test")
    print("=" * 60)

    # Known optima (ILP ground truth for this problem variant)
    known = {4: 4, 5: 5, 6: 8, 7: 13, 8: 14}

    print(f"\n{'n':>4} | {'opt':>5} | {'mis_lb':>8} | {'node_lb':>9} | mis<=opt | node<=opt")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")

    all_ok = True
    for n, opt in known.items():
        b = Board(n)
        mis  = mis_lower_bound(b)
        node = node_lower_bound(b, fixed_vars={})

        mis_ok  = mis  <= opt
        node_ok = node <= opt
        all_ok  = all_ok and mis_ok and node_ok

        print(f"{n:>4} | {opt:>5} | {mis:>8} | {node:>9} | "
              f"{'OK' if mis_ok  else 'FAIL':^8} | "
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

    # Full cover: node_lb should equal exactly the number placed
    b4 = Board(4)
    # Use a known valid cover for n=4
    from strategies import greedy_with_random_restarts
    _, cover = greedy_with_random_restarts(b4, restarts=30)
    assert b4.is_valid_solution(cover), "greedy cover must be valid"
    fv_full = {sq: 1 for sq in cover}
    lb_full = node_lower_bound(b4, fv_full)
    print(f"  n=4 full cover ({len(cover)} knights)  ->  node_lb = {lb_full}")
    assert lb_full <= len(cover), f"node_lb={lb_full} should be <= placed={len(cover)}"

    print("\nAll smoke tests passed!")
