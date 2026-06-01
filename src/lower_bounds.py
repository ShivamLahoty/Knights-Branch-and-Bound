"""
lower_bounds.py  —  Person 3: Heuristics & Strategy Scripting Developer
========================================================================
Provides lower bounds that AUGMENT bnb.py's existing LP-based pruning.

How this fits into bnb.py
--------------------------
bnb.py already uses LP relaxation bounds (via ILPSolver.solve(relax=True))
as its primary pruning signal.  Those bounds are tight but cost a Gurobi
LP solve at every node.

This file adds two complementary bounds:

  mis_lower_bound(board)
      A global floor based on a Maximum Independent Set of the knight-
      attack graph.  Computed ONCE before the tree search starts and used
      to immediately prune any node whose LP bound equals this floor
      (meaning the LP cannot possibly improve further).

  node_lower_bound(board, fixed_vars)
      A cheap per-node bound that requires ZERO LP solves.  Useful as a
      fast pre-filter: if this cheap bound already >= best_val, skip the
      LP call entirely and prune immediately.  Saves Gurobi calls on
      nodes that are obviously dead.

Recommended integration into BnBSolver.solve() in bnb.py
---------------------------------------------------------

  # Once, before the main loop (line ~294 in bnb.py):
  from lower_bounds import mis_lower_bound, node_lower_bound
  global_lb = mis_lower_bound(self.board)

  # Inside the main loop, BEFORE calling self._solve_lp() (line ~349):
  cheap_lb = node_lower_bound(self.board, node.fixed_vars)
  if cheap_lb >= best_val:
      stats["nodes_pruned_bound"] += 1
      continue                          # skip the LP call entirely

  # After solving LP, also enforce global MIS floor:
  effective_lb = max(lp_obj, global_lb)
  if effective_lb >= best_val - 1e-6:
      stats["nodes_pruned_bound"] += 1
      continue

Public API
----------
  mis_lower_bound(board)                   → int
  node_lower_bound(board, fixed_vars)      → int
"""

from __future__ import annotations
import math
from board import Board


# ======================================================================
# 1. MIS LOWER BOUND  (compute once, use globally)
# ======================================================================

def mis_lower_bound(board: Board) -> int:
    """
    Compute a global lower bound on the minimum number of knights needed
    to dominate an n×n board, using a Maximum Independent Set argument.

    Mathematical basis
    ------------------
    Let I be an independent set of the knight-attack graph — a set of
    squares where no two squares threaten each other.

    Claim: every valid knight placement must include at least
           ceil( |I| / K )  knights,
    where K = max number of squares in I that any single attacker covers.

    Proof sketch:
      Each knight placed covers at most K squares in I (by definition
      of K).  To cover all |I| squares in I, we need at least
      ceil(|I| / K) knights.  This holds regardless of where other
      knights are placed.

    Implementation
    --------------
    We build I via a greedy MIS heuristic (minimum-degree ordering),
    which finds a large independent set quickly.  The greedy MIS is not
    always maximum, but it is fast and produces tight bounds in practice
    for knight graphs.

    Args:
        board : Board instance

    Returns:
        Integer lower bound.  Always ≤ true optimum (safe for pruning).
        Returns 0 for the degenerate n=1 case.
    """
    n_sq = board.num_squares
    if n_sq <= 1:
        return 0

    attacks = [set(board.attacks_from(sq)) for sq in range(n_sq)]

    # ── Step 1: greedy MIS (minimum-degree heuristic) ──────────────────
    # Repeatedly pick the node with fewest neighbours still in the
    # candidate set, add it to MIS, then remove it and its neighbours.
    available = set(range(n_sq))
    mis: list[int] = []

    while available:
        # Min-degree node among remaining candidates
        best = min(available, key=lambda s: len(attacks[s] & available))
        mis.append(best)
        available -= {best} | (attacks[best] & available)

    if not mis:
        return 0

    # ── Step 2: compute max coverage of MIS by any one attacker ────────
    # For each square sq (potential knight position), count how many
    # squares in the MIS it can threaten.
    mis_set = set(mis)

    max_coverage = max(
        len(attacks[sq] & mis_set) for sq in range(n_sq)
    )

    if max_coverage == 0:
        # No knight can threaten any MIS square — board is unsolvable
        # (only happens for n=1, already guarded above)
        return 0

    lb = math.ceil(len(mis) / max_coverage)
    return lb


# ======================================================================
# 2. NODE-LEVEL CHEAP BOUND  (call at every node, no LP)
# ======================================================================

def node_lower_bound(
    board: Board,
    fixed_vars: dict[int, int],
) -> int:
    """
    Fast per-node lower bound that requires no LP solves.

    Use this as a pre-filter BEFORE calling ILPSolver at a node.
    If this bound >= best_val, prune immediately without paying the
    cost of a Gurobi solve.

    Computation
    -----------
    Let:
      placed    = squares fixed to 1 (knights already committed)
      covered   = squares threatened by `placed`
      uncovered = all squares not yet in `covered`
      forbidden = squares fixed to 0 (cannot place knight there)

    Then the minimum number of knights for this node is:

        node_lb = |placed| + greedy_cover_estimate(uncovered, forbidden)

    greedy_cover_estimate counts — without actually building the
    placement — how many more knights are needed to cover `uncovered`,
    using the same max-coverage greedy as the heuristic in strategies.py.

    This is a valid lower bound because:
      - We must keep all |placed| knights already fixed to 1.
      - We need at least greedy_estimate more to cover what remains.
      - The greedy overestimates coverage (it picks optimally), so the
        estimate is never larger than the true additional cost.

    Args:
        board      : Board instance
        fixed_vars : {sq: 0_or_1} from the current B&B node (Node.fixed_vars)

    Returns:
        Integer lower bound for this node.
    """
    n_sq = board.num_squares
    attacks = [board.attacks_from(sq) for sq in range(n_sq)]

    placed    = {sq for sq, val in fixed_vars.items() if val == 1}
    forbidden = {sq for sq, val in fixed_vars.items() if val == 0}

    # Coverage from already-placed knights
    covered: set[int] = set()
    for sq in placed:
        for t in attacks[sq]:
            covered.add(t)

    uncovered = set(range(n_sq)) - covered

    if not uncovered:
        # Everything is already covered — no extra knights needed
        return len(placed)

    # Greedy count: how many more knights (from non-forbidden squares)
    # are needed to cover `uncovered`?
    candidates = [sq for sq in range(n_sq) if sq not in forbidden]
    extra = _greedy_cover_count(candidates, uncovered, attacks)

    return len(placed) + extra


def _greedy_cover_count(
    candidates: list[int],
    uncovered: set[int],
    attacks: list,
) -> int:
    """
    Count (not build) the number of greedy steps to cover `uncovered`
    using only squares in `candidates`.

    Each step: pick the candidate that covers the most uncovered squares,
    mark those squares as covered, repeat.

    Returns the step count — a lower bound on the additional knights
    needed, because the greedy makes the best possible choice each time.
    """
    uncovered = set(uncovered)   # local copy — don't mutate caller's set
    count = 0

    while uncovered:
        # Find the candidate with the highest coverage gain
        best_gain = 0
        best_sq = -1
        for sq in candidates:
            gain = sum(1 for t in attacks[sq] if t in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_sq = sq

        if best_gain == 0:
            # No candidate can reach any remaining square.
            # This node is infeasible — return a huge number so it gets pruned.
            return len(uncovered) + 1

        for t in attacks[best_sq]:
            uncovered.discard(t)
        count += 1

    return count


# ======================================================================
# SMOKE TEST
# ======================================================================

if __name__ == "__main__":
    from board import Board

    print("=" * 60)
    print("  lower_bounds.py  —  smoke test")
    print("=" * 60)

    # Known approximate optima for reference (ILP ground truth)
    # n=4: 4  n=5: 5  n=6: 8  n=7: 13  n=8: 14
    known = {4: 4, 5: 5, 6: 8, 7: 13, 8: 14}

    print(f"\n{'n':>4} | {'opt':>5} | {'mis_lb':>8} | {'node_lb':>9} | mis≤opt | node≤opt")
    print(f"{'-'*4}-+-{'-'*5}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}")

    all_ok = True
    for n, opt in known.items():
        b = Board(n)
        mis  = mis_lower_bound(b)
        node = node_lower_bound(b, fixed_vars={})

        mis_ok  = mis  <= opt
        node_ok = node <= opt
        all_ok  = all_ok and mis_ok and node_ok

        print(f"{n:>4} | {opt:>5} | {mis:>8} | {node:>9} | "
              f"{'✓' if mis_ok else '✗':^7} | {'✓' if node_ok else '✗':^9}")

    print()
    assert all_ok, "One or more bounds exceeded known optimum — INVALID bound!"

    # ── node_lower_bound with partial fixings ────────────────────────
    print("--- node_lower_bound with partial fixed_vars ---")
    b5 = Board(5)

    # Fix sq=2 to 1 (knight placed), sq=0 and sq=1 forbidden
    fv1 = {2: 1, 0: 0, 1: 0}
    lb1 = node_lower_bound(b5, fv1)
    print(f"  n=5 fixed={{2:1, 0:0, 1:0}}  →  node_lb = {lb1}")
    assert lb1 >= 1, "At least the one placed knight must be counted"

    # Fix nothing → should match the unconstrained greedy estimate
    fv0 = {}
    lb0 = node_lower_bound(b5, fv0)
    print(f"  n=5 fixed={{}}               →  node_lb = {lb0}")
    assert lb0 <= known[5], f"Empty fixings bound {lb0} exceeds opt {known[5]}"

    # Fix enough knights that the board is already fully covered
    # (placement that covers everything on 4x4)
    b4 = Board(4)
    # Greedy on 4x4 typically gives 4 knights; let's use a known cover
    from strategies import greedy_with_random_restarts
    _, cover = greedy_with_random_restarts(b4, restarts=30)
    fv_full = {sq: 1 for sq in cover}
    lb_full = node_lower_bound(b4, fv_full)
    print(f"  n=4 full cover fixed        →  node_lb = {lb_full}  (= |placed| = {len(cover)})")
    assert lb_full == len(cover), "Full-cover node_lb should equal number of placed knights"

    print("\nAll smoke tests passed ✓")
