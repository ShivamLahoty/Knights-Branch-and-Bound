"""
strategies.py  —  Person 3: Heuristics & Strategy Scripting Developer
======================================================================
Provides branching heuristics and variable-selection strategies for the
Branch-and-Bound engine in bnb.py.

How this fits into bnb.py
--------------------------
bnb.py already contains three variable selectors internally:
    select_most_constrained, select_least_constrained, select_first_fractional

This file:
  1. Re-exports those three selectors so the full strategy menu lives in
     one place (bnb.py can optionally import from here instead).
  2. Adds a FOURTH selector:  select_most_coverage  — a domain-aware
     rule that bnb.py does not have, giving Person 4 an extra strategy
     to benchmark.
  3. Provides pick_branch_order() — decides whether to try 0 or 1 first
     when creating child nodes.  bnb.py currently hard-codes this via
     branch_order='zero_first'/'one_first'; this function is the smarter
     LP-guided version Person 2 can call instead.
  4. Provides an enhanced greedy (greedy_with_random_restarts) that can
     replace the single-pass greedy_heuristic in bnb.py for a tighter
     initial upper bound on larger boards.

Public API summary
------------------
  select_most_constrained(x_values, fixed_vars)   → int | None
  select_least_constrained(x_values, fixed_vars)  → int | None
  select_first_fractional(x_values, fixed_vars)   → int | None
  select_most_coverage(board, x_values, fixed_vars) → int | None

  pick_branch_order(sq, x_values, strategy)        → [int, int]

  greedy_with_random_restarts(board, restarts, seed) → (int, list[int])

All selector functions match bnb.py's expected signature:
    fn(x_values: list[float], fixed_vars: dict) -> Optional[int]
so they can be dropped straight into BnBSolver.VAR_SELECTORS.
"""

from __future__ import annotations
import random
from typing import Optional
from board import Board


# ======================================================================
# HELPERS  (shared across all selectors, mirrors bnb.py internals)
# ======================================================================

def _free_fractionals(
    x_values: list[float],
    fixed_vars: dict,
) -> list[tuple[float, int]]:
    """
    Return (value, index) pairs for variables that are:
      - not already fixed  (not in fixed_vars)
      - genuinely fractional  (not within 1e-6 of 0 or 1)

    Mirrors the private _free_fractionals() inside bnb.py so all
    selectors here use identical filtering logic.
    """
    result = []
    for i, v in enumerate(x_values):
        if i in fixed_vars:
            continue
        frac = v - int(v)
        if 1e-6 < frac < 1 - 1e-6:
            result.append((v, i))
    return result


# ======================================================================
# 1. VARIABLE SELECTION STRATEGIES
# ======================================================================

def select_most_constrained(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
    """
    Pick the fractional variable closest to 0.5.

    Rationale: this variable is maximally "undecided" — fixing it forces
    the LP to change the most, giving the strongest bound improvement per
    branch.  This is the standard strong-branching proxy used in most
    commercial B&B solvers.

    Identical logic to bnb.py's select_most_constrained; re-exported
    here so callers can import everything from one place.
    """
    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_least_constrained(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
    """
    Pick the fractional variable furthest from 0.5 (closest to 0 or 1).

    Rationale: this variable already "leans" toward a natural value;
    fixing it in the LP's preferred direction disturbs the relaxation
    minimally.  Usually weaker than most_constrained but useful as a
    statistical baseline to show how much variable selection matters.
    """
    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return max(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_first_fractional(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
    """
    Pick the first fractional variable by square index order.

    Rationale: zero computation cost — the ultimate baseline.
    Benchmarking this against the smarter rules shows the real value of
    intelligent variable selection.
    """
    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[1])[1]   # smallest index


def select_most_coverage(
    board: Board,
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
    """
    *New strategy — not in bnb.py.*

    Pick the fractional variable whose square, if fixed to 1 (knight
    placed), would cover the most squares not yet covered by any
    already-fixed knight.

    Rationale: in the knight-domination problem, every constraint is
    "cover this square."  Branching on the variable that resolves the
    most still-open constraints reduces the remaining problem fastest.
    This is a domain-aware rule that exploits the problem structure,
    unlike the LP-only rules above.

    Note: signature takes `board` as a first argument.  To use inside
    BnBSolver.VAR_SELECTORS, wrap it with functools.partial:

        from functools import partial
        from strategies import select_most_coverage
        BnBSolver.VAR_SELECTORS["most_coverage"] = partial(select_most_coverage, board)

    Args:
        board      : Board instance (for attack map lookups)
        x_values   : LP relaxation values, one per square
        fixed_vars : currently fixed variables {sq: 0_or_1}

    Returns:
        Square index to branch on, or None if no fractional variable exists.
    """
    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None

    # Squares already covered by fixed-to-1 knights
    covered_by_fixed: set[int] = set()
    for sq, val in fixed_vars.items():
        if val == 1:
            for t in board.attacks_from(sq):
                covered_by_fixed.add(t)

    def coverage_score(item: tuple[float, int]) -> tuple[int, float]:
        val, sq = item
        new_coverage = sum(
            1 for t in board.attacks_from(sq)
            if t not in covered_by_fixed
        )
        # Tie-break: prefer squares closest to 0.5 (stronger branching)
        return (new_coverage, -abs(val - 0.5))

    return max(candidates, key=coverage_score)[1]


# Map of all strategy names → functions
# (board-aware select_most_coverage needs partial binding — see docstring)
SELECTORS = {
    "most_constrained":  select_most_constrained,
    "least_constrained": select_least_constrained,
    "first_fractional":  select_first_fractional,
    # "most_coverage": partial(select_most_coverage, board)  ← add in BnBSolver
}


# ======================================================================
# 2. BRANCH VALUE ORDER
# ======================================================================

def pick_branch_order(
    sq: int,
    x_values: list[float],
    strategy: str = "lp_guided",
) -> list[int]:
    """
    Decide whether to explore child node fix=1 or fix=0 first.

    bnb.py currently uses a static branch_order parameter ('zero_first'
    or 'one_first').  This function provides a smarter LP-guided option
    that adapts per variable.

    Strategies
    ----------
    'lp_guided'  (recommended):
        Round toward the LP value.  If x[sq] >= 0.5, try fix=1 first —
        the LP already "wants" to place a knight here, so we follow its
        hint.  This tends to find good integer solutions faster, tightening
        the upper bound early and enabling stronger pruning.

    'one_first':
        Always try placing a knight (fix=1) before skipping (fix=0).
        Useful when the board is large and coverage is sparse.

    'zero_first':
        Always try skipping (fix=0) first.  Default in bnb.py.
        Tends to find sparser solutions early.

    Args:
        sq        : square index being branched on
        x_values  : current LP values (x_values[sq] is the relevant one)
        strategy  : 'lp_guided' | 'one_first' | 'zero_first'

    Returns:
        [first_value, second_value]  e.g. [1, 0] or [0, 1]

    Usage in bnb.py (replace the static line at line 397):
        branch_vals = pick_branch_order(branch_idx, lp_values, strategy="lp_guided")
        for val in branch_vals: ...
    """
    if strategy == "lp_guided":
        return [1, 0] if x_values[sq] >= 0.5 else [0, 1]
    elif strategy == "one_first":
        return [1, 0]
    elif strategy == "zero_first":
        return [0, 1]
    else:
        raise ValueError(
            f"Unknown branch-order strategy '{strategy}'. "
            "Choose 'lp_guided', 'one_first', or 'zero_first'."
        )


# ======================================================================
# 3. ENHANCED GREEDY WITH RANDOM RESTARTS
# ======================================================================

def greedy_with_random_restarts(
    board: Board,
    restarts: int = 10,
    seed: int = 42,
) -> tuple[int, list[int]]:
    """
    Run the greedy heuristic multiple times with tie-breaking randomness,
    returning the best (fewest knights) result found.

    Why this improves on bnb.py's single-pass greedy
    -------------------------------------------------
    The single-pass greedy in bnb.py always picks the *first* square
    that achieves max coverage when there are ties.  On many board sizes
    (especially n=6,7,8) there are many tied squares at each step, and
    the arbitrary tie-break can lead to suboptimal solutions.

    Random restarts explore the space of tie-breaks cheaply.  Each
    restart runs in O(n^4) time; 10 restarts still finish in milliseconds
    for n ≤ 10, but the best upper bound found can be meaningfully lower,
    which translates to stronger pruning from the very first B&B node.

    Usage in bnb.py (replace the greedy_heuristic call at line 294):
        from strategies import greedy_with_random_restarts
        ub_count, ub_placement = greedy_with_random_restarts(self.board, restarts=10)

    Args:
        board    : Board instance
        restarts : number of random restarts (default 10)
        seed     : random seed for reproducibility

    Returns:
        (num_knights, placement_list) — best found across all restarts.
        Guaranteed valid cover (every square threatened).
        Returns (n*n, list(range(n*n))) only as a last-resort fallback.
    """
    rng = random.Random(seed)

    n_sq = board.num_squares
    if n_sq == 1:
        return (n_sq, list(range(n_sq)))   # n=1 is unsolvable; return fallback

    attacks_from = [list(board.attacks_from(sq)) for sq in range(n_sq)]

    best_count = n_sq + 1
    best_placement: list[int] = list(range(n_sq))

    for _ in range(restarts):
        threatened = [False] * n_sq
        placed: list[int] = []

        while not all(threatened):
            best_gain = -1
            best_candidates: list[int] = []

            for i in range(n_sq):
                gain = sum(1 for j in attacks_from[i] if not threatened[j])
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)

            if best_gain == 0:
                break  # no progress — shouldn't happen on solvable boards

            # Random tie-break: pick uniformly among all tied squares
            chosen = rng.choice(best_candidates)
            placed.append(chosen)
            for j in attacks_from[chosen]:
                threatened[j] = True

        if all(threatened) and len(placed) < best_count:
            best_count = len(placed)
            best_placement = list(placed)

    # Safety fallback if all restarts somehow failed
    if best_count > n_sq:
        return (n_sq, list(range(n_sq)))

    return (best_count, best_placement)


# ======================================================================
# SMOKE TEST
# ======================================================================

if __name__ == "__main__":
    from board import Board
    from functools import partial

    print("=" * 60)
    print("  strategies.py  —  smoke test")
    print("=" * 60)

    # ── greedy_with_random_restarts ──────────────────────────────────
    print("\n--- greedy_with_random_restarts ---")
    for n in [4, 5, 6, 7]:
        b = Board(n)
        count, placement = greedy_with_random_restarts(b, restarts=20)
        valid = b.is_valid_solution(placement)
        print(f"  n={n}: {count} knights  valid={valid}")
        assert valid, f"n={n}: greedy solution is not valid!"

    # ── variable selectors ───────────────────────────────────────────
    print("\n--- variable selectors ---")
    b5 = Board(5)

    # Fake LP: two fractional squares, rest integer
    x = [0.0] * 25
    x[7]  = 0.48   # close to 0.5 → most_constrained should pick this
    x[12] = 0.85   # far from 0.5 → least_constrained should pick this
    x[3]  = 0.6    # fractional but not extreme
    fixed: dict[int, int] = {0: 1, 1: 0}

    mc  = select_most_constrained(x, fixed)
    lc  = select_least_constrained(x, fixed)
    ff  = select_first_fractional(x, fixed)
    cov = select_most_coverage(b5, x, fixed)

    print(f"  most_constrained  → sq {mc}  (expect 7, |0.48-0.5|=0.02 is min)")
    print(f"  least_constrained → sq {lc}  (expect 12, |0.85-0.5|=0.35 is max)")
    print(f"  first_fractional  → sq {ff}  (expect 3, lowest index)")
    print(f"  most_coverage     → sq {cov} (highest uncovered attack count)")

    assert mc  == 7,  f"most_constrained failed: got {mc}"
    assert lc  == 12, f"least_constrained failed: got {lc}"
    assert ff  == 3,  f"first_fractional failed: got {ff}"

    # ── pick_branch_order ────────────────────────────────────────────
    print("\n--- pick_branch_order ---")
    print(f"  lp_guided sq=7  (x=0.48) → {pick_branch_order(7,  x, 'lp_guided')}  (expect [0,1])")
    print(f"  lp_guided sq=12 (x=0.85) → {pick_branch_order(12, x, 'lp_guided')}  (expect [1,0])")
    print(f"  one_first  sq=7          → {pick_branch_order(7,  x, 'one_first')}   (expect [1,0])")
    print(f"  zero_first sq=7          → {pick_branch_order(7,  x, 'zero_first')}  (expect [0,1])")

    assert pick_branch_order(7,  x, "lp_guided")  == [0, 1]
    assert pick_branch_order(12, x, "lp_guided")  == [1, 0]
    assert pick_branch_order(7,  x, "one_first")  == [1, 0]
    assert pick_branch_order(7,  x, "zero_first") == [0, 1]

    print("\nAll smoke tests passed ✓")
