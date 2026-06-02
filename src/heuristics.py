

from __future__ import annotations
import random
from typing import Optional
from board import Board




def _free_fractionals(
    x_values: list[float],
    fixed_vars: dict,
) -> list[tuple[float, int]]:

    result = []
    for i, v in enumerate(x_values):
        if i in fixed_vars:
            continue
        frac = v - int(v)
        if 1e-6 < frac < 1 - 1e-6:
            result.append((v, i))
    return result



def select_most_constrained(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_least_constrained(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:

    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return max(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_first_fractional(
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:

    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[1])[1]   # smallest index


def select_most_coverage(
    board: Board,
    x_values: list[float],
    fixed_vars: dict,
) -> Optional[int]:
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
    "most_coverage":     select_most_coverage,
   
}




def pick_branch_order(
    sq: int,
    x_values: list[float],
    strategy: str = "lp_guided",
) -> list[int]:

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

def greedy_with_random_restarts(
    board: Board,
    restarts: int = 10,
    seed: int = 42,
) -> tuple[int, list[int]]:

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

if __name__ == "__main__":
    from board import Board
    from functools import partial

    print("=" * 60)
    print("  strategies.py  —  smoke test")
    print("=" * 60)

    print("\n--- greedy_with_random_restarts ---")
    for n in [4, 5, 6, 7]:
        b = Board(n)
        count, placement = greedy_with_random_restarts(b, restarts=20)
        valid = b.is_valid_solution(placement)
        print(f"  n={n}: {count} knights  valid={valid}")
        assert valid, f"n={n}: greedy solution is not valid!"

    print("\n--- variable selectors ---")
    b5 = Board(5)

    x = [0.0] * 25
    x[7] = 0.48   # close to 0.5 → most_constrained should pick this
    x[12] = 0.85   # far from 0.5 → least_constrained should pick this
    x[3] = 0.6    # fractional but not extreme
    fixed: dict[int, int] = {0: 1, 1: 0}

    mc = select_most_constrained(x, fixed)
    lc = select_least_constrained(x, fixed)
    ff = select_first_fractional(x, fixed)
    cov = select_most_coverage(b5, x, fixed)

    print(f"  most_constrained  → sq {mc}  (expect 7, |0.48-0.5|=0.02 is min)")
    print(
        f"  least_constrained → sq {lc}  (expect 12, |0.85-0.5|=0.35 is max)")
    print(f"  first_fractional  → sq {ff}  (expect 3, lowest index)")
    print(f"  most_coverage     → sq {cov} (highest uncovered attack count)")

    assert mc == 7,  f"most_constrained failed: got {mc}"
    assert lc == 12, f"least_constrained failed: got {lc}"
    assert ff == 3,  f"first_fractional failed: got {ff}"

    # ── pick_branch_order ────────────────────────────────────────────
    print("\n--- pick_branch_order ---")
    print(
        f"  lp_guided sq=7  (x=0.48) → {pick_branch_order(7,  x, 'lp_guided')}  (expect [0,1])")
    print(
        f"  lp_guided sq=12 (x=0.85) → {pick_branch_order(12, x, 'lp_guided')}  (expect [1,0])")
    print(
        f"  one_first  sq=7          → {pick_branch_order(7,  x, 'one_first')}   (expect [1,0])")
    print(
        f"  zero_first sq=7          → {pick_branch_order(7,  x, 'zero_first')}  (expect [0,1])")

    assert pick_branch_order(7,  x, "lp_guided") == [0, 1]
    assert pick_branch_order(12, x, "lp_guided") == [1, 0]
    assert pick_branch_order(7,  x, "one_first") == [1, 0]
    assert pick_branch_order(7,  x, "zero_first") == [0, 1]

    print("\nAll smoke tests passed ✓")
