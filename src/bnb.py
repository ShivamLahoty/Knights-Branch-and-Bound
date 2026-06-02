"""
bnb.py — Branch and Bound Solver for Knights on the Chessboard 2
=================================================================

Plugs directly into the project structure:
  - Uses Board      from board.py      (attack map, validation, display)
  - Uses ILPSolver  from ilp_solver.py (LP relaxation at each node)
  - Returns         make_result(...)   from utils.py (standard result dict)

Called by main.py as:
    solver = BnBSolver(n=n, strategy=strategy, branch_var=branch_var, verbose=verbose)
    result = solver.solve()

Strategies supported
---------------------
Node selection  (--strategy):
    best_first      explore lowest LP bound first  [default, best pruning]
    depth_first     explore deepest node first      [memory-efficient]
    breadth_first   explore shallowest node first   [rarely used, for comparison]

Variable selection  (--branch_var):
    most_constrained    branch on var closest to 0.5  [default, strongest branching]
    least_constrained   branch on var furthest from 0.5
    first_fractional    first fractional variable found  [fast baseline]
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Optional

from board import Board
from ilp_solver import ILPSolver
from utils import make_result


@dataclass
class Node:

    lb:         float
    depth:      int
    fixed_vars: dict = field(default_factory=dict)

    def __lt__(self, other):
        # Needed so heapq can compare Nodes when priorities tie
        return self.lb < other.lb


def greedy_heuristic(board: Board) -> tuple[int, list[int]]:

    n_sq = board.num_squares
    threatened = [False] * n_sq
    placed = []

    attacks_from = [set(board.attacks_from(sq)) for sq in range(n_sq)]

    while not all(threatened):
        best_sq, best_gain = -1, -1

        for i in range(n_sq):
            gain = sum(1 for j in attacks_from[i] if not threatened[j])
            if gain > best_gain:
                best_gain = gain
                best_sq = i

        if best_sq == -1 or best_gain == 0:
            break  # no progress — shouldn't happen on solvable boards

        placed.append(best_sq)
        for j in attacks_from[best_sq]:
            threatened[j] = True

    # Fallback: if heuristic somehow failed, place everywhere
    if not all(threatened):
        return n_sq, list(range(n_sq))

    return len(placed), placed


# ─────────────────────────────────────────────────────────────────────────────
# Variable selection strategies
# ─────────────────────────────────────────────────────────────────────────────

def _free_fractionals(x_values: list[float], fixed_vars: dict) -> list[tuple[float, int]]:

    result = []
    for i, v in enumerate(x_values):
        if i in fixed_vars:
            continue
        frac = v - int(v)
        if 1e-6 < frac < 1 - 1e-6:
            result.append((v, i))
    return result


def select_most_constrained(x_values: list[float], fixed_vars: dict) -> Optional[int]:

    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    # Sort by distance from 0.5 (ascending = closest first)
    return min(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_least_constrained(x_values: list[float], fixed_vars: dict) -> Optional[int]:

    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return max(candidates, key=lambda t: abs(t[0] - 0.5))[1]


def select_first_fractional(x_values: list[float], fixed_vars: dict) -> Optional[int]:

    candidates = _free_fractionals(x_values, fixed_vars)
    if not candidates:
        return None
    return min(candidates, key=lambda t: t[1])[1]   # smallest index


class BnBSolver:

    # Map string names (from main.py argparse) to selector functions
    VAR_SELECTORS = {
        "most_constrained":  select_most_constrained,
        "least_constrained": select_least_constrained,
        "first_fractional":  select_first_fractional,
        # most_coverage is added per-instance in __init__ (needs self.board)
    }

    def __init__(
        self,
        n:            int,
        strategy:     str = "best_first",
        branch_var:   str = "most_constrained",
        branch_order: str = "zero_first",
        time_limit:   float = 300.0,
        verbose:      bool = False,
    ):
        if strategy not in ("best_first", "depth_first", "breadth_first"):
            raise ValueError(f"Unknown strategy '{strategy}'")

        self.n = n
        self.strategy = strategy
        self.branch_var = branch_var
        self.branch_order = branch_order
        self.time_limit = time_limit
        self.verbose = verbose

       # Shared objects (built once)
        self.board = Board(n)
        self.lp_engine = ILPSolver(n)   # reused for every LP solve

        # Build per-instance selector dict so most_coverage can use self.board
        from functools import partial
        from heuristics import select_most_coverage
        instance_selectors = dict(self.VAR_SELECTORS)
        instance_selectors["most_coverage"] = partial(
            select_most_coverage, self.board)

        if branch_var not in instance_selectors:
            raise ValueError(f"Unknown branch_var '{branch_var}'")
        self._var_select = instance_selectors[branch_var]

    # ── Heap helpers ─────────────────────────────────────────────────────────

    def _priority(self, node: Node) -> tuple:

        if self.strategy == "best_first":
            return (node.lb, node.depth)
        elif self.strategy == "depth_first":
            return (-node.depth, node.lb)
        else:  # breadth_first
            return (node.depth, node.lb)

    def _push(self, heap: list, node: Node, counter: list):
        """Push a node onto the heap with tie-breaking counter."""
        heapq.heappush(heap, (self._priority(node), counter[0], node))
        counter[0] += 1

    # ── LP solve wrapper ─────────────────────────────────────────────────────

    def _solve_lp(self, fixed_vars: dict) -> dict:

        raw = self.lp_engine.solve(
            relax=True, fixed_vars=fixed_vars, verbose=False)

        # Guarantee these keys always exist
        raw.setdefault("obj_value", None)
        raw.setdefault("x_values", [])

        # If Gurobi gave us something other than optimal/infeasible,
        # treat it as infeasible so the node gets pruned safely.
        if raw["status"] not in ("optimal", "infeasible"):
            if self.verbose:
                print(
                    f"[BnB] Unexpected LP status: {raw['status']} — treating as infeasible")
            raw["status"] = "infeasible"

        return raw

    # ── Main solve ────────────────────────────────────────────────────────────

    def solve(self) -> dict:

        start_time = time.perf_counter()

        # ── Stats counters ────────────────────────────────────────────────
        stats = {
            "nodes_explored":          0,
            "nodes_pruned_bound":      0,
            "nodes_pruned_infeasible": 0,
            "nodes_integer":           0,
        }

        # ── Step 1: greedy heuristic → initial upper bound ────────────────
        from heuristics import greedy_with_random_restarts
        ub_count, ub_placement = greedy_with_random_restarts(
            self.board, restarts=10)
        best_val = ub_count
        best_placement = list(ub_placement)

        if self.verbose:
            print(f"\n[BnB] n={self.n} | strategy={self.strategy} "
                  f"| branch_var={self.branch_var}")
            print(f"[BnB] Heuristic upper bound: {best_val} knights")

        # ── Step 2: root node (LP relaxation with no fixings) ─────────────
        root_result = self._solve_lp({})

        if root_result["status"] == "infeasible":
            if self.verbose:
                print("[BnB] Root LP infeasible — problem unsolvable.")
            elapsed = time.perf_counter() - start_time
            return make_result(
                solver=f"BnB_{self.strategy}",
                n=self.n,
                status="infeasible",
                solve_time=elapsed,
            )

        root_lb = root_result["obj_value"]
# ── MIS global lower bound — better than LP relaxation ────────────
        from lower_bounds import mis_lower_bound
        mis_lb = mis_lower_bound(self.board)
        root_lb = max(root_lb, float(mis_lb))   # tighten with MIS bound

        if self.verbose:
            print(f"[BnB] Root LP lower bound:  {root_result['obj_value']:.4f}")
            print(f"[BnB] MIS lower bound:      {mis_lb}")
            print(f"[BnB] Effective root bound: {root_lb:.4f}")

        root_node = Node(lb=root_lb, depth=0, fixed_vars={})

        # ── Step 3: initialise the open-node heap ─────────────────────────
        heap = []
        # tie-breaker for heap (list so inner funcs can mutate)
        counter = [0]
        self._push(heap, root_node, counter)

        # ── Step 4: main loop ─────────────────────────────────────────────
        while heap:

            # Time limit check
            if time.perf_counter() - start_time > self.time_limit:
                if self.verbose:
                    print("[BnB] Time limit reached.")
                break

            _, _, node = heapq.heappop(heap)
            stats["nodes_explored"] += 1

            # ── Prune by bound (before solving) ───────────────────────────
            # If this node's lb is already >= best integer solution,
            # it can NEVER improve → skip it.
            if node.lb >= best_val - 1e-6:
                stats["nodes_pruned_bound"] += 1
                continue

            # ── Solve LP at this node ─────────────────────────────────────
            lp_result = self._solve_lp(node.fixed_vars)

            if lp_result["status"] == "infeasible":
                # This fixing combination is infeasible → dead branch
                stats["nodes_pruned_infeasible"] += 1
                continue

            lp_obj = lp_result["obj_value"]
            lp_values = lp_result["x_values"]

            # ── Prune by bound (after solving) — use tighter of LP and node_lb ──
            from lower_bounds import node_lower_bound
            node_lb = node_lower_bound(self.board, node.fixed_vars)
            effective_lb = max(lp_obj, node_lb)
            if effective_lb >= best_val - 1e-6:
                stats["nodes_pruned_bound"] += 1
                continue

            # ── Check if LP solution is already all-integer ───────────────
            is_integer = all(
                abs(v - round(v)) < 1e-6
                for i, v in enumerate(lp_values)
                if i not in node.fixed_vars
            )

            if is_integer:
                # This IS a valid integer solution
                int_val = int(round(lp_obj))
                int_placement = [i for i, v in enumerate(lp_values) if v > 0.5]
                stats["nodes_integer"] += 1

                # Verify with board logic (sanity check)
                if self.board.is_valid_solution(int_placement) and int_val < best_val:
                    best_val = int_val
                    best_placement = int_placement
                    elapsed_so_far = time.perf_counter() - start_time
                    if self.verbose:
                        print(f"[BnB] ✓ New best: {best_val} knights "
                              f"(node #{stats['nodes_explored']}, "
                              f"t={elapsed_so_far:.2f}s)")
                continue   # no children needed — leaf node

            # ── Select branching variable ─────────────────────────────────
            branch_idx = self._var_select(lp_values, node.fixed_vars)

            if branch_idx is None:
                # All free variables are integer — shouldn't reach here
                # but safe fallback
                continue

            from heuristics import pick_branch_order
            if self.branch_order == "lp_guided":
                branch_vals = pick_branch_order(
                    branch_idx, lp_values, strategy="lp_guided")
            elif self.branch_order == "zero_first":
                branch_vals = [0, 1]
            else:
                branch_vals = [1, 0]

# ── Build child fixings list ──────────────────────────────────
            # Standard two children: fix=0 and fix=1
            children_fixings = []
            for val in branch_vals:
                cf = dict(node.fixed_vars)
                cf[branch_idx] = val
                children_fixings.append(cf)

            # ── Three-way: add LP-rounded third child ─────────────────────
            if self.branch_order == "three_way":
                lp_mid = 1 if lp_values[branch_idx] >= 0.5 else 0
                cf_mid = dict(node.fixed_vars)
                cf_mid[branch_idx] = lp_mid
                if cf_mid not in children_fixings:
                    children_fixings.append(cf_mid)

            for child_fixings in children_fixings:
                child_lp = self._solve_lp(child_fixings)

                if child_lp["status"] == "infeasible":
                    stats["nodes_pruned_infeasible"] += 1
                    continue

                child_lb = child_lp["obj_value"]

                if child_lb >= best_val - 1e-6:
                    stats["nodes_pruned_bound"] += 1
                    continue

                child_node = Node(
                    lb=child_lb,
                    depth=node.depth + 1,
                    fixed_vars=child_fixings,
                )
                self._push(heap, child_node, counter)

        # WITH THIS:
        elapsed = time.perf_counter() - start_time   # ← THIS is the bug fix

        if not heap:
            status = "optimal"
        else:
            status = "timeout"

        if self.verbose:
            print(f"\n[BnB] ── Final Results ───────────────────────────────")
            print(f"[BnB]   Status           : {status}")
            print(f"[BnB]   Optimal knights  : {best_val}")
            print(f"[BnB]   Nodes explored   : {stats['nodes_explored']}")
            print(f"[BnB]   Pruned (bound)   : {stats['nodes_pruned_bound']}")
            print(
                f"[BnB]   Pruned (infeasib): {stats['nodes_pruned_infeasible']}")
            print(f"[BnB]   Integer solns    : {stats['nodes_integer']}")
            print(f"[BnB]   Total time       : {elapsed:.3f}s")
            print(f"[BnB] ───────────────────────────────────────────────────")

        return make_result(
            solver=f"{self.strategy}_{self.branch_var}",
            n=self.n,
            status=status,
            num_knights=best_val,
            placement=best_placement,
            solve_time=elapsed,
            nodes_explored=stats["nodes_explored"],
            upper_bound=best_val,
            lower_bound=root_lb,
            extra={
                "nodes_pruned_bound":      stats["nodes_pruned_bound"],
                "nodes_pruned_infeasible": stats["nodes_pruned_infeasible"],
                "nodes_integer":           stats["nodes_integer"],
                "branch_order":            self.branch_order,
            },
        )


if __name__ == "__main__":
    from utils import print_comparison_table

    print("=" * 60)
    print("bnb.py smoke test")
    print("=" * 60)

    # Test 1: single solve with verbose output
    solver = BnBSolver(n=5, strategy="best_first",
                       branch_var="most_constrained", verbose=True)
    result = solver.solve()

    board = Board(5)
    board.display(result["placement"])

    # Test 2: verify against ILP ground truth
    from ilp_solver import ILPSolver
    ilp = ILPSolver(5)
    ilp_result = ilp.solve()

    print(f"\nILP says: {ilp_result['num_knights']} knights")
    print(f"BnB says: {result['num_knights']} knights")
    match = ilp_result['num_knights'] == result['num_knights']
    print(f"Match: {'✓ PASSED' if match else '✗ FAILED'}")

    # Test 3: compare all strategy combinations on n=5
    print("\n\nStrategy comparison on 5×5 board:")
    comparison = []
    for strat in ["best_first", "depth_first", "breadth_first"]:
        for bvar in ["most_constrained", "first_fractional"]:
            s = BnBSolver(n=5, strategy=strat, branch_var=bvar, verbose=False)
            r = s.solve()
            comparison.append(r)

    print_comparison_table(comparison)

    print("Smoke test complete ✓")