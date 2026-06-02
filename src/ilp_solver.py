import time
from board import Board

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("[WARNING] gurobipy not found. Install Gurobi and run again.")


class ILPSolver:

    def __init__(self, n: int):
        """
        Args:
            n: board size (n x n)
        """
        self.n = n
        self.board = Board(n)
        self.attack_matrix = self.board.get_attack_matrix()

    def solve(
        self,
        verbose: bool = False,
        relax: bool = False,
        fixed_vars: dict[int, int] = None
    ) -> dict:
        if not GUROBI_AVAILABLE:
            return {'status': 'error', 'message': 'Gurobi not installed'}

        result = {
            'status': 'error',
            'num_knights': None,
            'placement': [],
            'solve_time': 0.0,
            'obj_value': None,
            'x_values': [],
            'relaxed': relax,
        }

        n_sq = self.board.num_squares
        start = time.time()

        try:
           
            model = gp.Model(f"knights_{self.n}x{self.n}")
            model.setParam("OutputFlag", 0)

            if relax:
                x = model.addVars(
                    n_sq,
                    vtype=GRB.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                    name="x"
                )
            else:
                x = model.addVars(
                    n_sq,
                    vtype=GRB.BINARY,
                    name="x"
                )
            if fixed_vars:
                for var_idx, value in fixed_vars.items():

                    x[var_idx].lb = float(value)
                x[var_idx].ub = float(value)

            # Objective: minimize total knights placed
            model.setObjective(gp.quicksum(x[i]
                               for i in range(n_sq)), GRB.MINIMIZE)

            # Constraints: every square j must be threatened
            for j in range(n_sq):
                attackers_of_j = self.attack_matrix[j]

                if len(attackers_of_j) == 0:
                   
                    result['status'] = 'infeasible'
                    result['solve_time'] = time.time() - start
                    return result

                model.addConstr(
                    gp.quicksum(x[i] for i in attackers_of_j) >= 1,
                    name=f"cover_{j}"
                )

            model.optimize()

            result['solve_time'] = time.time() - start

            if model.Status == GRB.OPTIMAL:
                result['status'] = 'optimal'
                result['obj_value'] = model.ObjVal
                result['x_values'] = [
                    x[i].X for i in range(n_sq)
                ]
                result['num_knights'] = int(round(model.ObjVal))
                if not relax:
                    result['placement'] = [
                        i for i in range(n_sq)
                        if x[i].X > 0.5
                    ]
            elif model.Status == GRB.INFEASIBLE:
                result['status'] = 'infeasible'
            else:
                result['status'] = f"gurobi_status_{model.Status}"

        except gp.GurobiError as e:
            result['status'] = 'error'
            result['message'] = str(e)

        return result

    def solve_and_display(self) -> dict:
        """Solve and pretty-print the result."""
        print(f"\n{'='*50}")
        print(f"ILP Solver — {self.n}x{self.n} board")
        print(f"{'='*50}")

        result = self.solve(relax=False)

        if result['status'] == 'optimal':
            print(f"Status      : OPTIMAL")
            print(f"Knights     : {result['num_knights']}")
            print(f"Solve time  : {result['solve_time']:.4f}s")
            print(f"Placement   : {result['placement']}")
            coords = [self.board.square_coords(sq)
                      for sq in result['placement']]
            print(f"Coords      : {coords}")

            # Verify with board logic
            valid = self.board.is_valid_solution(result['placement'])
            print(f"Valid check : {'✓ PASSED' if valid else '✗ FAILED'}")

            # Display board
            self.board.display(result['placement'])

        elif result['status'] == 'infeasible':
            print(f"Status      : INFEASIBLE (no valid solution exists)")
            print(f"             (e.g. n=1: no knight can threaten the only square)")
        else:
            print(f"Status      : {result['status']}")
            if 'message' in result:
                print(f"Message     : {result['message']}")

        return result


def run_benchmark(sizes: list[int], verbose: bool = False):
   
    print(f"\n{'='*60}")
    print(f"ILP Benchmark")
    print(f"{'='*60}")
    print(f"{'n':>4} | {'knights':>8} | {'time (s)':>10} | {'status':>12}")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}")

    results = {}
    for n in sizes:
        solver = ILPSolver(n)
        r = solver.solve(verbose=verbose)
        results[n] = r

        knights_str = str(
            r['num_knights']) if r['num_knights'] is not None else "N/A"
        print(
            f"{n:>4} | {knights_str:>8} | {r['solve_time']:>10.4f} | {r['status']:>12}")

    print(f"{'='*60}\n")
    return results




if __name__ == "__main__":
    if not GUROBI_AVAILABLE:
        print("Gurobi not available yet — come back once your license is set up.")
        print("The board.py file works independently in the meantime.\n")
    else:
        # Single board solve with display
        solver = ILPSolver(n=5)
        solver.solve_and_display()

        # Benchmark across sizes
        run_benchmark(sizes=[3, 4, 5, 6, 7, 8])
