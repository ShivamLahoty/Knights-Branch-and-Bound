import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from board import Board
from ilp_solver import ILPSolver
from bnb import BnBSolver


def test_board_creation():

    b = Board(5)

    assert b.n == 5
    assert b.num_squares == 25

    print("✓ Board creation test passed")


def test_knight_moves():

    b = Board(5)

    center_attackers = b.attackers(12)

    assert len(center_attackers) == 8

    print("✓ Knight move generation test passed")


def test_ilp_solution_validity():

    solver = ILPSolver(5)

    result = solver.solve()

    assert result['status'] == 'optimal'

    board = Board(5)

    assert board.is_valid_solution(result['placement'])

    print("✓ ILP solution validity test passed")


def test_bnb_matches_ilp():

    ilp_solver = ILPSolver(5)
    ilp_result = ilp_solver.solve()

    bnb_solver = BnBSolver(
        n=5,
        strategy='best_first',
        branch_var='most_constrained'
    )

    bnb_result = bnb_solver.solve()

    assert ilp_result['num_knights'] == bnb_result['num_knights']

    print("✓ BnB matches ILP optimum")


def test_lp_relaxation_bound():

    lp_solver = ILPSolver(5)

    lp_result = lp_solver.solve(relax=True)

    ilp_solver = ILPSolver(5)

    ilp_result = ilp_solver.solve()

    assert lp_result['obj_value'] <= ilp_result['num_knights']

    print("✓ LP relaxation lower bound test passed")


if __name__ == "__main__":

    print("\nRunning project tests...\n")

    test_board_creation()

    test_knight_moves()

    test_ilp_solution_validity()

    test_bnb_matches_ilp()

    test_lp_relaxation_bound()

    print("\nALL TESTS PASSED ✓")