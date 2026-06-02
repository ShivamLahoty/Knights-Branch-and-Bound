import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from board import Board


def test_board_creation():
    b = Board(5)

    assert b.n == 5
    assert b.num_squares == 25

    print("Board creation test passed")


def test_knight_moves():
    b = Board(5)

    attacks = b.attacks_from(12)

    assert isinstance(attacks, list)
    assert len(attacks) > 0

    print("Knight move generation test passed")


if __name__ == "__main__":
    test_board_creation()
    test_knight_moves()

    print("\nAll board tests passed!")