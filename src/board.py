class Board:
    # All 8 possible knight move offsets
    KNIGHT_MOVES = [
        (-2, -1), (-2, +1),
        (-1, -2), (-1, +2),
        (+1, -2), (+1, +2),
        (+2, -1), (+2, +1),
    ]

    def __init__(self, n: int):
        """
        Args:
            n: board size (n x n)
        """
        if n < 1:
            raise ValueError("Board size n must be at least 1.")
        self.n = n
        self.num_squares = n * n

        # Precompute for every square: which squares can ATTACK it
        # attack_map[sq] = list of squares whose knight can reach sq
        self._attack_map = self._build_attack_map()


    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def _sq(self, r: int, c: int) -> int:
        """Convert (row, col) to a flat square index."""
        return r * self.n + c

    def _rc(self, sq: int) -> tuple[int, int]:
        """Convert flat square index to (row, col)."""
        return divmod(sq, self.n)

    def _build_attack_map(self) -> dict[int, list[int]]:
        """
        For every square sq, compute which other squares can attack it
        with a knight move.

        attack_map[sq] = [sq1, sq2, ...] where a knight on sq_i
                         threatens sq.
        """
        attack_map = {sq: [] for sq in range(self.num_squares)}

        for sq in range(self.num_squares):
            r, c = self._rc(sq)
            for dr, dc in self.KNIGHT_MOVES:
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    attacker_sq = self._sq(nr, nc)
                    attack_map[sq].append(attacker_sq)

        return attack_map



    def attackers(self, sq: int = None, row: int = None, col: int = None) -> list[int]:

        if sq is None:
            if row is None or col is None:
                raise ValueError("Provide either sq or both row and col.")
            sq = self._sq(row, col)
        return self._attack_map[sq]

    def attacks_from(self, sq: int = None, row: int = None, col: int = None) -> list[int]:

        return self.attackers(sq=sq, row=row, col=col)

    def get_attack_matrix(self) -> list[list[int]]:

        return [self._attack_map[sq] for sq in range(self.num_squares)]

    def square_index(self, row: int, col: int) -> int:
        """Convert (row, col) to flat index. Public convenience method."""
        if not self._in_bounds(row, col):
            raise ValueError(f"({row},{col}) is out of bounds for {self.n}x{self.n} board.")
        return self._sq(row, col)

    def square_coords(self, sq: int) -> tuple[int, int]:
        """Convert flat index to (row, col). Public convenience method."""
        if not (0 <= sq < self.num_squares):
            raise ValueError(f"Square index {sq} out of range for {self.n}x{self.n} board.")
        return self._rc(sq)

    def display(self, placement: list[int] = None):

        occupied = set(placement) if placement else set()

        # Compute which squares are threatened by the placement
        threatened = set()
        for sq in occupied:
            for attacked_sq in self.attacks_from(sq):
                threatened.add(attacked_sq)

        print(f"\n{self.n}x{self.n} Board  (K=knight, *=threatened, .=uncovered)\n")
        for r in range(self.n):
            row_str = ""
            for c in range(self.n):
                sq = self._sq(r, c)
                if sq in occupied:
                    row_str += " K "
                elif sq in threatened:
                    row_str += " * "
                else:
                    row_str += " . "
            print(row_str)
        print()

        if placement is not None:
            # Count uncovered squares (must be 0 for a valid solution)
            uncovered = [
                sq for sq in range(self.num_squares)
                if sq not in threatened
            ]
            print(f"Knights placed : {len(occupied)}")
            print(f"Threatened     : {len(threatened)}")
            print(f"Uncovered      : {len(uncovered)}")
            if uncovered:
                coords = [self._rc(sq) for sq in uncovered]
                print(f"Uncovered squares: {coords}")
            print()

    def is_valid_solution(self, placement: list[int]) -> bool:
        
        occupied = set(placement)
        threatened = set()
        for sq in occupied:
            for attacked_sq in self.attacks_from(sq):
                threatened.add(attacked_sq)

        # Every square must be threatened
        return all(sq in threatened for sq in range(self.num_squares))



if __name__ == "__main__":
    print("=" * 50)
    print("Board smoke test")
    print("=" * 50)

    # Test 1: Knight move counts on a 5x5 board
    b5 = Board(5)
    center_attackers = b5.attackers(row=2, col=2)
    corner_attackers = b5.attackers(row=0, col=0)
    print(f"5x5 board:")
    print(f"  Squares that can attack (2,2): {len(center_attackers)} -> {center_attackers}")
    print(f"  Squares that can attack (0,0): {len(corner_attackers)} -> {corner_attackers}")
    assert len(center_attackers) == 8, "Center of 5x5 should have 8 attackers"
    assert len(corner_attackers) == 2, "Corner of 5x5 should have 2 attackers"

    # Test 2: 1x1 board — impossible to threaten the single square
    b1 = Board(1)
    assert not b1.is_valid_solution([0]), "1x1: square 0 cannot threaten itself"
    assert not b1.is_valid_solution([]),  "1x1: empty placement is invalid"
    print(f"\n1x1 board: correctly identified as unsolvable")

    # Test 3: Display a small placement on a 5x5 board
    # Known good placement for 5x5 (just for display purposes)
    test_placement = [2, 4, 10, 14, 20, 22]  # example, may not be optimal
    b5.display(test_placement)

    # Test 4: attack matrix shape
    matrix = b5.get_attack_matrix()
    assert len(matrix) == 25, "5x5 attack matrix should have 25 rows"
    print(f"Attack matrix for 5x5: {len(matrix)} rows (one per square) ✓")

    print("\nAll smoke tests passed ✓")