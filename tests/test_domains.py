import unittest

from ccb.domains.alien_grid import (
    AlienGridDomain,
    GridOperation,
    apply_grid_operation,
)
from ccb.domains.social_logic import (
    Relation,
    SocialLogicDomain,
    SocialOperation,
    signed_closure,
)
from ccb.domains.symbolic_pointers import (
    PointerOperation,
    SymbolicPointersDomain,
    apply_pointer_operation,
)


class AlienGridTests(unittest.TestCase):
    def test_paper_rotation_example(self) -> None:
        initial = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        expected = ((7, 4, 1), (8, 5, 2), (9, 6, 3))
        self.assertEqual(apply_grid_operation(initial, GridOperation.ROTATE_90_CW), expected)

    def test_shift_middle_row_left(self) -> None:
        initial = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        expected = ((1, 2, 3), (5, 6, 4), (7, 8, 9))
        self.assertEqual(apply_grid_operation(initial, GridOperation.SHIFT_ROW_2_LEFT), expected)

    def test_all_official_operations_preserve_entities(self) -> None:
        initial = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        for operation in GridOperation:
            state = apply_grid_operation(initial, operation)
            self.assertEqual(set(value for row in state for value in row), set(range(1, 10)))

    def test_generation_is_reproducible_and_preserves_entities(self) -> None:
        domain = AlienGridDomain()
        left = domain.generate(depth=50, seed=71)
        right = domain.generate(depth=50, seed=71)
        self.assertEqual(left, right)
        self.assertEqual(left.depth, 50)
        for state in left.trace:
            self.assertEqual(set(value for row in state for value in row), set(range(1, 10)))


class SymbolicPointersTests(unittest.TestCase):
    def test_shift_right(self) -> None:
        state = (0, 1, 2, 3, 4, 5, 6)
        self.assertEqual(
            apply_pointer_operation(state, PointerOperation.SHIFT_RIGHT),
            (6, 0, 1, 2, 3, 4, 5),
        )

    def test_generation_preserves_digit_range_but_allows_duplicates(self) -> None:
        episode = SymbolicPointersDomain().generate(depth=100, seed=3)
        for state in episode.trace:
            self.assertTrue(all(0 <= value <= 9 for value in state))
        self.assertTrue(any(len(set(state)) < 7 for state in episode.trace))

    def test_generation_is_reproducible(self) -> None:
        domain = SymbolicPointersDomain()
        self.assertEqual(domain.generate(depth=25, seed=9), domain.generate(depth=25, seed=9))


class SocialLogicTests(unittest.TestCase):
    def test_signed_transitive_closure(self) -> None:
        edges = (
            SocialOperation(0, 1, Relation.ALLY),
            SocialOperation(1, 2, Relation.RIVAL),
            SocialOperation(2, 3, Relation.RIVAL),
        )
        state = signed_closure(4, edges)
        self.assertEqual(state[0][1], Relation.ALLY)
        self.assertEqual(state[0][2], Relation.RIVAL)
        self.assertEqual(state[0][3], Relation.ALLY)
        self.assertEqual(state[3][0], Relation.ALLY)

    def test_contradiction_is_rejected(self) -> None:
        edges = (
            SocialOperation(0, 1, Relation.ALLY),
            SocialOperation(1, 2, Relation.ALLY),
            SocialOperation(0, 2, Relation.RIVAL),
        )
        with self.assertRaises(ValueError):
            signed_closure(3, edges)

    def test_generation_is_reproducible_and_symmetric(self) -> None:
        domain = SocialLogicDomain(number_of_agents=10)
        episode = domain.generate(depth=50, seed=44)
        self.assertEqual(episode, domain.generate(depth=50, seed=44))
        for state in episode.trace:
            for row in range(10):
                self.assertEqual(state[row][row], Relation.ALLY)
                for column in range(10):
                    self.assertEqual(state[row][column], state[column][row])


if __name__ == "__main__":
    unittest.main()
