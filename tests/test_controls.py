import unittest

import torch

from ccb.controls import corrupt_initial_states, permute_operation_order
from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes


class CausalityControlTests(unittest.TestCase):
    def setUp(self) -> None:
        domain = AlienGridDomain()
        self.batch = collate_episodes(
            [domain.generate(depth=4, seed=seed) for seed in range(4)]
        )

    def test_operation_permutation_preserves_tokens_and_targets(self) -> None:
        transformed = permute_operation_order(self.batch, seed=19)
        self.assertTrue(torch.equal(transformed.targets, self.batch.targets))
        self.assertTrue(torch.equal(transformed.step_mask, self.batch.step_mask))
        for row, depth in enumerate(self.batch.depths.tolist()):
            self.assertTrue(
                torch.equal(
                    transformed.operations[row, :depth].sort().values,
                    self.batch.operations[row, :depth].sort().values,
                )
            )

    def test_initial_state_control_preserves_program_and_targets(self) -> None:
        transformed = corrupt_initial_states(self.batch)
        self.assertTrue(torch.equal(transformed.operations, self.batch.operations))
        self.assertTrue(torch.equal(transformed.targets, self.batch.targets))
        self.assertFalse(torch.equal(transformed.initial_state, self.batch.initial_state))


if __name__ == "__main__":
    unittest.main()
