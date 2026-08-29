import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.models import DirectTransformer, PublishedTRMCCB


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = collate_episodes(
            [AlienGridDomain().generate(depth=5, seed=seed) for seed in range(2)]
        )

    def test_forward_backward_contract(self) -> None:
        models = (
            DirectTransformer(self.batch.codec, width=16, heads=2, layers=1),
            PublishedTRMCCB(
                self.batch.codec, width=16, heads=2, layers=1, h_cycles=1, l_cycles=1
            ),
        )
        for model in models:
            output = model(self.batch)
            self.assertEqual(tuple(output.logits.shape), (2, 5, 9, 9))
            F.cross_entropy(output.logits.flatten(0, 2), self.batch.targets.flatten()).backward()
            self.assertTrue(any(item.grad is not None for item in model.parameters()))

    def test_targets_are_not_model_inputs(self) -> None:
        altered = replace(self.batch, targets=(self.batch.targets + 1) % 9)
        for model in (
            DirectTransformer(self.batch.codec, width=16, heads=2, layers=1),
            PublishedTRMCCB(
                self.batch.codec, width=16, heads=2, layers=1, h_cycles=1, l_cycles=1
            ),
        ):
            model.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(model(self.batch).logits, model(altered).logits))

    def test_trm_core_uses_fixed_detached_carry(self) -> None:
        model = PublishedTRMCCB(
            self.batch.codec, width=16, heads=2, layers=1, h_cycles=2, l_cycles=1
        )
        self.assertFalse(model.h_init.requires_grad)
        carry, output, _ = model.refine(self.batch, model.initial_carry(self.batch))
        self.assertFalse(carry.z_h.requires_grad)
        self.assertFalse(carry.z_l.requires_grad)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 9, 9))


if __name__ == "__main__":
    unittest.main()
