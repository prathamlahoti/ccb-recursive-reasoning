import unittest

import torch

from ccb.compute import compute_signature, trainable_parameters
from ccb.decoding import constrained_argmax, decode_state
from ccb.domains.alien_grid import AlienGridDomain
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.encoding import collate_episodes
from ccb.models import DirectTransformer, StateTransitionRecursiveModel


class DecodingTests(unittest.TestCase):
    def test_unique_grid_decoding(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=2, seed=0)])
        logits = torch.zeros(1, 2, 9, 9)
        logits[..., 0] = 10
        decoded = constrained_argmax(logits, batch.codec)
        for state in decoded.flatten(0, 1):
            self.assertEqual(len(set(state.tolist())), 9)
        restored = decode_state(decoded[0, 0], batch.codec)
        self.assertEqual(sorted(value for row in restored for value in row), list(range(1, 10)))

    def test_social_decoding_is_symmetric(self) -> None:
        domain = SocialLogicDomain(number_of_agents=4)
        batch = collate_episodes([domain.generate(depth=2, seed=0)])
        logits = torch.randn(1, 2, 16, 3)
        decoded = constrained_argmax(logits, batch.codec).reshape(1, 2, 4, 4)
        self.assertTrue(torch.equal(decoded, decoded.transpose(-1, -2)))
        self.assertTrue(torch.equal(torch.diagonal(decoded, dim1=-2, dim2=-1), torch.full((1, 2, 4), 2)))

    def test_d2_decoding_allows_official_duplicate_values(self) -> None:
        batch = collate_episodes([SymbolicPointersDomain().generate(depth=2, seed=0)])
        logits = torch.zeros(1, 2, 7, 10)
        logits[..., 3] = 10
        decoded = constrained_argmax(logits, batch.codec)
        self.assertTrue(torch.equal(decoded, torch.full((1, 2, 7), 3)))


class ComputeTests(unittest.TestCase):
    def test_parameter_and_block_accounting(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=5, seed=0)])
        transformer = DirectTransformer(batch.codec, width=16, heads=2, layers=2)
        strm = StateTransitionRecursiveModel(batch.codec, width=16, inner_loops=3)
        self.assertGreater(trainable_parameters(transformer), 0)
        self.assertEqual(compute_signature(transformer, depth=5)["block_evaluations"], 2)
        self.assertEqual(compute_signature(strm, depth=5)["block_evaluations"], 20)


if __name__ == "__main__":
    unittest.main()
