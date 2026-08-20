import unittest

import torch
from torch.nn import functional as F

from ccb.domains.alien_grid import AlienGridDomain
from ccb.domains.social_logic import SocialLogicDomain
from ccb.domains.symbolic_pointers import SymbolicPointersDomain
from ccb.encoding import collate_episodes
from ccb.models import (
    DirectTransformer,
    FastSlowRecurrentModel,
    LoopedTransformer,
    RecurrentBaseline,
    SocialMessagePassingGNN,
    StateTransitionRecursiveModel,
    VanillaTRM,
)


class EncodingTests(unittest.TestCase):
    def test_all_domains_collate(self) -> None:
        for domain, state_size in (
            (AlienGridDomain(), 9),
            (SymbolicPointersDomain(), 7),
            (SocialLogicDomain(number_of_agents=5), 25),
        ):
            episodes = [domain.generate(depth=3, seed=seed) for seed in range(2)]
            batch = collate_episodes(episodes)
            self.assertEqual(tuple(batch.initial_state.shape), (2, state_size))
            self.assertEqual(tuple(batch.operations.shape), (2, 3))
            self.assertEqual(tuple(batch.targets.shape), (2, 3, state_size))

    def test_mixed_depth_is_padded_and_masked(self) -> None:
        domain = AlienGridDomain()
        batch = collate_episodes(
            [domain.generate(depth=2, seed=0), domain.generate(depth=3, seed=1)]
        )
        self.assertEqual(tuple(batch.operations.shape), (2, 3))
        self.assertEqual(
            batch.step_mask.tolist(), [[True, True, False], [True, True, True]]
        )
        self.assertEqual(batch.depths.tolist(), [2, 3])


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        domain = AlienGridDomain()
        self.batch = collate_episodes(
            [domain.generate(depth=4, seed=seed) for seed in range(2)]
        )

    def test_forward_backward_contract_for_all_sequence_models(self) -> None:
        models = (
            DirectTransformer(self.batch.codec, width=16, heads=2, layers=1),
            RecurrentBaseline(self.batch.codec, width=16, layers=1, cell="gru"),
            RecurrentBaseline(self.batch.codec, width=16, layers=1, cell="lstm"),
            LoopedTransformer(self.batch.codec, width=16, heads=2, loops=2),
            VanillaTRM(self.batch.codec, width=16, loops=2),
            FastSlowRecurrentModel(self.batch.codec, width=16, fast_loops=2),
            StateTransitionRecursiveModel(self.batch.codec, width=16, inner_loops=2),
        )
        for model in models:
            model.zero_grad(set_to_none=True)
            output = model(self.batch)
            self.assertEqual(tuple(output.logits.shape), (2, 4, 9, 9))
            loss = F.cross_entropy(output.logits.flatten(0, 2), self.batch.targets.flatten())
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_recursive_models_export_each_loop(self) -> None:
        for model in (
            LoopedTransformer(self.batch.codec, width=16, heads=2, loops=3),
            VanillaTRM(self.batch.codec, width=16, loops=3),
            FastSlowRecurrentModel(self.batch.codec, width=16, fast_loops=3),
            StateTransitionRecursiveModel(self.batch.codec, width=16, inner_loops=3),
        ):
            output = model(self.batch)
            self.assertEqual(len(output.loop_logits), 3)
            self.assertTrue(all(item.shape == output.logits.shape for item in output.loop_logits))

    def test_social_gnn_forward_backward(self) -> None:
        domain = SocialLogicDomain(number_of_agents=5)
        batch = collate_episodes([domain.generate(depth=4, seed=seed) for seed in range(2)])
        model = SocialMessagePassingGNN(batch.codec, width=16, message_steps=2)
        output = model(batch)
        self.assertEqual(tuple(output.logits.shape), (2, 4, 25, 3))
        loss = F.cross_entropy(output.logits.flatten(0, 2), batch.targets.flatten())
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
