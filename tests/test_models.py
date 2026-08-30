import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.models import (
    CCBTokenTransformer,
    DirectTransformer,
    OfficialTRMCCBAdapter,
    PublishedTRMCCB,
)


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

    def test_act_state_retains_active_rows_and_resets_halted_rows(self) -> None:
        model = PublishedTRMCCB(
            self.batch.codec,
            width=16,
            heads=2,
            layers=1,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=3,
            halt_exploration_prob=0.0,
        )
        model.train()
        carry = model.initial_act_carry(self.batch)
        carry, _, _ = model.act_step(carry, self.batch)
        self.assertTrue(torch.equal(carry.steps, torch.ones_like(carry.steps)))
        carry = replace(carry, halted=torch.tensor([True, False]))
        updated_initial = self.batch.initial_state.clone()
        updated_initial[0] = torch.roll(updated_initial[0], 1)
        incoming = replace(self.batch, initial_state=updated_initial)
        carry, _, _ = model.act_step(carry, incoming)
        self.assertTrue(torch.equal(carry.current_batch.initial_state[0], updated_initial[0]))
        self.assertTrue(torch.equal(carry.current_batch.initial_state[1], self.batch.initial_state[1]))
        self.assertEqual(carry.steps.tolist(), [1, 2])

    def test_act_resets_after_a_short_final_batch(self) -> None:
        """A 64-row carry must not leak into a smaller final DataLoader batch."""

        model = PublishedTRMCCB(
            self.batch.codec,
            width=16,
            heads=2,
            layers=1,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=3,
            halt_exploration_prob=0.0,
        )
        model.train()
        carry = model.initial_act_carry(self.batch)
        carry = replace(carry, halted=torch.ones(2, dtype=torch.bool))
        short_batch = collate_episodes([AlienGridDomain().generate(depth=5, seed=99)])
        carry, output, (q_halt, q_continue) = model.act_step(carry, short_batch)
        self.assertEqual(carry.steps.tolist(), [1])
        self.assertEqual(tuple(carry.halted.shape), (1,))
        self.assertEqual(tuple(output.logits.shape), (1, 5, 9, 9))
        self.assertEqual(tuple(q_halt.shape), (1,))
        self.assertEqual(tuple(q_continue.shape), (1,))

    def test_official_core_ccb_adapter_is_target_free_and_shapes_outputs(self) -> None:
        model = OfficialTRMCCBAdapter(
            self.batch.codec,
            max_depth=5,
            hidden_size=32,
            num_heads=4,
            l_layers=1,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=1,
        )
        altered = replace(self.batch, targets=(self.batch.targets + 1) % 9)
        self.assertTrue(torch.equal(model.input_tokens(self.batch), model.input_tokens(altered)))
        model.eval()
        with torch.no_grad():
            output = model(self.batch)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 9, 9))

    def test_official_ccb_vocabulary_namespaces_and_padding_are_disjoint(self) -> None:
        mixed = collate_episodes(
            [
                AlienGridDomain().generate(depth=2, seed=20),
                AlienGridDomain().generate(depth=5, seed=21),
            ]
        )
        model = OfficialTRMCCBAdapter(
            mixed.codec,
            max_depth=5,
            hidden_size=32,
            num_heads=4,
            l_layers=1,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=1,
        )
        tokens = model.input_tokens(mixed)
        layout = model.layout
        special = {
            layout.bos_token,
            layout.ops_token,
            layout.output_token,
            layout.mask_token,
            layout.pad_token,
        }
        self.assertEqual(len(special), 5)
        self.assertTrue(
            all(
                item >= mixed.codec.state_vocab_size + mixed.codec.operation_vocab_size
                for item in special
            )
        )
        self.assertTrue(torch.all(tokens[:, layout.state_tokens] < mixed.codec.state_vocab_size))
        valid_operations = tokens[1, layout.operation_tokens]
        self.assertTrue(torch.all(valid_operations >= layout.operation_offset))
        self.assertTrue(
            torch.all(valid_operations < layout.operation_offset + mixed.codec.operation_vocab_size)
        )
        self.assertTrue(torch.all(tokens[0, layout.operation_tokens][2:] == layout.pad_token))
        first_queries = tokens[0, layout.query_tokens].reshape(5, mixed.codec.state_size)
        self.assertTrue(torch.all(first_queries[:2] == layout.mask_token))
        self.assertTrue(torch.all(first_queries[2:] == layout.pad_token))

    def test_token_transformer_uses_identical_target_free_serialization(self) -> None:
        trm = OfficialTRMCCBAdapter(
            self.batch.codec,
            max_depth=5,
            hidden_size=32,
            num_heads=4,
            l_layers=1,
            h_cycles=1,
            l_cycles=1,
            halt_max_steps=1,
        )
        transformer = CCBTokenTransformer(
            self.batch.codec, max_depth=5, width=32, heads=4, layers=1
        )
        altered = replace(self.batch, targets=(self.batch.targets + 1) % 9)
        self.assertTrue(
            torch.equal(trm.input_tokens(self.batch), transformer.input_tokens(self.batch))
        )
        self.assertTrue(
            torch.equal(transformer.input_tokens(self.batch), transformer.input_tokens(altered))
        )
        output = transformer(self.batch)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 9, 9))


if __name__ == "__main__":
    unittest.main()
