import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.models.common import ModelOutput
from ccb.models.grid_executor import LearnedGridExecutor
from ccb.prefix import correct_prefix, prefix_weights, refinement_metrics
from ccb.training import (
    TrainConfig, ExponentialMovingAverage, build_model, build_optimizer,
    load_checkpoint, save_checkpoint, seed_everything, train_trm_act_batches,
    trm_sequence_loss,
)


class PrefixTests(unittest.TestCase):
    def test_prefix_stops_at_first_error_and_ignores_padding(self):
        targets = torch.zeros(2, 4, 1, dtype=torch.long)
        predictions = torch.tensor([[[0], [0], [1], [0]], [[0], [0], [1], [1]]])
        mask = torch.tensor([[True]*4, [True, True, False, False]])
        self.assertEqual(correct_prefix(predictions, targets, mask).sum(1).tolist(), [2, 2])
        weights = prefix_weights(predictions, targets, mask, frontier_weight=2)
        self.assertEqual(weights.tolist(), [[1, 1, 3, 1], [1, 1, 0, 0]])

    def test_preservation_and_regression_exclude_replaced_act_rows(self):
        targets = torch.zeros(2, 3, 1, dtype=torch.long)
        previous = targets.clone()
        predictions = targets.clone()
        predictions[:, 1] = 1
        mask = torch.ones(2, 3, dtype=torch.bool)
        continuing = torch.tensor([True, False])
        weights = prefix_weights(predictions, targets, mask, preservation_weight=2,
                                 previous_predictions=previous, continuing_rows=continuing)
        self.assertEqual(weights.tolist(), [[3, 3, 3], [1, 1, 1]])
        report = refinement_metrics(predictions, targets, mask,
                                    previous_predictions=previous, continuing_rows=continuing)
        self.assertEqual(report['previous_prefix_states'], 3)
        self.assertEqual(report['regressed_prefix_states'], 1)
        self.assertEqual(report['prefix_change_sum'], -2)

    def test_loss_uses_labels_only_in_loss_and_zero_weights_preserve_default(self):
        batch = collate_episodes([AlienGridDomain().generate(depth=3, seed=51)])
        logits = torch.randn(1, 3, 9, 9, requires_grad=True)
        q = torch.zeros(1, requires_grad=True)
        default = trm_sequence_loss(ModelOutput(logits), q, batch)[0]
        zero = trm_sequence_loss(ModelOutput(logits), q, batch, frontier_weight=0)[0]
        self.assertTrue(torch.equal(default, zero))
        weighted = trm_sequence_loss(ModelOutput(logits), q, batch, frontier_weight=2)[0]
        weighted.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_prefix_act_checkpoint_resume_preserves_previous_prediction_and_weights(self):
        batch = collate_episodes([AlienGridDomain().generate(depth=3, seed=s) for s in (31,32)])
        config = TrainConfig(model='official_trm_ccb', width=16, layers_or_loops=1,
                             trm_h_cycles=1, trm_l_cycles=1, trm_max_depth=3,
                             trm_halt_max_steps=3, trm_halt_exploration_prob=0,
                             steps=4, seed=7, ema_decay=.9,
                             prefix_frontier_weight=1, prefix_preservation_weight=1)
        seed_everything(7)
        full = build_model(config, batch.codec)
        _, _, _, full_history = train_trm_act_batches(full, [batch]*6, config)
        self.assertEqual([h['accepted_examples'] for h in full_history], [2,0,0,2])
        seed_everything(7)
        partial = build_model(config, batch.codec)
        opt, ema, state, _ = train_trm_act_batches(partial, [batch]*6, replace(config, steps=2))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'checkpoint.pt'
            save_checkpoint(path, model=partial, optimizer=opt, config=config,
                            codec=batch.codec, step=2, ema=ema, act_carry=state)
            resumed = build_model(config, batch.codec)
            opt2 = build_optimizer(resumed, config)
            ema2 = ExponentialMovingAverage(resumed, config.ema_decay)
            start, payload = load_checkpoint(path, model=resumed, optimizer=opt2,
                                             ema=ema2, return_payload=True)
            self.assertTrue(torch.equal(state[0].previous_predictions,
                                        payload['act_carry'][0].previous_predictions))
            train_trm_act_batches(resumed, [batch]*6, config, optimizer=opt2, ema=ema2,
                                 act_state=payload['act_carry'], start_step=start)
        for a,b in zip(full.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_negative_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            prefix_weights(torch.zeros(1,1,1), torch.zeros(1,1,1),
                           torch.ones(1,1,dtype=torch.bool), frontier_weight=-1)

    def test_learned_executor_rolls_out_predictions_and_ignores_target_changes(self):
        seed_everything(3)
        domain = AlienGridDomain(randomize_initial_state=True)
        batch = collate_episodes([domain.generate(depth=1,seed=s) for s in range(64)])
        model = LearnedGridExecutor()
        labels = (batch.targets[:,0,:,None] == batch.initial_state[:,None,:]).long().argmax(-1)
        optimizer = torch.optim.Adam(model.parameters(),lr=.1)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model.source_logits(batch.operations[:,0]).reshape(-1,9),labels.reshape(-1))
            loss.backward()
            optimizer.step()
        test = collate_episodes([domain.generate(depth=50,seed=100+s) for s in range(10)])
        output = model(test).logits.argmax(-1)
        self.assertTrue(torch.equal(output,test.targets))
        corrupted = replace(test,targets=torch.zeros_like(test.targets))
        self.assertTrue(torch.equal(output,model(corrupted).logits.argmax(-1)))
