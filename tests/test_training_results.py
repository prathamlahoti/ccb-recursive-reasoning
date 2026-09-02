import tempfile
import unittest
from pathlib import Path

import torch

from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.results import markdown_table, write_result
from ccb.training import (
    TrainConfig,
    build_optimizer,
    build_model,
    evaluate_batch,
    ExponentialMovingAverage,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_batches,
    train_trm_act_batches,
    train_fixed_batch,
)


class TrainingTests(unittest.TestCase):
    def test_tiny_batch_loss_decreases_and_checkpoint_loads(self) -> None:
        seed_everything(0)
        domain = AlienGridDomain()
        batch = collate_episodes([domain.generate(depth=2, seed=seed) for seed in range(4)])
        config = TrainConfig(
            model="transformer", width=16, layers_or_loops=1, learning_rate=0.01, steps=20
        )
        model = build_model(config, batch.codec)
        optimizer, history = train_fixed_batch(model, batch, config)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        metrics = evaluate_batch(model, batch)
        self.assertGreaterEqual(metrics["element_accuracy"], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                config=config,
                codec=batch.codec,
                step=config.steps,
            )
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["schema"], "ccb_checkpoint_v2")
            self.assertEqual(payload["step"], 20)

    def test_resumed_batches_match_uninterrupted_training(self) -> None:
        domain = AlienGridDomain()
        batch = collate_episodes([domain.generate(depth=2, seed=seed) for seed in range(4)])
        full_config = TrainConfig(
            model="transformer", width=16, layers_or_loops=1, learning_rate=0.01, steps=6, seed=7
        )
        partial_config = TrainConfig(
            model="transformer", width=16, layers_or_loops=1, learning_rate=0.01, steps=3, seed=7
        )

        seed_everything(7)
        uninterrupted = build_model(full_config, batch.codec)
        train_batches(uninterrupted, [batch] * 6, full_config)

        seed_everything(7)
        interrupted = build_model(partial_config, batch.codec)
        optimizer, _ = train_batches(interrupted, [batch] * 3, partial_config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=interrupted,
                optimizer=optimizer,
                config=partial_config,
                codec=batch.codec,
                step=3,
            )
            resumed = build_model(full_config, batch.codec)
            resumed_optimizer = build_optimizer(resumed, full_config)
            self.assertEqual(
                load_checkpoint(path, model=resumed, optimizer=resumed_optimizer), 3
            )
            train_batches(
                resumed,
                [batch] * 6,
                full_config,
                optimizer=resumed_optimizer,
                start_step=3,
            )
        for left, right in zip(uninterrupted.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_ema_is_a_copied_evaluation_model(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=5, seed=0)])
        config = TrainConfig(model="trm_upstream_core", width=16, layers_or_loops=1)
        model = build_model(config, batch.codec)
        ema = ExponentialMovingAverage(model, 0.9)
        original = [item.detach().clone() for item in model.parameters()]
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        ema.update(model)
        self.assertTrue(all(torch.equal(old, now) for old, now in zip(original[1:], list(model.parameters())[1:])))
        self.assertFalse(any(parameter.requires_grad for parameter in ema.evaluation_model.parameters()))

    def test_generic_training_updates_supplied_ema(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=2, seed=0)])
        config = TrainConfig(
            model="transformer", width=16, layers_or_loops=1,
            learning_rate=0.01, steps=2,
        )
        model = build_model(config, batch.codec)
        ema = ExponentialMovingAverage(model, 0.9)
        before = [value.detach().clone() for value in ema.evaluation_model.parameters()]
        train_batches(model, [batch] * 2, config, ema=ema)
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(before, ema.evaluation_model.parameters())
            )
        )

    def test_ema_rejects_cross_device_updates_before_tensor_math(self) -> None:
        model = torch.nn.Linear(2, 2)
        ema = ExponentialMovingAverage(model, 0.9)
        model.to("meta")
        with self.assertRaisesRegex(RuntimeError, "move the live model before constructing"):
            ema.update(model)

    def test_act_resume_matches_uninterrupted_training(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=5, seed=seed) for seed in range(2)])
        full = TrainConfig(
            model="trm_upstream_core", width=16, layers_or_loops=1,
            learning_rate=0.003, steps=4, seed=13, trm_h_cycles=1,
            trm_l_cycles=1, trm_halt_max_steps=3, trm_halt_exploration_prob=0.0,
            ema_decay=0.9,
        )
        partial = TrainConfig(**{**full.__dict__, "steps": 2})
        seed_everything(13)
        uninterrupted = build_model(full, batch.codec)
        _, full_ema, _, _ = train_trm_act_batches(uninterrupted, [batch] * 6, full)
        seed_everything(13)
        interrupted = build_model(partial, batch.codec)
        checkpoint_state = {}
        def save_state(step, model, optimizer, ema, carry, pending):
            checkpoint_state.update(step=step, optimizer=optimizer, ema=ema, act_state=(carry, pending))
        _, _, _, _ = train_trm_act_batches(
            interrupted, [batch] * 6, partial, checkpoint_callback=save_state
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "act.pt"
            save_checkpoint(path, model=interrupted, optimizer=checkpoint_state["optimizer"],
                            config=partial, codec=batch.codec, step=checkpoint_state["step"],
                            ema=checkpoint_state["ema"], act_carry=checkpoint_state["act_state"])
            resumed = build_model(full, batch.codec)
            resumed_optimizer = build_optimizer(resumed, full)
            resumed_ema = ExponentialMovingAverage(resumed, full.ema_decay)
            start, payload = load_checkpoint(path, model=resumed, optimizer=resumed_optimizer,
                                             ema=resumed_ema, return_payload=True)
            _, final_ema, _, _ = train_trm_act_batches(
                resumed, [batch] * 6, full, optimizer=resumed_optimizer, ema=resumed_ema,
                act_state=payload["act_carry"], start_step=start,
            )
        for left, right in zip(uninterrupted.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(full_ema.evaluation_model.parameters(), final_ema.evaluation_model.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_official_trm_adapter_runs_through_act_and_adam_atan2(self) -> None:
        batch = collate_episodes([AlienGridDomain().generate(depth=5, seed=seed) for seed in range(2)])
        config = TrainConfig(
            model="official_trm_ccb", width=32, layers_or_loops=1, steps=3,
            learning_rate=1e-3, weight_decay=0.1, optimizer="adam_atan2",
            optimizer_betas=(0.9, 0.95), trm_h_cycles=1, trm_l_cycles=1,
            trm_max_depth=5, trm_halt_max_steps=1, trm_halt_exploration_prob=0.0,
        )
        model = build_model(config, batch.codec)
        _, ema, carry, history = train_trm_act_batches(model, [batch] * 4, config)
        self.assertEqual(len(history), 3)
        self.assertTrue(torch.equal(carry[0].current_batch.targets, batch.targets))
        self.assertFalse(any(parameter.requires_grad for parameter in ema.evaluation_model.parameters()))

class ResultTests(unittest.TestCase):
    def test_result_artifacts(self) -> None:
        table = markdown_table([{"model": "oracle", "accuracy": 1.0}], ("model", "accuracy"))
        self.assertIn("| oracle | 1.0 |", table)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_result(path, {"accuracy": 1.0})
            self.assertIn('"schema": "ccb_result_v1"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
