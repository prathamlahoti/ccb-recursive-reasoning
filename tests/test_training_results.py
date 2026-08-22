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
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_batches,
    train_faithful_trm_batches,
    train_fixed_batch,
)


class TrainingTests(unittest.TestCase):
    def test_tiny_batch_loss_decreases_and_checkpoint_loads(self) -> None:
        seed_everything(0)
        domain = AlienGridDomain()
        batch = collate_episodes([domain.generate(depth=2, seed=seed) for seed in range(4)])
        config = TrainConfig(
            model="gru", width=16, layers_or_loops=1, learning_rate=0.01, steps=20
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
            self.assertEqual(payload["schema"], "ccb_checkpoint_v1")
            self.assertEqual(payload["step"], 20)

    def test_resumed_batches_match_uninterrupted_training(self) -> None:
        domain = AlienGridDomain()
        batch = collate_episodes([domain.generate(depth=2, seed=seed) for seed in range(4)])
        full_config = TrainConfig(
            model="gru", width=16, layers_or_loops=1, learning_rate=0.01, steps=6, seed=7
        )
        partial_config = TrainConfig(
            model="gru", width=16, layers_or_loops=1, learning_rate=0.01, steps=3, seed=7
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

    def test_faithful_trm_deep_supervision_and_ema(self) -> None:
        seed_everything(3)
        batch = collate_episodes(
            [AlienGridDomain().generate(depth=2, seed=seed) for seed in range(4)]
        )
        config = TrainConfig(
            model="trm_faithful",
            width=16,
            layers_or_loops=1,
            learning_rate=0.01,
            steps=4,
            trm_latent_steps=1,
            trm_refinement_steps=2,
            trm_supervision_steps=2,
            ema_decay=0.9,
        )
        model = build_model(config, batch.codec)
        optimizer, history = train_faithful_trm_batches(model, [batch] * 4, config)
        self.assertEqual(len(history), 4)
        self.assertTrue(all("halt_loss" in item for item in history))
        self.assertTrue(torch.isfinite(torch.tensor(history[-1]["loss"])))
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertTrue(
            all(
                parameter.device == next(model.parameters()).device
                for parameter in model.parameters()
            )
        )


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
