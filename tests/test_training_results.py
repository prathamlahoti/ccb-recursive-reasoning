import tempfile
import unittest
from pathlib import Path

import torch

from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.results import markdown_table, write_result
from ccb.training import (
    TrainConfig,
    build_model,
    evaluate_batch,
    save_checkpoint,
    seed_everything,
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
