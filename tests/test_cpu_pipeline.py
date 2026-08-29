import tempfile
import unittest
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from ccb.dataset import (
    OfficialEvaluationFirewall,
    SplitConfig,
    generate_firewalled_split,
)
from ccb.domains.alien_grid import AlienGridDomain
from ccb.encoding import collate_episodes
from ccb.evaluation import evaluate_model
from ccb.experiment import (
    ExperimentConfig,
    _act_batch_stream,
    aggregate_experiment_results,
    run_experiment_matrix,
)
from ccb.loading import make_dataloader
from ccb.models.common import ModelOutput
from ccb.official import load_official_episodes
from ccb.training import deep_improvement_targets, supervised_loss


class OracleModel(nn.Module):
    def forward(self, batch):
        vocabulary = batch.codec.state_vocab_size
        logits = torch.nn.functional.one_hot(batch.targets, vocabulary).float() * 10
        return ModelOutput(logits)


class DatasetFirewallTests(unittest.TestCase):
    def test_official_episode_is_blocked(self) -> None:
        firewall = OfficialEvaluationFirewall.from_official_records("d1")
        official = load_official_episodes("d1")[0]
        self.assertTrue(firewall.blocks(official))
        self.assertFalse(firewall.audit((official,)).safe)

    def test_filtered_generation_is_deterministic_and_safe(self) -> None:
        firewall = OfficialEvaluationFirewall.from_official_records("d1")
        config = SplitConfig("tiny", (5, 10), 3, 9_000_000)
        left = generate_firewalled_split(AlienGridDomain().generate, config, firewall)
        right = generate_firewalled_split(AlienGridDomain().generate, config, firewall)
        self.assertEqual(left, right)
        self.assertTrue(firewall.audit(left).safe)


class VariableDepthAndEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = AlienGridDomain().generate
        self.episodes = (generator(depth=2, seed=11), generator(depth=4, seed=12))

    def test_masked_loss_ignores_padding(self) -> None:
        batch = collate_episodes(self.episodes)
        logits = torch.nn.functional.one_hot(
            batch.targets, batch.codec.state_vocab_size
        ).float() * 10
        logits[0, 2:] = -10
        loss = supervised_loss(
            ModelOutput(logits),
            batch.targets,
            loop_supervision_weight=0,
            step_mask=batch.step_mask,
        )
        self.assertLess(float(loss), 0.01)

    def test_per_depth_evaluator(self) -> None:
        report = evaluate_model(
            OracleModel(),
            make_dataloader(self.episodes, batch_size=2),
            bootstrap_resamples=10,
        )
        self.assertEqual(set(report["per_depth"]), {"2", "4"})
        self.assertEqual(report["overall"]["final_exact_accuracy"], 1.0)
        self.assertEqual(report["overall"]["trace_exact_accuracy"], 1.0)

    def test_act_stream_has_fixed_full_shape_with_variable_depth_examples(self) -> None:
        generator = AlienGridDomain().generate
        episodes = tuple(
            generator(depth=2 if seed % 2 else 4, seed=100 + seed) for seed in range(8)
        )
        config = ExperimentConfig(
            domain="d1", models=("trm_upstream_core",), batch_size=4, steps=4
        )
        batches = list(islice(_act_batch_stream(episodes, config, seed=7), 4))
        self.assertTrue(all(tuple(batch.operations.shape) == (4, 4) for batch in batches))
        self.assertTrue(any(not bool(batch.step_mask.all()) for batch in batches))


class DeepImprovementTests(unittest.TestCase):
    def test_targets_start_at_input_and_end_at_oracle(self) -> None:
        batch = collate_episodes(
            (AlienGridDomain().generate(depth=3, seed=21),)
        )
        torch.manual_seed(3)
        targets = deep_improvement_targets(
            batch.initial_state, batch.targets, batch.step_mask, 4
        )
        expected_start = batch.initial_state[:, None, :].expand_as(batch.targets)
        self.assertTrue(torch.equal(targets[0], expected_start))
        self.assertTrue(torch.equal(targets[-1], batch.targets))
        distances = [
            int(((frame != batch.targets) & batch.step_mask[:, :, None]).sum())
            for frame in targets
        ]
        self.assertEqual(distances, sorted(distances, reverse=True))


class ExperimentLauncherTests(unittest.TestCase):
    def test_dry_run_expands_model_seed_matrix_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig(
                domain="d1",
                models=("transformer", "trm_upstream_core"),
                seeds=(2, 5),
                output_directory=directory,
                steps=1,
            )
            plans = run_experiment_matrix(config, dry_run=True)
            self.assertEqual(len(plans), 4)
            self.assertFalse(any(Path(directory).iterdir()))

    def test_seed_aggregation_preserves_per_depth_results(self) -> None:
        results = []
        for seed, accuracy in ((1, 0.25), (2, 0.75)):
            evaluation = {
                "per_step_retention_p_d": 0.9,
                "success_horizon_50": 6.5,
                "success_horizon_90": 1.0,
                "normalized_accuracy_depth_auc": accuracy,
                "overall": {
                    "final_exact_accuracy": accuracy,
                    "trace_exact_accuracy": accuracy,
                    "transition_exact_accuracy": accuracy,
                    "element_accuracy": accuracy,
                    "valid_state_rate": 1.0,
                    "tfbc_rate_all": 0.0,
                    "tfbc_rate_given_final_correct": 0.0,
                },
                "per_depth": {"5": {"final_exact_accuracy": accuracy}},
            }
            results.append(
                {"model": "transformer", "seed": seed, "evaluations": {"validation": evaluation}}
            )
        summary = aggregate_experiment_results(results)
        split = summary["models"]["transformer"]["splits"]["validation"]
        self.assertEqual(split["metrics"]["final_exact_accuracy"]["mean"], 0.5)
        self.assertEqual(
            split["final_exact_accuracy_by_depth"]["5"]["runs"], 2
        )


if __name__ == "__main__":
    unittest.main()
