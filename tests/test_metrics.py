import math
import unittest

from ccb.metrics import (
    Outcome,
    bootstrap_retention_interval,
    clopper_pearson_interval,
    evaluate_trace,
    fit_step_retention,
    normalized_accuracy_depth_auc,
    state_element_accuracy,
    success_horizon,
    wilson_interval,
)


class TraceMetricTests(unittest.TestCase):
    def test_fully_correct_trace(self) -> None:
        metrics = evaluate_trace([1, 2, 3], 3, [1, 2, 3], 3)
        self.assertTrue(metrics.final_correct)
        self.assertTrue(metrics.trace_correct)
        self.assertIsNone(metrics.first_divergence)
        self.assertFalse(metrics.tfbc)
        self.assertEqual(metrics.outcome, Outcome.CORRECT)

    def test_tfbc_and_first_divergence(self) -> None:
        metrics = evaluate_trace([1, 9, 3], 3, [1, 2, 3], 3)
        self.assertTrue(metrics.final_correct)
        self.assertFalse(metrics.trace_correct)
        self.assertEqual(metrics.first_divergence, 2)
        self.assertTrue(metrics.tfbc)
        self.assertEqual(metrics.outcome, Outcome.REASONING)

    def test_truncation(self) -> None:
        metrics = evaluate_trace([1], 1, [1, 2, 3], 3)
        self.assertEqual(metrics.outcome, Outcome.TRUNCATION)
        self.assertEqual(metrics.first_divergence, 2)

    def test_constraint_failure(self) -> None:
        metrics = evaluate_trace(
            [1, -1, 3], 0, [1, 2, 3], 3, validity_check=lambda state: state >= 0
        )
        self.assertEqual(metrics.outcome, Outcome.CONSTRAINT)

    def test_state_element_accuracy(self) -> None:
        self.assertEqual(state_element_accuracy([[1, 2], [3, 4]], [[1, 0], [3, 0]]), 0.5)


class RetentionMetricTests(unittest.TestCase):
    def test_recovers_known_probability(self) -> None:
        counts = {5: (59, 100), 10: (35, 100), 20: (12, 100)}
        estimate = fit_step_retention(counts)
        self.assertAlmostEqual(estimate, 0.9, delta=0.015)

    def test_lower_bound(self) -> None:
        estimate = fit_step_retention({5: (0, 100), 10: (0, 100)})
        self.assertEqual(estimate, 0.5)

    def test_horizon(self) -> None:
        self.assertAlmostEqual(success_horizon(0.9, 0.5), math.log(0.5) / math.log(0.9))
        self.assertTrue(math.isinf(success_horizon(1.0)))

    def test_bootstrap_is_deterministic(self) -> None:
        counts = {5: (59, 100), 10: (35, 100)}
        left = bootstrap_retention_interval(counts, resamples=50, seed=123)
        right = bootstrap_retention_interval(counts, resamples=50, seed=123)
        self.assertEqual(left, right)
        self.assertLess(left[0], left[1])

    def test_wilson_interval(self) -> None:
        lower, upper = wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertAlmostEqual(lower, 1 - upper)

    def test_normalized_auc(self) -> None:
        self.assertAlmostEqual(normalized_accuracy_depth_auc({1: 1.0, 3: 0.5}), 0.75)

    def test_clopper_pearson_interval(self) -> None:
        lower, upper = clopper_pearson_interval(50, 100)
        self.assertAlmostEqual(lower, 0.398321, places=5)
        self.assertAlmostEqual(upper, 0.601679, places=5)
        self.assertAlmostEqual(clopper_pearson_interval(0, 10)[1], 0.308497, places=5)


if __name__ == "__main__":
    unittest.main()
