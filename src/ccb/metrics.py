from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class Outcome(str, Enum):
    CORRECT = "correct"
    REASONING = "reasoning"
    CONSTRAINT = "constraint"
    FORMAT = "format"
    TRUNCATION = "truncation"
    API = "api"


@dataclass(frozen=True)
class TraceMetrics:
    final_correct: bool
    trace_correct: bool
    first_divergence: int | None
    tfbc: bool
    transition_accuracy: float
    outcome: Outcome


def evaluate_trace(
    predicted_trace: Sequence[Any],
    predicted_answer: Any,
    oracle_trace: Sequence[Any],
    oracle_answer: Any,
    *,
    validity_check: Callable[[Any], bool] | None = None,
) -> TraceMetrics:
    if len(predicted_trace) < len(oracle_trace):
        return TraceMetrics(
            final_correct=False,
            trace_correct=False,
            first_divergence=len(predicted_trace) + 1,
            tfbc=False,
            transition_accuracy=sum(
                left == right for left, right in zip(predicted_trace, oracle_trace)
            )
            / len(oracle_trace),
            outcome=Outcome.TRUNCATION,
        )

    matches = [left == right for left, right in zip(predicted_trace, oracle_trace)]
    first_divergence = next(
        (index for index, matches_oracle in enumerate(matches, start=1) if not matches_oracle),
        None,
    )
    final_correct = predicted_answer == oracle_answer
    trace_correct = first_divergence is None and len(predicted_trace) == len(oracle_trace)
    tfbc = final_correct and not trace_correct
    if validity_check is not None and any(not validity_check(state) for state in predicted_trace):
        outcome = Outcome.CONSTRAINT
    elif final_correct and trace_correct:
        outcome = Outcome.CORRECT
    else:
        outcome = Outcome.REASONING
    return TraceMetrics(
        final_correct=final_correct,
        trace_correct=trace_correct,
        first_divergence=first_divergence,
        tfbc=tfbc,
        transition_accuracy=sum(matches) / len(oracle_trace) if oracle_trace else 1.0,
        outcome=outcome,
    )


def state_element_accuracy(predicted: Any, oracle: Any) -> float:
    def flatten(value: Any) -> list[Any]:
        if isinstance(value, (list, tuple)):
            result: list[Any] = []
            for item in value:
                result.extend(flatten(item))
            return result
        return [value]

    predicted_values = flatten(predicted)
    oracle_values = flatten(oracle)
    if len(predicted_values) != len(oracle_values) or not oracle_values:
        return 0.0
    return sum(left == right for left, right in zip(predicted_values, oracle_values)) / len(
        oracle_values
    )


def _log_likelihood(p: float, counts: Mapping[int, tuple[int, int]]) -> float:
    total = 0.0
    for depth, (correct, trials) in counts.items():
        if depth < 1 or not 0 <= correct <= trials:
            raise ValueError("Invalid depth/count entry")
        success_probability = min(max(p**depth, 1e-15), 1 - 1e-15)
        total += correct * math.log(success_probability)
        total += (trials - correct) * math.log1p(-success_probability)
    return total


def fit_step_retention(
    counts: Mapping[int, tuple[int, int]], *, lower_bound: float = 0.5
) -> float:
    """Maximum-likelihood estimate for P(correct at depth N)=p_d**N."""

    if not counts:
        raise ValueError("At least one depth cell is required")
    if not 0 < lower_bound < 1:
        raise ValueError("lower_bound must lie in (0, 1)")
    left = lower_bound
    right = 1 - 1e-12
    golden_ratio = (math.sqrt(5) - 1) / 2
    x1 = right - golden_ratio * (right - left)
    x2 = left + golden_ratio * (right - left)
    f1 = _log_likelihood(x1, counts)
    f2 = _log_likelihood(x2, counts)
    for _ in range(120):
        if f1 < f2:
            left = x1
            x1, f1 = x2, f2
            x2 = left + golden_ratio * (right - left)
            f2 = _log_likelihood(x2, counts)
        else:
            right = x2
            x2, f2 = x1, f1
            x1 = right - golden_ratio * (right - left)
            f1 = _log_likelihood(x1, counts)
    estimate = (left + right) / 2
    if _log_likelihood(lower_bound, counts) >= _log_likelihood(estimate, counts):
        return lower_bound
    return estimate


def success_horizon(step_retention: float, threshold: float = 0.5) -> float:
    if not 0 < step_retention <= 1 or not 0 < threshold < 1:
        raise ValueError("Probabilities are outside their valid ranges")
    if step_retention == 1:
        return math.inf
    return math.log(threshold) / math.log(step_retention)


def bootstrap_retention_interval(
    counts: Mapping[int, tuple[int, int]],
    *,
    resamples: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
    lower_bound: float = 0.5,
) -> tuple[float, float]:
    if resamples < 2:
        raise ValueError("resamples must be at least two")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    estimate = fit_step_retention(counts, lower_bound=lower_bound)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        sampled_counts: dict[int, tuple[int, int]] = {}
        for depth, (_, trials) in counts.items():
            probability = estimate**depth
            correct = sum(rng.random() < probability for _ in range(trials))
            sampled_counts[depth] = (correct, trials)
        samples.append(fit_step_retention(sampled_counts, lower_bound=lower_bound))
    samples.sort()
    tail = (1 - confidence) / 2
    lower_index = max(0, min(resamples - 1, int(tail * resamples)))
    upper_index = max(0, min(resamples - 1, int((1 - tail) * resamples) - 1))
    return samples[lower_index], samples[upper_index]


def wilson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes/trials are invalid")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def normalized_accuracy_depth_auc(accuracy_by_depth: Mapping[int, float]) -> float:
    """Trapezoidal AUC divided by the evaluated depth range."""

    if len(accuracy_by_depth) < 2:
        raise ValueError("at least two depth cells are required")
    points = sorted(accuracy_by_depth.items())
    if any(depth < 1 or not 0 <= accuracy <= 1 for depth, accuracy in points):
        raise ValueError("invalid depth/accuracy entry")
    area = sum(
        (right_depth - left_depth) * (left_accuracy + right_accuracy) / 2
        for (left_depth, left_accuracy), (right_depth, right_accuracy) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def _binomial_cdf(at_most: int, trials: int, probability: float) -> float:
    if at_most < 0:
        return 0.0
    if at_most >= trials:
        return 1.0
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 0.0
    term = (1 - probability) ** trials
    total = term
    ratio = probability / (1 - probability)
    for successes in range(at_most):
        term *= (trials - successes) / (successes + 1) * ratio
        total += term
    return min(1.0, max(0.0, total))


def clopper_pearson_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Exact two-sided binomial interval used in the CCB paper."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes/trials are invalid")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    tail = (1 - confidence) / 2

    if successes == 0:
        lower = 0.0
    else:
        left, right = 0.0, successes / trials
        for _ in range(100):
            middle = (left + right) / 2
            upper_tail = 1 - _binomial_cdf(successes - 1, trials, middle)
            if upper_tail < tail:
                left = middle
            else:
                right = middle
        lower = (left + right) / 2

    if successes == trials:
        upper = 1.0
    else:
        left, right = successes / trials, 1.0
        for _ in range(100):
            middle = (left + right) / 2
            lower_tail = _binomial_cdf(successes, trials, middle)
            if lower_tail > tail:
                left = middle
            else:
                right = middle
        upper = (left + right) / 2
    return lower, upper
