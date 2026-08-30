import math

import torch

from ccb.optim import AdamATan2, official_trm_learning_rate


def test_adam_atan2_matches_released_scalar_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = AdamATan2([parameter], lr=0.1, betas=(0.5, 0.25), weight_decay=0.1)
    parameter.grad = torch.tensor([3.0])
    optimizer.step()

    decayed = 2.0 * (1.0 - 0.1 * 0.1)
    mean = 3.0 * (1.0 - 0.5)
    squared = 9.0 * (1.0 - 0.25)
    denominator = math.sqrt(squared) / math.sqrt(1.0 - 0.25**1)
    expected = decayed - 0.1 / (1.0 - 0.5**1) * math.atan2(mean, denominator)
    assert torch.allclose(parameter.detach(), torch.tensor([expected]))


def test_official_warmup_cosine_schedule() -> None:
    assert official_trm_learning_rate(0, base_learning_rate=1.0, warmup_steps=2, total_steps=10) == 0.0
    assert official_trm_learning_rate(1, base_learning_rate=1.0, warmup_steps=2, total_steps=10) == 0.5
    assert official_trm_learning_rate(2, base_learning_rate=1.0, warmup_steps=2, total_steps=10) == 1.0
    assert official_trm_learning_rate(10, base_learning_rate=1.0, warmup_steps=2, total_steps=10) == 0.0
