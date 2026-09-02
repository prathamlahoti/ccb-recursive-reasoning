import math
from pathlib import Path

import torch

from ccb.optim import AdamATan2, official_trm_learning_rate
from ccb.fit_gate import load_fit_gate_config


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
    assert official_trm_learning_rate(
        10, base_learning_rate=1.0, warmup_steps=2, total_steps=10, min_ratio=1.0
    ) == 1.0


def test_corrected_fit_ladder_uses_released_training_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    names = (
        "d1_official_trm_fit_8x1_v2.json",
        "d1_official_trm_fit_16x5_v2.json",
        "d1_official_trm_fit_64x5_v2.json",
    )
    configs = [load_fit_gate_config(root / "configs" / name) for name in names]
    assert [(item.examples, item.depth) for item in configs] == [(8, 1), (16, 5), (64, 5)]
    for item in configs:
        assert item.model == "official_trm_ccb"
        assert item.width == 512
        assert item.layers_or_loops == 2
        assert item.optimizer == "adam_atan2"
        assert item.optimizer_betas == (0.9, 0.95)
        assert item.weight_decay == 0.1
        assert item.learning_rate == 1e-4
        assert item.lr_warmup_steps == 2000
        assert item.lr_min_ratio == 1.0
        assert item.trm_h_cycles == 3
        assert item.trm_l_cycles == 6
        assert item.trm_halt_max_steps == 16
        assert item.official_trm_forward_dtype == "bfloat16"
        assert item.evaluate_ema is True
        assert item.gate_weights == "ema"
    control_names = (
        "d1_token_transformer_fit_8x1_v2.json",
        "d1_token_transformer_fit_16x5_v2.json",
        "d1_token_transformer_fit_64x5_v2.json",
    )
    controls = [load_fit_gate_config(root / "configs" / name) for name in control_names]
    for recurrent, control in zip(configs, controls):
        assert control.model == "ccb_token_transformer"
        matched = (
            "examples", "depth", "seed", "data_seed", "width", "layers_or_loops",
            "steps", "learning_rate", "weight_decay", "optimizer", "optimizer_betas",
            "lr_warmup_steps", "lr_min_ratio", "trm_max_depth",
            "official_trm_forward_dtype", "pass_threshold",
        )
        assert all(getattr(recurrent, key) == getattr(control, key) for key in matched)
