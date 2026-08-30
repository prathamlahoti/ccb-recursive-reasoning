"""Compare the local mechanical core against a checked-out official TRM tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    if not (args.upstream / "models" / "recursive_reasoning" / "trm.py").is_file():
        raise SystemExit("--upstream must point to TinyRecursiveModels checkout root")
    sys.path.insert(0, str(args.upstream))
    from models.recursive_reasoning.trm import (  # type: ignore[import-not-found]
        TinyRecursiveReasoningModel_ACTV1Config,
        TinyRecursiveReasoningModel_ACTV1_Inner,
    )
    from ccb.models.official_trm_core import OfficialTRMConfig, OfficialTRMInner

    torch.manual_seed(20260830)
    shared = {
        "batch_size": 2,
        "seq_len": 7,
        "puzzle_emb_ndim": 0,
        "puzzle_emb_len": 0,
        "num_puzzle_identifiers": 1,
        "vocab_size": 11,
        "H_cycles": 3,
        "L_cycles": 2,
        "H_layers": 0,
        "L_layers": 2,
        "hidden_size": 32,
        "expansion": 4.0,
        "num_heads": 4,
        "pos_encodings": "rope",
        "halt_max_steps": 3,
        "halt_exploration_prob": 0.0,
        "forward_dtype": "float32",
        "no_ACT_continue": True,
    }
    reference = TinyRecursiveReasoningModel_ACTV1_Inner(
        TinyRecursiveReasoningModel_ACTV1Config(**shared)
    )
    local = OfficialTRMInner(
        OfficialTRMConfig(
            batch_size=2,
            seq_len=7,
            vocab_size=11,
            h_cycles=3,
            l_cycles=2,
            l_layers=2,
            hidden_size=32,
            expansion=4.0,
            num_heads=4,
            halt_max_steps=3,
            halt_exploration_prob=0.0,
            pos_encodings="rope",
            forward_dtype=torch.float32,
            no_act_continue=True,
        )
    )
    local.load_state_dict(reference.state_dict(), strict=True)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]])
    reference_carry = reference.reset_carry(
        torch.ones(2, dtype=torch.bool), reference.empty_carry(2)
    )
    local_carry = local.reset_carry(torch.ones(2, dtype=torch.bool), local.empty_carry(2))
    reference_new, reference_logits, reference_q = reference(
        reference_carry, {"inputs": inputs, "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    )
    local_new, local_logits, local_q = local(local_carry, inputs)
    torch.testing.assert_close(local_logits, reference_logits, rtol=0, atol=0)
    torch.testing.assert_close(local_q[0], reference_q[0], rtol=0, atol=0)
    torch.testing.assert_close(local_q[1], reference_q[1], rtol=0, atol=0)
    torch.testing.assert_close(local_new.z_h, reference_new.z_H, rtol=0, atol=0)
    torch.testing.assert_close(local_new.z_l, reference_new.z_L, rtol=0, atol=0)
    local_loss = local_logits.square().sum() + local_q[0].square().sum() + local_q[1].square().sum()
    reference_loss = (
        reference_logits.square().sum() + reference_q[0].square().sum() + reference_q[1].square().sum()
    )
    local_loss.backward()
    reference_loss.backward()
    for name, parameter in local.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
    print("OFFICIAL_TRM_FORWARD_AND_GRADIENT_EQUIVALENCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
