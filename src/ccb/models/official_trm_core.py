"""Mechanical local port of the pinned official TRM primitives and core.

This module deliberately mirrors `TinyRecursiveModels/models/recursive_reasoning/trm.py`
at commit c0110373 for the no-puzzle-embedding path.  CCB-specific tokenisation
belongs outside this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def trunc_normal_init_(tensor: Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> Tensor:
    """Exact JAX-style truncated-normal initializer used by the official code."""

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * upper**2)
            pdf_l = c * math.exp(-0.5 * lower**2)
            compensated_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * compensated_std)
            tensor.clip_(lower * compensated_std, upper * compensated_std)
    return tensor


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / math.sqrt(in_features))
        )
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(
            value,
            self.weight.to(value.dtype),
            self.bias.to(value.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype) -> None:
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return F.embedding(tokens, self.embedding_weight.to(self.cast_to))


def _rotate_half(value: Tensor) -> Tensor:
    left, right = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def _apply_rotary(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    original_dtype = query.dtype
    query, key = query.to(cos.dtype), key.to(cos.dtype)
    query = query * cos.unsqueeze(-2) + _rotate_half(query) * sin.unsqueeze(-2)
    key = key * cos.unsqueeze(-2) + _rotate_half(key) * sin.unsqueeze(-2)
    return query.to(original_dtype), key.to(original_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.cos_cached = nn.Buffer(embeddings.cos(), persistent=False)
        self.sin_cached = nn.Buffer(embeddings.sin(), persistent=False)

    def forward(self) -> tuple[Tensor, Tensor]:
        return self.cos_cached, self.sin_cached


def rms_norm(hidden_states: Tensor, variance_epsilon: float) -> Tensor:
    input_dtype = hidden_states.dtype
    value = hidden_states.to(torch.float32)
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + variance_epsilon)
    return value.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float) -> None:
        super().__init__()
        intermediate = (-(round(expansion * hidden_size * 2 / 3) // -256)) * 256
        self.gate_up_proj = CastedLinear(hidden_size, intermediate * 2, bias=False)
        self.down_proj = CastedLinear(intermediate, hidden_size, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(value).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.o_proj = CastedLinear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, cos_sin: tuple[Tensor, Tensor] | None, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(batch_size, sequence_length, 3 * self.num_heads, self.head_dim)
        query, key, value = qkv[:, :, : self.num_heads], qkv[:, :, self.num_heads : 2 * self.num_heads], qkv[:, :, 2 * self.num_heads :]
        if cos_sin is not None:
            query, key = _apply_rotary(query, key, *cos_sin)
        query, key, value = (item.transpose(1, 2) for item in (query, key, value))
        attended = F.scaled_dot_product_attention(query=query, key=key, value=value, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        return self.o_proj(attended)


@dataclass(frozen=True)
class OfficialTRMConfig:
    batch_size: int
    seq_len: int
    vocab_size: int
    h_cycles: int
    l_cycles: int
    l_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    halt_max_steps: int
    halt_exploration_prob: float
    pos_encodings: str = "rope"
    forward_dtype: torch.dtype = torch.float32
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    no_act_continue: bool = True


class OfficialTRMBlock(nn.Module):
    def __init__(self, config: OfficialTRMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.hidden_size // config.num_heads, config.num_heads)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: tuple[Tensor, Tensor] | None, hidden_states: Tensor) -> Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)


class OfficialTRMReasoningModule(nn.Module):
    def __init__(self, layers: list[OfficialTRMBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: Tensor, input_injection: Tensor, *, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(cos_sin, hidden_states)
        return hidden_states


@dataclass(frozen=True)
class OfficialTRMInnerCarry:
    z_h: Tensor
    z_l: Tensor


class OfficialTRMInner(nn.Module):
    """Exact official inner TRM for the no-puzzle-embedding configuration."""

    def __init__(self, config: OfficialTRMConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.pos_encodings not in {"rope", "learned"}:
            raise ValueError("pos_encodings must be rope or learned")
        self.config = config
        embed_init_std = 1.0 / math.sqrt(config.hidden_size)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, embed_init_std, config.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.seq_len, config.rope_theta)
        else:
            self.embed_pos = CastedEmbedding(config.seq_len, config.hidden_size, embed_init_std, config.forward_dtype)
        self.L_level = OfficialTRMReasoningModule([OfficialTRMBlock(config) for _ in range(config.l_layers)])
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=config.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(config.hidden_size, dtype=config.forward_dtype), std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, inputs: Tensor) -> Tensor:
        embedding = self.embed_tokens(inputs.to(torch.int32))
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.config.forward_dtype))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> OfficialTRMInnerCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return OfficialTRMInnerCarry(
            torch.empty(shape, dtype=self.config.forward_dtype, device=self.H_init.device),
            torch.empty(shape, dtype=self.config.forward_dtype, device=self.L_init.device),
        )

    def reset_carry(self, reset_flag: Tensor, carry: OfficialTRMInnerCarry) -> OfficialTRMInnerCarry:
        return OfficialTRMInnerCarry(
            torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_h),
            torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_l),
        )

    def forward(self, carry: OfficialTRMInnerCarry, inputs: Tensor) -> tuple[OfficialTRMInnerCarry, Tensor, tuple[Tensor, Tensor]]:
        if inputs.shape[1] != self.config.seq_len:
            raise ValueError("inputs must have configured seq_len")
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        injection = self._input_embeddings(inputs)
        z_h, z_l = carry.z_h, carry.z_l
        with torch.no_grad():
            for _ in range(self.config.h_cycles - 1):
                for _ in range(self.config.l_cycles):
                    z_l = self.L_level(z_l, z_h + injection, cos_sin=cos_sin)
                z_h = self.L_level(z_h, z_l, cos_sin=cos_sin)
        for _ in range(self.config.l_cycles):
            z_l = self.L_level(z_l, z_h + injection, cos_sin=cos_sin)
        z_h = self.L_level(z_h, z_l, cos_sin=cos_sin)
        carry = OfficialTRMInnerCarry(z_h.detach(), z_l.detach())
        logits = self.lm_head(z_h)
        q_logits = self.q_head(z_h[:, 0]).to(torch.float32)
        return carry, logits, (q_logits[..., 0], q_logits[..., 1])


@dataclass(frozen=True)
class OfficialTRMACTCarry:
    inner_carry: OfficialTRMInnerCarry
    steps: Tensor
    halted: Tensor
    current_inputs: Tensor


class OfficialTRMACTWrapper(nn.Module):
    """Official ACT wrapper, parameterised by tensor inputs instead of dict batches."""

    def __init__(self, config: OfficialTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.inner = OfficialTRMInner(config)

    def initial_carry(self, inputs: Tensor) -> OfficialTRMACTCarry:
        batch_size = inputs.shape[0]
        return OfficialTRMACTCarry(
            self.inner.empty_carry(batch_size),
            torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            torch.ones(batch_size, dtype=torch.bool, device=inputs.device),
            torch.empty_like(inputs),
        )

    def forward(self, carry: OfficialTRMACTCarry, inputs: Tensor) -> tuple[OfficialTRMACTCarry, dict[str, Tensor]]:
        inner = self.inner.reset_carry(carry.halted, carry.inner_carry)
        steps = torch.where(carry.halted, 0, carry.steps)
        current = torch.where(carry.halted[:, None], inputs, carry.current_inputs)
        inner, logits, (q_halt, q_continue) = self.inner(inner, current)
        outputs: dict[str, Tensor] = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_continue}
        with torch.no_grad():
            steps = steps + 1
            is_last_step = steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and self.config.halt_max_steps > 1:
                halted = halted | (q_halt > 0 if self.config.no_act_continue else q_halt > q_continue)
                minimum = (
                    (torch.rand_like(q_halt) < self.config.halt_exploration_prob)
                    * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (steps >= minimum)
        return OfficialTRMACTCarry(inner, steps, halted, current), outputs
