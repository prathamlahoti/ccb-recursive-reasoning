from __future__ import annotations

import torch
from torch import Tensor, nn

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models.common import CellMixer, ModelOutput, ResidualMLP, StateDecoder


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


def _sinusoidal_positions(length: int, width: int, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / width)
    )
    encoding = torch.zeros(length, width, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class BatchModel(nn.Module):
    def __init__(self, codec: DomainCodec, width: int) -> None:
        super().__init__()
        self.codec = codec
        self.width = width
        self.state_embedding = nn.Embedding(codec.state_vocab_size, width)
        self.state_position = nn.Embedding(codec.state_size, width)
        self.operation_embedding = nn.Embedding(codec.operation_vocab_size, width)
        self.initial_projection = nn.Linear(codec.state_size * width, width)
        self.decoder = StateDecoder(width, codec.state_size, codec.state_vocab_size)

    def _inputs(self, batch: TransitionBatch) -> tuple[Tensor, Tensor]:
        if batch.codec != self.codec:
            raise ValueError("batch codec does not match model codec")
        positions = torch.arange(self.codec.state_size, device=batch.initial_state.device)
        initial_cells = self.state_embedding(batch.initial_state) + self.state_position(positions)
        initial_summary = self.initial_projection(initial_cells.flatten(1))
        step_encoding = _sinusoidal_positions(
            batch.operations.shape[1], self.width, batch.operations.device
        )
        operation_inputs = (
            self.operation_embedding(batch.operations)
            + initial_summary[:, None, :]
            + step_encoding[None, :, :]
        )
        return initial_cells, operation_inputs


class DirectTransformer(BatchModel):
    def __init__(
        self, codec: DomainCodec, *, width: int = 128, heads: int = 4, layers: int = 4
    ) -> None:
        super().__init__(codec, width)
        block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        _, inputs = self._inputs(batch)
        hidden = self.encoder(inputs, mask=_causal_mask(inputs.shape[1], inputs.device))
        return ModelOutput(self.decoder(hidden))


class RecurrentBaseline(BatchModel):
    def __init__(
        self, codec: DomainCodec, *, width: int = 128, layers: int = 2, cell: str = "gru"
    ) -> None:
        super().__init__(codec, width)
        recurrent = {"gru": nn.GRU, "lstm": nn.LSTM}.get(cell)
        if recurrent is None:
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.recurrent = recurrent(width, width, num_layers=layers, batch_first=True)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        _, inputs = self._inputs(batch)
        hidden, _ = self.recurrent(inputs)
        return ModelOutput(self.decoder(hidden))


class LoopedTransformer(BatchModel):
    def __init__(
        self, codec: DomainCodec, *, width: int = 128, heads: int = 4, loops: int = 4
    ) -> None:
        super().__init__(codec, width)
        self.loops = loops
        self.block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, batch_first=True, norm_first=True
        )
        self.outer_norm = nn.LayerNorm(width)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        _, recall = self._inputs(batch)
        hidden = recall
        trajectory = []
        mask = _causal_mask(recall.shape[1], recall.device)
        for _ in range(self.loops):
            hidden = self.outer_norm(self.block(hidden + recall, src_mask=mask))
            trajectory.append(self.decoder(hidden))
        return ModelOutput(trajectory[-1], tuple(trajectory))


class VanillaTRM(BatchModel):
    """Prefix-wise adaptation of the official TRM recursive update schedule."""

    def __init__(
        self,
        codec: DomainCodec,
        *,
        width: int = 128,
        loops: int = 3,
        low_cycles: int = 6,
        detach_warmup: bool = True,
    ) -> None:
        super().__init__(codec, width)
        self.loops = loops
        self.low_cycles = low_cycles
        self.detach_warmup = detach_warmup
        self.prefix = nn.GRU(width, width, batch_first=True)
        self.register_buffer("y_initial", torch.randn(1, 1, codec.state_size, width))
        self.register_buffer("z_initial", torch.randn(1, 1, codec.state_size, width))
        # Official TRM deliberately uses the same L_level for z_L and z_H.
        self.reasoning_cells = CellMixer(width)
        self.reasoning_mlp = ResidualMLP(width, expansion=4)

    def _reason(self, hidden: Tensor, injection: Tensor) -> Tensor:
        return self.reasoning_mlp(self.reasoning_cells(hidden + injection))

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        initial_cells, inputs = self._inputs(batch)
        prefix, _ = self.prefix(inputs)
        batch_size, depth, _ = prefix.shape
        initial = initial_cells[:, None, :, :].expand(-1, depth, -1, -1)
        context = (
            prefix[:, :, None, :].expand(-1, -1, self.codec.state_size, -1) + initial
        )
        y = self.y_initial.expand(batch_size, depth, -1, -1)
        z = self.z_initial.expand(batch_size, depth, -1, -1)
        trajectory = []
        # Match the released code: H_cycles-1 are refinement-only/no-grad;
        # gradients flow through the final high cycle.
        context_manager = torch.no_grad() if self.detach_warmup else torch.enable_grad()
        with context_manager:
            for _ in range(max(0, self.loops - 1)):
                for _ in range(self.low_cycles):
                    z = self._reason(z, y + context)
                y = self._reason(y, z)
                trajectory.append(self.decoder(y))
        for _ in range(self.low_cycles):
            z = self._reason(z, y + context)
        y = self._reason(y, z)
        trajectory.append(self.decoder(y))
        return ModelOutput(trajectory[-1], tuple(trajectory))


class FaithfulCCBTRM(BatchModel):
    """CCB adaptation of TRM's answer/latent deep-refinement algorithm.

    This keeps TRM's two persistent features, ``y`` (answer) and ``z``
    (latent), one shared network, detached outer refinements, and a final
    gradient-bearing refinement. It is intentionally separate from
    :class:`VanillaTRM`, the earlier prefix-wise diagnostic adaptation.
    """

    def __init__(
        self,
        codec: DomainCodec,
        *,
        width: int = 128,
        latent_steps: int = 6,
        refinement_steps: int = 3,
    ) -> None:
        super().__init__(codec, width)
        if latent_steps < 1 or refinement_steps < 1:
            raise ValueError("TRM recursion counts must be positive")
        self.latent_steps = latent_steps
        self.refinement_steps = refinement_steps
        self.y_initial = nn.Parameter(torch.randn(1, 1, codec.state_size, width))
        self.z_initial = nn.Parameter(torch.randn(1, 1, codec.state_size, width))
        self.recall_projection = nn.Linear(3 * width, width)
        self.refinement_norm = nn.LayerNorm(width)
        self.refinement_cells = CellMixer(width)
        self.refinement_mlp = ResidualMLP(width, expansion=4)
        self.halt_head = nn.Linear(width, 1)

    def _question(self, batch: TransitionBatch) -> Tensor:
        initial, operations = self._inputs(batch)
        return initial[:, None, :, :] + operations[:, :, None, :]

    def initial_states(self, batch: TransitionBatch) -> tuple[Tensor, Tensor]:
        batch_size, depth = batch.operations.shape
        shape = (batch_size, depth, self.codec.state_size, self.width)
        return self.y_initial.expand(shape), self.z_initial.expand(shape)

    def _net(self, question: Tensor, answer: Tensor, latent: Tensor) -> Tensor:
        hidden = self.recall_projection(torch.cat((question, answer, latent), dim=-1))
        hidden = self.refinement_cells(self.refinement_norm(hidden))
        return self.refinement_mlp(hidden)

    def latent_recursion(self, question: Tensor, answer: Tensor, latent: Tensor) -> tuple[Tensor, Tensor]:
        for _ in range(self.latent_steps):
            latent = self._net(question, answer, latent)
        answer = self._net(torch.zeros_like(question), answer, latent)
        return answer, latent

    def refine(
        self,
        batch: TransitionBatch,
        answer: Tensor,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor, ModelOutput, Tensor]:
        """One TRM deep-refinement update, with only its final pass tracked."""

        question = self._question(batch)
        with torch.no_grad():
            for _ in range(self.refinement_steps - 1):
                answer, latent = self.latent_recursion(question, answer, latent)
        answer, latent = self.latent_recursion(question, answer, latent)
        logits = self.decoder(answer)
        halt_logits = self.halt_head(answer.mean(dim=(1, 2))).squeeze(-1)
        return answer.detach(), latent.detach(), ModelOutput(logits), halt_logits

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        answer, latent = self.initial_states(batch)
        answer, latent, output, _ = self.refine(batch, answer, latent)
        return output


class FastSlowRecurrentModel(BatchModel):
    def __init__(self, codec: DomainCodec, *, width: int = 128, fast_loops: int = 4) -> None:
        super().__init__(codec, width)
        self.fast_loops = fast_loops
        self.fast_initial = nn.Parameter(torch.zeros(1, codec.state_size, width))
        self.fast_update = nn.GRUCell(width, width)
        self.slow_update = nn.GRUCell(width, width)
        self.fast_cells = CellMixer(width)
        self.slow_cells = CellMixer(width)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        slow, operations = self._inputs(batch)
        outputs = []
        loop_outputs: list[list[Tensor]] = [[] for _ in range(self.fast_loops)]
        for step in range(operations.shape[1]):
            recall = operations[:, step, None, :].expand(-1, self.codec.state_size, -1)
            fast = self.fast_initial.expand(slow.shape[0], -1, -1)
            for loop in range(self.fast_loops):
                fast = self.fast_update(
                    (fast + recall + self.slow_cells(slow)).reshape(-1, self.width),
                    fast.reshape(-1, self.width),
                ).reshape_as(fast)
                fast = self.fast_cells(fast)
                loop_outputs[loop].append(self.decoder(fast[:, None])[:, 0])
            slow = self.slow_update(
                self.fast_cells(fast).reshape(-1, self.width), slow.reshape(-1, self.width)
            ).reshape_as(slow)
            slow = self.slow_cells(slow)
            outputs.append(self.decoder(slow[:, None])[:, 0])
        logits = torch.stack(outputs, dim=1)
        trajectory = tuple(torch.stack(values, dim=1) for values in loop_outputs)
        return ModelOutput(logits, trajectory)


class StateTransitionRecursiveModel(BatchModel):
    """Persistent explicit-state/latent recurrence with operation and input recall."""

    def __init__(self, codec: DomainCodec, *, width: int = 128, inner_loops: int = 4) -> None:
        super().__init__(codec, width)
        self.inner_loops = inner_loops
        self.z_update = nn.GRUCell(width, width)
        self.y_update = nn.GRUCell(width, width)
        self.recall_projection = nn.Linear(3 * width, width)
        self.readout_projection = nn.Linear(2 * width, width)
        self.outer_norm_y = nn.LayerNorm(width)
        self.outer_norm_z = nn.LayerNorm(width)
        self.z_cells = CellMixer(width)
        self.y_cells = CellMixer(width)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        initial, operations = self._inputs(batch)
        y = initial
        z = torch.zeros_like(y)
        outputs = []
        loop_outputs: list[list[Tensor]] = [[] for _ in range(self.inner_loops)]
        for step in range(operations.shape[1]):
            operation = operations[:, step, None, :].expand(-1, self.codec.state_size, -1)
            recall = self.recall_projection(
                torch.cat((self.y_cells(y), operation, initial), dim=-1)
            )
            for loop in range(self.inner_loops):
                z = self.z_update(
                    recall.reshape(-1, self.width), z.reshape(-1, self.width)
                ).reshape_as(z)
                z = self.outer_norm_z(z)
                z = self.z_cells(z)
                readout = self.readout_projection(torch.cat((y, z), dim=-1))
                loop_outputs[loop].append(self.decoder(readout[:, None])[:, 0])
            y = self.y_update(
                self.z_cells(z).reshape(-1, self.width), y.reshape(-1, self.width)
            ).reshape_as(y)
            y = self.outer_norm_y(y)
            y = self.y_cells(y)
            outputs.append(self.decoder(y[:, None])[:, 0])
        logits = torch.stack(outputs, dim=1)
        trajectory = tuple(torch.stack(values, dim=1) for values in loop_outputs)
        return ModelOutput(logits, trajectory)
