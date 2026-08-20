from __future__ import annotations

import torch
from torch import Tensor, nn

from ccb.encoding import DomainCodec, TransitionBatch
from ccb.models.common import ModelOutput


class SocialMessagePassingGNN(nn.Module):
    """Permutation-equivariant D3 baseline with persistent observed edges."""

    def __init__(self, codec: DomainCodec, *, width: int = 128, message_steps: int = 4) -> None:
        super().__init__()
        if codec.domain != "social_logic" or codec.number_of_agents is None:
            raise ValueError("SocialMessagePassingGNN requires a social_logic codec")
        self.codec = codec
        self.width = width
        self.message_steps = message_steps
        self.node_initial = nn.Parameter(torch.zeros(1, 1, width))
        self.relation_embedding = nn.Embedding(3, width)
        self.message = nn.Sequential(
            nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.node_update = nn.GRUCell(width, width)
        self.pair_output = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 3)
        )

    def _readout(self, nodes: Tensor, edges: Tensor) -> Tensor:
        agents = self.codec.number_of_agents
        if agents is None:
            raise RuntimeError("number_of_agents disappeared from codec")
        left = nodes[:, :, None, :].expand(-1, -1, agents, -1)
        right = nodes[:, None, :, :].expand(-1, agents, -1, -1)
        edge_features = self.relation_embedding(edges)
        logits = self.pair_output(torch.cat((left, right, edge_features), dim=-1))
        return logits.reshape(nodes.shape[0], agents * agents, 3)

    def forward(self, batch: TransitionBatch) -> ModelOutput:
        if batch.codec != self.codec:
            raise ValueError("batch codec does not match model codec")
        agents = self.codec.number_of_agents
        if agents is None:
            raise RuntimeError("number_of_agents disappeared from codec")
        batch_size, depth = batch.operations.shape
        nodes = self.node_initial.expand(batch_size, agents, -1)
        edges = torch.ones(batch_size, agents, agents, dtype=torch.long, device=nodes.device)
        diagonal = torch.arange(agents, device=nodes.device)
        edges[:, diagonal, diagonal] = 2
        outputs = []
        trajectory: list[list[Tensor]] = [[] for _ in range(self.message_steps)]
        batch_indices = torch.arange(batch_size, device=nodes.device)

        for step in range(depth):
            operation = batch.operations[:, step]
            pair = operation // 2
            source = pair // agents
            target = pair % agents
            relation = torch.where(operation % 2 == 1, 2, 0)
            edges = edges.clone()
            edges[batch_indices, source, target] = relation
            edges[batch_indices, target, source] = relation

            for message_step in range(self.message_steps):
                senders = nodes[:, None, :, :].expand(-1, agents, -1, -1)
                edge_features = self.relation_embedding(edges)
                messages = self.message(torch.cat((senders, edge_features), dim=-1)).mean(dim=2)
                nodes = self.node_update(
                    messages.reshape(-1, self.width), nodes.reshape(-1, self.width)
                ).reshape_as(nodes)
                trajectory[message_step].append(self._readout(nodes, edges))
            outputs.append(self._readout(nodes, edges))

        logits = torch.stack(outputs, dim=1)
        loop_logits = tuple(torch.stack(values, dim=1) for values in trajectory)
        return ModelOutput(logits, loop_logits)

