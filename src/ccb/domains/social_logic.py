from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum

from ccb.records import Episode, Transition


class Relation(IntEnum):
    RIVAL = -1
    NEUTRAL = 0
    ALLY = 1


SocialState = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SocialOperation:
    source: int
    target: int
    relation: Relation

    def __str__(self) -> str:
        return f"ADD {self.relation.name} {self.source} {self.target}"


def signed_closure(number_of_agents: int, edges: tuple[SocialOperation, ...]) -> SocialState:
    """Compute consistent undirected signed-transitive closure.

    Composition follows sign multiplication: ally*ally and rival*rival imply
    ally; ally*rival and rival*ally imply rival. Contradictory signed paths are
    rejected.
    """

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(number_of_agents)]
    for edge in edges:
        if edge.source == edge.target:
            raise ValueError("Self edges are not valid updates")
        sign = int(edge.relation)
        if sign == 0:
            raise ValueError("Only ally or rival edges may be added")
        adjacency[edge.source].append((edge.target, sign))
        adjacency[edge.target].append((edge.source, sign))

    matrix = [[0] * number_of_agents for _ in range(number_of_agents)]
    for root in range(number_of_agents):
        inferred: dict[int, int] = {root: 1}
        stack = [root]
        while stack:
            current = stack.pop()
            for neighbor, edge_sign in adjacency[current]:
                candidate = inferred[current] * edge_sign
                if neighbor in inferred and inferred[neighbor] != candidate:
                    raise ValueError("Contradictory signed social graph")
                if neighbor not in inferred:
                    inferred[neighbor] = candidate
                    stack.append(neighbor)
        for node, sign in inferred.items():
            matrix[root][node] = sign
    return tuple(tuple(row) for row in matrix)


def _component_labels(state: SocialState) -> tuple[int, ...]:
    labels: list[int] = []
    for row in state:
        members = [index for index, value in enumerate(row) if value != 0]
        labels.append(min(members))
    return tuple(labels)


@dataclass(frozen=True)
class SocialLogicDomain:
    number_of_agents: int = 10
    topology: str = "official_random"

    def generate(self, *, depth: int, seed: int) -> Episode[SocialState, SocialOperation]:
        if depth < 1:
            raise ValueError("depth must be positive")
        if self.number_of_agents < 2:
            raise ValueError("At least two agents are required")
        rng = random.Random(seed)
        if self.topology not in {"official_random", "chain", "star"}:
            raise ValueError("topology must be official_random, chain, or star")
        operations: list[SocialOperation] = []
        initial = signed_closure(self.number_of_agents, tuple())
        state = initial
        transitions: list[Transition[SocialState, SocialOperation]] = []

        for index in range(1, depth + 1):
            labels = _component_labels(state)
            between = [
                (left, right)
                for left in range(self.number_of_agents)
                for right in range(left + 1, self.number_of_agents)
                if labels[left] != labels[right]
            ]
            if self.topology == "official_random":
                source, target = sorted(rng.sample(range(self.number_of_agents), 2))
                if labels[source] == labels[target]:
                    relation = Relation(state[source][target])
                else:
                    relation = rng.choice((Relation.ALLY, Relation.RIVAL))
                preferred = []
            elif self.topology == "chain":
                preferred = [(node, node + 1) for node in range(self.number_of_agents - 1)]
            else:
                preferred = [(0, node) for node in range(1, self.number_of_agents)]
            if self.topology != "official_random":
                unused_preferred = [
                    pair
                    for pair in preferred
                    if all((operation.source, operation.target) != pair for operation in operations)
                ]
                preferred_between = [pair for pair in unused_preferred if pair in between]
                if preferred_between:
                    source, target = preferred_between[0]
                    relation = rng.choice((Relation.ALLY, Relation.RIVAL))
                elif between:
                    source, target = rng.choice(between)
                    relation = rng.choice((Relation.ALLY, Relation.RIVAL))
                elif unused_preferred:
                    source, target = unused_preferred[0]
                    relation = Relation(state[source][target])
                else:
                    source, target = sorted(rng.sample(range(self.number_of_agents), 2))
                    relation = Relation(state[source][target])
            operation = SocialOperation(source, target, relation)
            operations.append(operation)
            next_state = signed_closure(self.number_of_agents, tuple(operations))
            transitions.append(Transition(index, operation, state, next_state))
            state = next_state

        query = tuple(sorted(rng.sample(range(self.number_of_agents), 2)))
        return Episode(
            domain="social_logic",
            seed=seed,
            initial_state=initial,
            operations=tuple(operations),
            transitions=tuple(transitions),
            final_state=state,
            metadata={
                "number_of_agents": self.number_of_agents,
                "query": query,
                "query_answer": int(state[query[0]][query[1]]),
                "semantics": "ccb_official_component_faction_v1",
                "topology": self.topology,
                "provenance": (
                    "ccb_official"
                    if self.number_of_agents == 10 and self.topology == "official_random"
                    else "ccb_learn"
                ),
            },
            specification=(
                "ccb_official_v1"
                if self.number_of_agents == 10 and self.topology == "official_random"
                else "ccb_learn_v1"
            ),
        )

    def solve(self, *, depth: int, seed: int) -> tuple[str, list[str], str]:
        episode = self.generate(depth=depth, seed=seed)

        def pair_state(state: SocialState) -> str:
            relations = []
            for left in range(self.number_of_agents):
                for right in range(left + 1, self.number_of_agents):
                    relation = state[left][right]
                    if relation:
                        tag = "F" if relation == int(Relation.ALLY) else "R"
                        relations.append(f"{chr(65 + left)}{chr(65 + right)}:{tag}")
            return f"[{', '.join(sorted(relations))}]"

        prompt_lines = []
        trace = []
        for transition in episode.transitions:
            operation = transition.operation
            relationship = "ALLIANCE" if operation.relation is Relation.ALLY else "RIVALRY"
            left, right = chr(65 + operation.source), chr(65 + operation.target)
            prompt_lines.append(
                f"Step {transition.index}: {left} and {right} form an {relationship}."
            )
            trace.append(f"Step {transition.index}: {pair_state(transition.after)}")
        return "\n".join(prompt_lines), trace, pair_state(episode.final_state)
