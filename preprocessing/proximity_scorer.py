"""
proximity_scorer.py
────────────────────
Computes the syntactic proximity score s_i for each token relative to a
target entity, as specified in the paper's mathematical formulation:

    alpha_i = softmax(w^T tanh(We*hi + Ws*si))

where s_i is "the inverse shortest path length between token i and the
entity head token" in the dependency parse tree.

This is what makes the attention "entity-aware" rather than generic —
without this, the model attends uniformly across the whole document
regardless of which entity it's scoring framing for.
"""

from __future__ import annotations

import logging
from collections import deque

from preprocessing.ner_pipeline import EntitySpan, ProcessedDoc

log = logging.getLogger(__name__)


def build_dependency_graph(head_indices: list[int]) -> dict[int, list[int]]:
    """
    Build an undirected adjacency list from spaCy's head-index array.

    spaCy represents the dependency tree as: token i's head is
    head_indices[i]. A token whose head is itself is the sentence root.
    We treat the tree as undirected for shortest-path purposes — distance
    from "the minister" to "criticized" should be the same regardless of
    which one governs the other grammatically.

    Args:
        head_indices: head_indices[i] = index of token i's syntactic head

    Returns:
        Adjacency list: {token_idx: [neighbor_idx, ...]}
    """
    graph: dict[int, list[int]] = {i: [] for i in range(len(head_indices))}

    for i, head in enumerate(head_indices):
        if head == i:
            continue  # root token, no edge to itself
        graph[i].append(head)
        graph[head].append(i)

    return graph


def shortest_path_length(
    graph: dict[int, list[int]],
    source: int,
    target: int,
    max_distance: int = 50,
) -> int:
    """
    BFS shortest path length between two tokens in the dependency graph.

    Args:
        graph:        adjacency list from build_dependency_graph
        source:       starting token index
        target:       target token index (entity head)
        max_distance: cutoff to avoid pathological worst-case on malformed trees

    Returns:
        Path length (number of edges). Returns max_distance if unreachable
        (shouldn't happen in a connected parse tree, but defensive).
    """
    if source == target:
        return 0
    if source not in graph or target not in graph:
        return max_distance

    visited = {source}
    queue: deque[tuple[int, int]] = deque([(source, 0)])

    while queue:
        node, dist = queue.popleft()
        if dist >= max_distance:
            continue
        for neighbor in graph.get(node, []):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return max_distance


def compute_proximity_scores(
    processed: ProcessedDoc,
    target_entity: EntitySpan,
    decay: str = "inverse",
) -> list[float]:
    """
    Compute s_i for every token i relative to the target entity's head token,
    per the paper's formulation: "inverse shortest path length between
    token i and the entity head token."

    Args:
        processed:     ProcessedDoc from ner_pipeline.process_document
        target_entity: the entity being scored for framing (EntitySpan)
        decay:         'inverse' → s_i = 1/(1+dist)
                       'exponential' → s_i = exp(-dist)
                       Both are monotonically decreasing in distance;
                       'inverse' is what the paper's equation implies.

    Returns:
        List of proximity scores, one per token, same length as
        processed.tokens. Higher score = syntactically closer to entity.
    """
    n_tokens = len(processed.tokens)
    if n_tokens == 0:
        return []

    graph = build_dependency_graph(processed.head_indices)
    entity_head_idx = target_entity.token_idx

    scores: list[float] = []
    for i in range(n_tokens):
        dist = shortest_path_length(graph, i, entity_head_idx)
        if decay == "exponential":
            score = _exp_decay(dist)
        else:
            score = 1.0 / (1.0 + dist)
        scores.append(score)

    return scores


def _exp_decay(distance: int, rate: float = 0.5) -> float:
    import math
    return math.exp(-rate * distance)


def compute_all_entity_proximities(
    processed: ProcessedDoc,
) -> dict[str, list[float]]:
    """
    Convenience batch function: compute proximity scores for every entity
    found in the document, keyed by entity text.

    Useful when an article mentions multiple candidate target entities
    and Module 3 needs to score framing for each independently.

    Args:
        processed: ProcessedDoc with entities already extracted

    Returns:
        {entity_text: [proximity_score_per_token, ...]}
    """
    result: dict[str, list[float]] = {}
    for entity in processed.entities:
        # If the same entity text appears multiple times, keep the LAST
        # mention's proximity profile (most recent context) — could also
        # average across mentions, but last-mention is simpler and the
        # paper doesn't specify aggregation for repeated entities.
        result[entity.text] = compute_proximity_scores(processed, entity)
    return result


def get_entity_context_window(
    processed: ProcessedDoc,
    target_entity: EntitySpan,
    window_sentences: int = 1,
) -> tuple[int, int]:
    """
    Return the token range covering the target entity's sentence plus
    `window_sentences` sentences on either side — useful for truncating
    long articles to a relevant window before feeding to RoBERTa
    (max_seq_len: 512 per model_config.yaml).

    Args:
        processed:        ProcessedDoc with sent_boundaries populated
        target_entity:    the entity to center the window on
        window_sentences: how many sentences of context on each side

    Returns:
        (start_token_idx, end_token_idx) — half-open range [start, end)
    """
    if not processed.sent_boundaries:
        return (0, len(processed.tokens))

    target_sent_idx = target_entity.sent_idx
    n_sents = len(processed.sent_boundaries)

    lo = max(0, target_sent_idx - window_sentences)
    hi = min(n_sents - 1, target_sent_idx + window_sentences)

    start_tok = processed.sent_boundaries[lo][0]
    end_tok   = processed.sent_boundaries[hi][1]

    return (start_tok, end_tok)
