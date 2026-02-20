"""
Pareto front management for iterative multi-candidate prompt optimization.
"""

from __future__ import annotations

import random

from .models import ParetoCandidate


def is_dominated(candidate: ParetoCandidate, other: ParetoCandidate) -> bool:
    metrics = candidate.metric_scores
    other_scores = other.metric_scores
    if not metrics:
        return False
    all_ge = True
    any_gt = False
    for name, val in metrics.items():
        if name not in other_scores:
            return False
        ov = other_scores[name]
        if ov < val:
            all_ge = False
            break
        if ov > val:
            any_gt = True
    return all_ge and any_gt


def update_pareto_front(
    front: list[ParetoCandidate], new_candidate: ParetoCandidate
) -> list[ParetoCandidate]:
    kept: list[ParetoCandidate] = []
    for c in front:
        if is_dominated(c, new_candidate):
            c.dominated = True
        else:
            kept.append(c)
    if any(is_dominated(new_candidate, c) for c in kept):
        return kept
    return kept + [new_candidate]


def sample_from_front(
    front: list[ParetoCandidate], rng: random.Random | None = None
) -> ParetoCandidate:
    if not front:
        raise ValueError("Pareto front is empty")
    r = rng or random.Random()
    weights = [c.composite_score for c in front]
    total = sum(weights)
    if total <= 0:
        return r.choice(front)
    rnd = r.uniform(0, total)
    cum = 0.0
    for c, w in zip(front, weights):
        cum += w
        if rnd <= cum:
            return c
    return front[-1]


def sample_pair_from_front(
    front: list[ParetoCandidate], rng: random.Random | None = None
) -> tuple[ParetoCandidate, ParetoCandidate]:
    if len(front) < 2:
        raise ValueError("Pareto front has fewer than 2 candidates")
    r = rng or random.Random()
    first = sample_from_front(front, r)
    remaining = [c for c in front if c is not first]
    second = sample_from_front(remaining, r)
    return first, second


def compute_composite(
    metric_scores: dict[str, float], weights: dict[str, float]
) -> float:
    if not weights:
        return 0.0
    total = 0.0
    for name, w in weights.items():
        if name in metric_scores:
            total += metric_scores[name] * w
    return min(100.0, max(0.0, total * 100.0))


def select_best(front: list[ParetoCandidate]) -> ParetoCandidate:
    if not front:
        raise ValueError("Pareto front is empty")
    return max(front, key=lambda c: c.composite_score)
