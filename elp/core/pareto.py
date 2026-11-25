"""Multi-objective Pareto frontier, hypervolume indicator, and representative selection."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def pareto_frontier(points: Mapping[str, Sequence[float]]) -> list[str]:
    """Return IDs of non-dominated points under maximization across all objectives."""
    if not points:
        return []
    dims = {len(v) for v in points.values()}
    if len(dims) != 1 or 0 in dims:
        raise ValueError("All objective vectors must share a positive dimension")
    items = [(k, tuple(float(x) for x in v)) for k, v in points.items()]
    frontier_ids: list[str] = []
    for key, p in items:
        dominated = any(
            other_key != key and all(yq >= xp for xp, yq in zip(p, q)) and any(yq > xp for xp, yq in zip(p, q))
            for other_key, q in items
        )
        if not dominated:
            frontier_ids.append(key)
    return frontier_ids


def _recursive_hypervolume(points: list[tuple[float, ...]]) -> float:
    if not points:
        return 0.0
    if len(points[0]) == 1:
        return max(p[0] for p in points)
    levels = sorted({0.0, *(p[0] for p in points if p[0] > 0.0)})
    return sum(
        (high - low) * _recursive_hypervolume([p[1:] for p in points if p[0] >= high])
        for low, high in zip(levels, levels[1:])
    )


def hypervolume(points: Sequence[Sequence[float]], reference: Sequence[float] | None = None) -> float:
    """Compute exact dominated hypervolume relative to `reference` (default origin)."""
    if not points:
        return 0.0
    dim = len(points[0])
    if dim == 0 or any(len(p) != dim for p in points):
        raise ValueError("All points must share a positive dimension")
    ref = (0.0,) * dim if reference is None else tuple(float(r) for r in reference)
    if len(ref) != dim:
        raise ValueError("Reference dimension must match point dimension")
    shifted = [
        tuple(float(pi) - ri for pi, ri in zip(p, ref))
        for p in points
        if all(float(pi) >= ri for pi, ri in zip(p, ref))
    ]
    return _recursive_hypervolume(shifted)


def select_representatives(points: Mapping[str, Sequence[float]]) -> dict[str, str]:
    """Select Quality, Speed, and Balanced representatives from the Pareto frontier."""
    frontier_ids = pareto_frontier(points)
    if not frontier_ids:
        raise ValueError("Cannot select representatives from an empty objective set")
    front = {k: tuple(float(x) for x in points[k]) for k in frontier_ids}
    dim = len(next(iter(front.values())))

    quality_id = max(front, key=lambda k: (front[k][0], sum(front[k])))
    speed_id = max(front, key=lambda k: (front[k][-1], sum(front[k]))) if dim >= 2 else quality_id

    mins = [min(v[i] for v in front.values()) for i in range(dim)]
    maxs = [max(v[i] for v in front.values()) for i in range(dim)]

    def utopia_distance(vec: tuple[float, ...]) -> float:
        sq = 0.0
        for i in range(dim):
            span = maxs[i] - mins[i]
            if span > 0:
                sq += ((maxs[i] - vec[i]) / span) ** 2
        return math.sqrt(sq)

    balanced_id = min(front, key=lambda k: (utopia_distance(front[k]), -sum(front[k]), k))
    return {
        "quality": quality_id,
        "speed": speed_id,
        "balanced": balanced_id,
    }
