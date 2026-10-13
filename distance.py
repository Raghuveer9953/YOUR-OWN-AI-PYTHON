"""Distance metrics for VectorDB.

Implements Euclidean distance, Cosine distance (1 - cosine similarity),
and Manhattan distance.
"""

import math
from typing import Callable, Sequence

DistFn = Callable[[Sequence[float], Sequence[float]], float]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean (L2) distance."""
    s = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.sqrt(s)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance: 1.0 - (dot(a, b) / (|a| * |b|))."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    sim = dot / (math.sqrt(na) * math.sqrt(nb))
    # Clip numerical precision errors
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    """Manhattan (L1) distance."""
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    """Returns the distance function corresponding to metric name."""
    m = (metric or "").lower().strip()
    if m == "cosine":
        return cosine
    if m == "manhattan":
        return manhattan
    return euclidean
