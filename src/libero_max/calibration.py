"""Pure helpers for deterministic, feasibility-preserving calibration."""

import math
from typing import Iterable, List, Sequence, Tuple


AXIS_DIRECTIONS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)
DIAGONAL_DIRECTIONS: Tuple[Tuple[float, float], ...] = tuple(
    (x / math.sqrt(2.0), y / math.sqrt(2.0))
    for x, y in ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0))
)
CANONICAL_DIRECTIONS = AXIS_DIRECTIONS + DIAGONAL_DIRECTIONS


def rank_lateral_directions(
    task_axes: Iterable[Sequence[float]],
) -> List[Tuple[float, float]]:
    """Prefer directions perpendicular to the source-to-goal task axis.

    Ties retain the explicit canonical order, so calibration is deterministic
    and independent of dictionary or process ordering.
    """

    normalized_axes = []
    for axis in task_axes:
        if len(axis) != 2:
            raise ValueError("task axes must contain two numbers")
        x, y = float(axis[0]), float(axis[1])
        norm = math.hypot(x, y)
        if math.isfinite(norm) and norm > 1e-9:
            normalized_axes.append((x / norm, y / norm))

    def score(direction: Tuple[float, float]) -> float:
        if not normalized_axes:
            return 0.0
        return sum(
            abs(direction[0] * axis[0] + direction[1] * axis[1])
            for axis in normalized_axes
        ) / len(normalized_axes)

    return sorted(CANONICAL_DIRECTIONS, key=score)


def select_common_direction(
    ranked_directions: Iterable[Tuple[float, float]],
    passing_directions: Iterable[Tuple[float, float]],
) -> Tuple[float, float]:
    passing = set(passing_directions)
    for direction in ranked_directions:
        if direction in passing:
            return direction
    raise ValueError("no canonical direction passes every required state")
