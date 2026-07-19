"""Parameter Sampler for fold_punch (TRD Section 3.1, step 1).

Draws a random valid configuration (fold axis sequence, punch count) and
asks the geometry engine to realize it, retrying with a fresh draw if the
engine rejects the sequence (InvalidFoldError) or can't place the requested
number of well-separated punches.

Uniform sampling for now — difficulty-bucket-aware biasing is a Phase 3
concern (see IMPLEMENTATION_PHASES.md Phase 3, "Difficulty balancing at
scale") once real dataset stats exist to bias against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Axis, FoldPunchGeometry, InvalidFoldError, generate

MIN_FOLDS = 1
MAX_FOLDS = 3
MIN_PUNCHES = 1
MAX_PUNCHES = 4
DIAGONAL_PROBABILITY = 0.35


@dataclass
class FoldPunchParams:
    fold_count: int
    axis_sequence: list[Axis]
    punch_count: int


def sample_params(rng: np.random.Generator) -> FoldPunchParams:
    fold_count = int(rng.integers(MIN_FOLDS, MAX_FOLDS + 1))
    punch_count = int(rng.integers(MIN_PUNCHES, MAX_PUNCHES + 1))

    include_diagonal = rng.random() < DIAGONAL_PROBABILITY
    sequence: list[Axis] = []
    if include_diagonal:
        for _ in range(fold_count - 1):
            sequence.append("vertical" if rng.random() < 0.5 else "horizontal")
        sequence.append("diagonal")
    else:
        for _ in range(fold_count):
            sequence.append("vertical" if rng.random() < 0.5 else "horizontal")

    return FoldPunchParams(fold_count=fold_count, axis_sequence=sequence, punch_count=punch_count)


def sample_fold_punch_geometry(
    rng: np.random.Generator,
    max_retries: int = 30,
) -> tuple[FoldPunchParams, FoldPunchGeometry]:
    last_error: Exception | None = None
    for _ in range(max_retries):
        params = sample_params(rng)
        try:
            geometry = generate(params.axis_sequence, params.punch_count, rng)
            return params, geometry
        except (InvalidFoldError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to sample a valid fold_punch geometry after {max_retries} retries"
    ) from last_error
