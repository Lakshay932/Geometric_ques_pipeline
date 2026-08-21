from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    Axis,
    FoldPunchGeometry,
    InvalidFoldError,
    PunchKind,
    StartShape,
    apply_fold,
    candidate_lines_for_axis,
    generate,
    make_start_polygon,
)
from .punches import PUNCH_KINDS, generate_with_shapes

_ALL_AXES: tuple[Axis, ...] = ("vertical", "horizontal", "diagonal")

MIN_FOLDS = 1
MAX_FOLDS = 3

MAX_FOLDS_V2 = 4
MIN_PUNCHES = 1
MAX_PUNCHES = 4

START_SHAPES: tuple[StartShape, ...] = ("square", "circle", "rectangle_2_1")

DIAGONAL_PROBABILITY = 0.35

@dataclass
class FoldPunchParams:

    fold_count: int
    axis_sequence: list[Axis]
    punch_count: int

def _build_axis_sequence(fold_count: int, rng: np.random.Generator) -> list[Axis]:
    include_diagonal = rng.random() < DIAGONAL_PROBABILITY
    sequence: list[Axis] = []
    if include_diagonal:
        for _ in range(fold_count - 1):
            sequence.append("vertical" if rng.random() < 0.5 else "horizontal")
        sequence.append("diagonal")
    else:
        for _ in range(fold_count):
            sequence.append("vertical" if rng.random() < 0.5 else "horizontal")
    return sequence

def sample_params(rng: np.random.Generator, max_folds: int = MAX_FOLDS) -> FoldPunchParams:
    fold_count = int(rng.integers(MIN_FOLDS, max_folds + 1))
    punch_count = int(rng.integers(MIN_PUNCHES, MAX_PUNCHES + 1))
    sequence = _build_axis_sequence(fold_count, rng)
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

def sample_punch_kinds(punch_count: int, rng: np.random.Generator) -> list[PunchKind]:
    indices = rng.integers(0, len(PUNCH_KINDS), size=punch_count)
    return [PUNCH_KINDS[i] for i in indices]

def sample_punch_rotations(punch_count: int, rng: np.random.Generator) -> list[float]:
    return [float(rng.uniform(0.0, 2 * np.pi)) for _ in range(punch_count)]

def sample_fold_punch_geometry_with_shapes(
    rng: np.random.Generator,
    max_retries: int = 30,
) -> tuple[FoldPunchParams, FoldPunchGeometry]:
    last_error: Exception | None = None
    for _ in range(max_retries):
        params = sample_params(rng)
        punch_kinds = sample_punch_kinds(params.punch_count, rng)
        punch_rotations = sample_punch_rotations(params.punch_count, rng)
        try:
            geometry = generate_with_shapes(
                params.axis_sequence, punch_kinds, rng, punch_rotations=punch_rotations
            )
            return params, geometry
        except (InvalidFoldError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to sample a valid shaped fold_punch geometry after {max_retries} retries"
    ) from last_error

def sample_start_shape(rng: np.random.Generator) -> StartShape:
    return START_SHAPES[int(rng.integers(0, len(START_SHAPES)))]

def sample_fold_punch_geometry_v2(
    rng: np.random.Generator,
    max_retries: int = 30,
) -> tuple[FoldPunchParams, FoldPunchGeometry]:
    last_error: Exception | None = None
    for _ in range(max_retries):
        params = sample_params(rng, max_folds=MAX_FOLDS_V2)
        start_shape = sample_start_shape(rng)
        try:
            geometry = generate(
                params.axis_sequence, params.punch_count, rng, start_shape=start_shape
            )
            return params, geometry
        except (InvalidFoldError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to sample a valid fold-space-v2 geometry after {max_retries} retries"
    ) from last_error

def _count_axis_sequences(polygon, remaining_folds: int, _rng) -> int:
    if remaining_folds == 0:
        return 1
    total = 0
    for axis in _ALL_AXES:
        if axis == "diagonal" and remaining_folds != 1:
            continue
        if not candidate_lines_for_axis(polygon, axis):
            continue
        step = apply_fold(polygon, axis, _rng, keep_side=0)
        total += _count_axis_sequences(step.polygon_after, remaining_folds - 1, _rng)
    return total

def theoretical_axis_sequence_counts(
    start_shape: StartShape = "square", max_folds: int = MAX_FOLDS_V2
) -> dict[int, int]:
    dummy_rng = np.random.default_rng(0)
    start_polygon = make_start_polygon(start_shape)
    return {
        fold_count: _count_axis_sequences(start_polygon, fold_count, dummy_rng)
        for fold_count in range(1, max_folds + 1)
    }

REFRESH_INTERVAL = 500

BIAS_DAMPEN_EXPONENT = 0.5

def compute_bucket_histogram(records: list[dict]) -> dict[tuple[int, int], int]:
    histogram: dict[tuple[int, int], int] = {}
    for record in records:
        params = record["params"]
        key = (params["fold_count"], params["punch_count"])
        histogram[key] = histogram.get(key, 0) + 1
    return histogram

def _biased_bucket_weights(
    histogram: dict[tuple[int, int], int],
    max_folds: int,
    max_punches: int,
    dampen: float,
) -> dict[tuple[int, int], float]:
    buckets = [
        (f, p) for f in range(MIN_FOLDS, max_folds + 1) for p in range(MIN_PUNCHES, max_punches + 1)
    ]
    total = sum(histogram.get(b, 0) for b in buckets)
    weights = {}
    for bucket in buckets:
        frac_filled = histogram.get(bucket, 0) / total if total else 0.0
        weights[bucket] = (1.0 - frac_filled) ** dampen
    weight_sum = sum(weights.values()) or 1.0
    return {b: w / weight_sum for b, w in weights.items()}

def sample_params_biased(
    rng: np.random.Generator,
    histogram: dict[tuple[int, int], int],
    max_folds: int = MAX_FOLDS,
    max_punches: int = MAX_PUNCHES,
    dampen: float = BIAS_DAMPEN_EXPONENT,
) -> FoldPunchParams:
    weights = _biased_bucket_weights(histogram, max_folds, max_punches, dampen)
    buckets = list(weights.keys())
    idx = int(rng.choice(len(buckets), p=[weights[b] for b in buckets]))
    fold_count, punch_count = buckets[idx]
    sequence = _build_axis_sequence(fold_count, rng)
    return FoldPunchParams(fold_count=fold_count, axis_sequence=sequence, punch_count=punch_count)

def sample_fold_punch_geometry_biased(
    rng: np.random.Generator,
    histogram: dict[tuple[int, int], int],
    max_retries: int = 30,
    max_folds: int = MAX_FOLDS,
    max_punches: int = MAX_PUNCHES,
) -> tuple[FoldPunchParams, FoldPunchGeometry]:
    last_error: Exception | None = None
    for _ in range(max_retries):
        params = sample_params_biased(rng, histogram, max_folds=max_folds, max_punches=max_punches)
        try:
            geometry = generate(params.axis_sequence, params.punch_count, rng)
            return params, geometry
        except (InvalidFoldError, ValueError) as exc:
            last_error = exc
            continue
    raise RuntimeError(
        f"Failed to sample a valid biased fold_punch geometry after {max_retries} retries"
    ) from last_error
