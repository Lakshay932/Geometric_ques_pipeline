from __future__ import annotations

import os

from PIL import Image

from engine.rotation_series.geometry import RotationDirection
from engine.rotation_series.geometry import reconstruct_geometry_from_points as _rs_reconstruct_geometry_from_points

_ROUND_NDIGITS = 6
MAX_IMAGE_BYTES = 50_000


def _round_all(points, ndigits: int = _ROUND_NDIGITS) -> frozenset:
    return frozenset((round(x, ndigits), round(y, ndigits)) for x, y in points)


def correct_answer_unique(correct_points, distractor_point_sets) -> bool:
    correct_key = _round_all(correct_points)
    return all(_round_all(points) != correct_key for points in distractor_point_sets)


def options_pairwise_distinct(option_point_sets) -> bool:
    keys = [_round_all(points) for points in option_point_sets]
    return len(set(keys)) == len(keys)


def render_sanity(image_path: str, max_bytes: int = MAX_IMAGE_BYTES) -> bool:
    if not os.path.exists(image_path):
        return False
    if os.path.getsize(image_path) > max_bytes:
        return False
    grayscale = Image.open(image_path).convert("L")
    darkest, lightest = grayscale.getextrema()
    return darkest != lightest


def rotation_series_solvability_check(
    original_points,
    step_degrees: int,
    direction: RotationDirection,
    sequence_length: int,
    expected_answer_points,
) -> bool:
    rebuilt = _rs_reconstruct_geometry_from_points(original_points, step_degrees, direction, sequence_length)
    actual = list(rebuilt.answer.exterior.coords)[:-1]
    return actual == expected_answer_points
