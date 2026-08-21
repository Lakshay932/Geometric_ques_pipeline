from __future__ import annotations

import os

from PIL import Image

from engine.fold_punch.geometry import Axis, Point, StartShape, reconstruct_geometry

_ROUND_NDIGITS = 6
MAX_IMAGE_BYTES = 50_000

def _round_all(points: list[Point], ndigits: int = _ROUND_NDIGITS) -> frozenset[Point]:
    return frozenset((round(x, ndigits), round(y, ndigits)) for x, y in points)

def correct_answer_unique(correct_points: list[Point], distractor_point_sets: list[list[Point]]) -> bool:
    correct_key = _round_all(correct_points)
    return all(_round_all(points) != correct_key for points in distractor_point_sets)

def options_pairwise_distinct(option_point_sets: list[list[Point]]) -> bool:
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

def solvability_check(
    axis_sequence: list[Axis],
    fold_steps: list[dict],
    punch_points: list[Point],
    expected_answer_points: list[Point],
    start_shape: StartShape = "square",
) -> bool:
    rebuilt = reconstruct_geometry(axis_sequence, fold_steps, punch_points, start_shape=start_shape)
    return rebuilt.answer_points == expected_answer_points
