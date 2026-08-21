from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon

from engine.mirror_water.geometry import MIRROR_AXES, MirrorAxis, mirror_line_for_axis
from engine.fold_punch.geometry import reflect_polygon
from evaluate.dedup import PHASH_DUPLICATE_THRESHOLD, hamming_distance, perceptual_hash
from render.mirror_water import render_option_image

RULE_POOL = ["rotated_180", "wrong_axis", "identical_copy", "shifted_reflection"]

NEW_RULE_POOL_V2 = ["rotated_90", "scaled_reflection", "glide_reflection", "partial_reflection"]
RULE_POOL_V2 = RULE_POOL + NEW_RULE_POOL_V2

_SHIFT_MAGNITUDE = 0.05

_SCALE_FACTOR = 0.70
_ROUND_NDIGITS = 6

_DISTINGUISHABILITY_THRESHOLD = PHASH_DUPLICATE_THRESHOLD

@dataclass
class MirrorWaterDistractor:

    shape: Polygon
    rule: str

def _signature(shape: Polygon, ndigits: int = _ROUND_NDIGITS) -> tuple:
    return tuple(sorted((round(x, ndigits), round(y, ndigits)) for x, y in shape.exterior.coords[:-1]))

def _rotated_180(original: Polygon) -> Polygon:
    return rotate(original, 180, origin=original.centroid)

def _wrong_axis(original: Polygon, correct_axis: MirrorAxis, rng: np.random.Generator) -> Polygon:
    other_axes = [a for a in MIRROR_AXES if a != correct_axis]
    axis = other_axes[int(rng.integers(0, len(other_axes)))]
    return reflect_polygon(original, mirror_line_for_axis(axis))

def _identical_copy(original: Polygon) -> Polygon:
    return Polygon(original.exterior.coords)

def _shifted_reflection(answer: Polygon, rng: np.random.Generator, shift_mag: float = _SHIFT_MAGNITUDE) -> Polygon:
    angle = rng.uniform(0, 2 * np.pi)
    dx, dy = shift_mag * np.cos(angle), shift_mag * np.sin(angle)
    return translate(answer, dx, dy)

def _rotated_90(original: Polygon) -> Polygon:
    return rotate(original, 90, origin=original.centroid)

def _scaled_reflection(answer: Polygon, scale_factor: float = _SCALE_FACTOR) -> Polygon:
    return scale(answer, xfact=scale_factor, yfact=scale_factor, origin=answer.centroid)

def _glide_reflection(
    answer: Polygon, correct_axis: MirrorAxis, rng: np.random.Generator, shift_mag: float = _SHIFT_MAGNITUDE
) -> Polygon:
    (x1, y1), (x2, y2) = mirror_line_for_axis(correct_axis)
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return translate(answer, sign * shift_mag * ux, sign * shift_mag * uy)

def _partial_reflection(
    original: Polygon, answer: Polygon, rng: np.random.Generator, max_attempts: int = 5
) -> Polygon | None:
    orig_pts = list(original.exterior.coords)[:-1]
    ans_pts = list(answer.exterior.coords)[:-1]
    n = len(orig_pts)
    if n < 3:
        return None
    for _ in range(max_attempts):
        k = int(rng.integers(1, n))
        reflect_idx = set(rng.choice(n, size=k, replace=False).tolist())
        candidate_pts = [ans_pts[i] if i in reflect_idx else orig_pts[i] for i in range(n)]
        candidate = Polygon(candidate_pts)
        if candidate.is_valid:
            return candidate
    return None

def _apply_rule(
    rule: str, original: Polygon, answer: Polygon, correct_axis: MirrorAxis, rng: np.random.Generator
) -> Polygon | None:
    if rule == "rotated_180":
        return _rotated_180(original)
    if rule == "wrong_axis":
        return _wrong_axis(original, correct_axis, rng)
    if rule == "identical_copy":
        return _identical_copy(original)
    if rule == "shifted_reflection":
        return _shifted_reflection(answer, rng)
    if rule == "rotated_90":
        return _rotated_90(original)
    if rule == "scaled_reflection":
        return _scaled_reflection(answer)
    if rule == "glide_reflection":
        return _glide_reflection(answer, correct_axis, rng)
    if rule == "partial_reflection":
        return _partial_reflection(original, answer, rng)
    raise ValueError(f"Unknown distractor rule: {rule}")

def generate_distractors(
    original: Polygon,
    answer: Polygon,
    correct_axis: MirrorAxis,
    rng: np.random.Generator,
    count: int = 3,
    max_attempts_per_rule: int = 5,
    rule_pool: list[str] = RULE_POOL,
    axis_offset: float = 0.0,
) -> list[MirrorWaterDistractor]:
    seen = {_signature(answer)}
    results: list[MirrorWaterDistractor] = []

    mirror_line = mirror_line_for_axis(correct_axis, axis_offset)
    accepted_hashes = [perceptual_hash(render_option_image(answer, mirror_line))]

    pool = list(rule_pool)
    rng.shuffle(pool)

    for rule in pool:
        if len(results) >= count:
            break
        for _ in range(max_attempts_per_rule):
            candidate = _apply_rule(rule, original, answer, correct_axis, rng)
            if candidate is None:
                break
            key = _signature(candidate)
            if key in seen:
                continue
            candidate_hash = perceptual_hash(render_option_image(candidate, mirror_line))
            if any(
                hamming_distance(candidate_hash, h) <= _DISTINGUISHABILITY_THRESHOLD
                for h in accepted_hashes
            ):
                continue
            seen.add(key)
            accepted_hashes.append(candidate_hash)
            results.append(MirrorWaterDistractor(shape=candidate, rule=rule))
            break

    if len(results) < count:
        raise RuntimeError(f"Could only generate {len(results)}/{count} distinct mirror_water distractors")
    return results[:count]
