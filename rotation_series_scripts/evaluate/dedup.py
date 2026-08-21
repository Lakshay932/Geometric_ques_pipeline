from __future__ import annotations

import hashlib
import json

from PIL import Image

from engine.fold_punch.geometry import Point

GRID_SNAP = 1.0 / 64

_DIHEDRAL_TRANSFORMS = (
    lambda x, y: (x, y),
    lambda x, y: (1 - y, x),
    lambda x, y: (1 - x, 1 - y),
    lambda x, y: (y, 1 - x),
    lambda x, y: (1 - x, y),
    lambda x, y: (x, 1 - y),
    lambda x, y: (y, x),
    lambda x, y: (1 - y, 1 - x),
)

_PHASH_SIZE = 32

PHASH_DUPLICATE_THRESHOLD = 3

def canonical_param_hash(params: dict) -> str:
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _snap(value: float, grid: float = GRID_SNAP) -> float:
    return round(round(value / grid) * grid, 6)

def canonical_geometry_hash(answer_points: list[Point], grid: float = GRID_SNAP) -> str:
    candidates = []
    for transform in _DIHEDRAL_TRANSFORMS:
        transformed = sorted(
            (_snap(x2), _snap(y2)) for x2, y2 in (transform(x, y) for x, y in answer_points)
        )
        candidates.append(transformed)
    canonical = min(candidates)
    payload = json.dumps(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def canonical_asymmetric_geometry_hash(points: list[Point], grid: float = GRID_SNAP) -> str:
    snapped = sorted((_snap(x, grid), _snap(y, grid)) for x, y in points)
    payload = json.dumps(snapped)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def perceptual_hash(image: Image.Image, hash_size: int = _PHASH_SIZE) -> int:
    small = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(small.getdata())
    average = sum(pixels) / len(pixels)
    bits = 0
    for i, pixel in enumerate(pixels):
        if pixel > average:
            bits |= 1 << i
    return bits

def hamming_distance(hash_a: int, hash_b: int) -> int:
    return bin(hash_a ^ hash_b).count("1")

def embedding_similarity_flag(embedding_text: str, prior_texts: list[str]) -> bool:
    return False
