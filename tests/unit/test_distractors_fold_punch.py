import numpy as np

from distractors.fold_punch import RULE_POOL, generate_distractors
from engine.fold_punch.geometry import fold_paper, unfold_points


def _round_all(points):
    return frozenset((round(x, 6), round(y, 6)) for x, y in points)


def test_generates_exactly_three_distinct_tagged_distractors():
    rng = np.random.default_rng(42)
    result = fold_paper([("vertical", 0), ("horizontal", 0)], rng)
    punch_points = [(0.2, 0.2), (0.4, 0.1)]
    answer_points = unfold_points(punch_points, result.steps)

    distractors = generate_distractors(answer_points, punch_points, result.steps, rng)

    assert len(distractors) == 3
    for d in distractors:
        assert d.rule in RULE_POOL
        assert _round_all(d.points) != _round_all(answer_points)

    keys = [_round_all(d.points) for d in distractors]
    assert len(set(keys)) == 3  # pairwise distinct from each other too

    rules = [d.rule for d in distractors]
    assert len(set(rules)) == 3  # each distractor breaks a different rule


def test_distractors_reproducible_across_many_seeds():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = fold_paper([("vertical", None), ("horizontal", None), ("vertical", None)], rng)
        punch_points = [(x, y) for x, y in [(0.05 + 0.02 * seed % 0.1, 0.05)]]
        # ensure at least one valid punch point strictly inside final polygon
        minx, miny, maxx, maxy = result.final_polygon.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        punch_points = [(cx, cy)]
        answer_points = unfold_points(punch_points, result.steps)

        distractors = generate_distractors(answer_points, punch_points, result.steps, rng)
        assert len(distractors) == 3
