import numpy as np

from engine.fold_punch.difficulty import compute_difficulty
from engine.fold_punch.sampler import (
    MAX_FOLDS,
    MAX_PUNCHES,
    MIN_FOLDS,
    MIN_PUNCHES,
    sample_fold_punch_geometry,
)


def test_sampler_produces_valid_geometry_across_many_seeds():
    for seed in range(50):
        rng = np.random.default_rng(seed)
        params, geometry = sample_fold_punch_geometry(rng)

        assert MIN_FOLDS <= params.fold_count <= MAX_FOLDS
        assert MIN_PUNCHES <= params.punch_count <= MAX_PUNCHES
        assert len(geometry.steps) == params.fold_count
        assert len(geometry.punch_points) == params.punch_count
        assert len(geometry.answer_points) >= 1
        assert geometry.axis_sequence.count("diagonal") <= 1


def test_difficulty_within_1_to_5_across_configs():
    for fold_count in range(1, 4):
        for punch_count in range(1, 5):
            for axis_seq in ([], ["diagonal"], ["vertical", "horizontal"]):
                for rules in ([], ["missing_hole"], ["shifted_hole", "wrong_symmetry_axis"]):
                    difficulty = compute_difficulty(fold_count, punch_count, axis_seq, rules)
                    assert 1 <= difficulty <= 5


def test_more_folds_and_subtler_rules_increase_difficulty():
    easy = compute_difficulty(1, 1, [], ["missing_hole"])
    hard = compute_difficulty(3, 4, ["diagonal"], ["shifted_hole", "wrong_symmetry_axis"])
    assert hard > easy
