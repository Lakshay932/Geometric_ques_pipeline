"""FR-12: any question must be re-renderable from its stored params alone,
without recomputing/re-sampling anything.
"""
import numpy as np

from engine.fold_punch.geometry import reconstruct_geometry, serialize_fold_steps
from engine.fold_punch.sampler import sample_fold_punch_geometry


def test_reconstructed_geometry_matches_original_across_many_seeds():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        params, geometry = sample_fold_punch_geometry(rng)

        fold_steps = serialize_fold_steps(geometry.steps)
        punch_points = [list(p) for p in geometry.punch_points]

        rebuilt = reconstruct_geometry(params.axis_sequence, fold_steps, punch_points)

        assert rebuilt.answer_points == geometry.answer_points
        assert rebuilt.final_polygon.equals(geometry.final_polygon)
        assert len(rebuilt.steps) == len(geometry.steps)
        for original_step, rebuilt_step in zip(geometry.steps, rebuilt.steps):
            assert original_step.axis == rebuilt_step.axis
            assert original_step.kept_side == rebuilt_step.kept_side
            assert original_step.diagonal_variant == rebuilt_step.diagonal_variant
