import numpy as np
import pytest
from shapely.geometry import Point as ShapelyPoint

from engine.fold_punch.geometry import (
    InvalidFoldError,
    apply_fold,
    fold_paper,
    reflect_point,
    unfold_points,
    unit_square,
)


def _assert_points_close(actual, expected, tol=1e-6):
    actual_sorted = sorted(actual)
    expected_sorted = sorted(expected)
    assert len(actual_sorted) == len(expected_sorted)
    for a, e in zip(actual_sorted, expected_sorted):
        assert a[0] == pytest.approx(e[0], abs=tol)
        assert a[1] == pytest.approx(e[1], abs=tol)


def test_reflect_point_across_vertical_line():
    line = ((0.5, 0.0), (0.5, 1.0))
    assert reflect_point((0.25, 0.5), line) == pytest.approx((0.75, 0.5))


def test_reflect_point_across_main_diagonal():
    line = ((0.0, 0.0), (1.0, 1.0))
    assert reflect_point((0.7, 0.3), line) == pytest.approx((0.3, 0.7))


def test_single_vertical_fold_keep_left_unfold_single_punch():
    rng = np.random.default_rng(0)
    result = fold_paper([("vertical", 0)], rng)  # keep_side=0 -> left half
    final = result.final_polygon
    minx, miny, maxx, maxy = final.bounds
    assert (minx, maxx) == pytest.approx((0.0, 0.5))
    assert (miny, maxy) == pytest.approx((0.0, 1.0))

    answer = unfold_points([(0.25, 0.5)], result.steps)
    _assert_points_close(answer, [(0.25, 0.5), (0.75, 0.5)])


def test_vertical_then_horizontal_fold_quarters_the_square():
    rng = np.random.default_rng(0)
    result = fold_paper([("vertical", 0), ("horizontal", 0)], rng)
    final = result.final_polygon
    minx, miny, maxx, maxy = final.bounds
    assert (minx, maxx) == pytest.approx((0.0, 0.5))
    assert (miny, maxy) == pytest.approx((0.0, 0.5))

    answer = unfold_points([(0.25, 0.25)], result.steps)
    _assert_points_close(
        answer,
        [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)],
    )


def test_single_diagonal_fold_keep_lower_right_triangle():
    rng = np.random.default_rng(0)
    # main diagonal (0,0)-(1,1); side containing (1,0) is the lower-right triangle.
    result = fold_paper([("diagonal", None)], rng, diagonal_variant="main", keep_point=(1.0, 0.0))
    final = result.final_polygon
    assert final.contains(ShapelyPoint(0.9, 0.05))

    answer = unfold_points([(0.7, 0.3)], result.steps)
    _assert_points_close(answer, [(0.7, 0.3), (0.3, 0.7)])


def test_diagonal_then_same_diagonal_again_is_invalid():
    rng = np.random.default_rng(0)
    result = fold_paper([("diagonal", None)], rng, diagonal_variant="main", keep_point=(1.0, 0.0))
    current = result.final_polygon
    with pytest.raises(InvalidFoldError):
        apply_fold(current, "diagonal", rng, diagonal_variant="main")


def test_diagonal_then_vertical_is_invalid():
    rng = np.random.default_rng(0)
    result = fold_paper([("diagonal", None)], rng, diagonal_variant="main", keep_point=(1.0, 0.0))
    current = result.final_polygon
    with pytest.raises(InvalidFoldError):
        apply_fold(current, "vertical", rng)


def test_diagonal_then_opposite_diagonal_is_valid():
    rng = np.random.default_rng(0)
    result = fold_paper([("diagonal", None)], rng, diagonal_variant="main", keep_point=(1.0, 0.0))
    current = result.final_polygon
    # should not raise
    step = apply_fold(current, "diagonal", rng, diagonal_variant="anti")
    assert step.axis == "diagonal"


def test_unit_square_bounds():
    square = unit_square()
    assert square.bounds == pytest.approx((0.0, 0.0, 1.0, 1.0))
    assert square.area == pytest.approx(1.0)
