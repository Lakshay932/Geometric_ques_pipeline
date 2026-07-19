import numpy as np

from distractors.fold_punch import generate_distractors
from engine.fold_punch.sampler import sample_fold_punch_geometry
from render.fold_punch import (
    PANEL_SIZE,
    render_option_image,
    render_question_image,
)


def _has_dark_pixels(image, threshold=100):
    grayscale = image.convert("L")
    extrema = grayscale.getextrema()
    return extrema[0] < threshold


def test_question_and_option_images_render_and_contain_marks():
    rng = np.random.default_rng(7)
    params, geometry = sample_fold_punch_geometry(rng)
    distractors = generate_distractors(
        geometry.answer_points, geometry.punch_points, geometry.steps, rng
    )

    question_image = render_question_image(
        geometry.steps, geometry.final_polygon, geometry.punch_points
    )
    assert question_image.height == PANEL_SIZE + 24
    assert _has_dark_pixels(question_image)

    correct_image = render_option_image(geometry.answer_points)
    assert correct_image.size == (PANEL_SIZE, PANEL_SIZE)
    assert _has_dark_pixels(correct_image)

    for distractor in distractors:
        option_image = render_option_image(distractor.points)
        assert _has_dark_pixels(option_image)


def test_rendering_across_many_sampled_records():
    for seed in range(15):
        rng = np.random.default_rng(seed)
        params, geometry = sample_fold_punch_geometry(rng)
        distractors = generate_distractors(
            geometry.answer_points, geometry.punch_points, geometry.steps, rng
        )
        render_question_image(geometry.steps, geometry.final_polygon, geometry.punch_points)
        for pts in [geometry.answer_points] + [d.points for d in distractors]:
            render_option_image(pts)
