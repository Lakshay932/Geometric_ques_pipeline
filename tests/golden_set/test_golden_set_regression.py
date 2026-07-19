"""Golden-set regression (TRD Section 9): re-derive each frozen record's
geometry from its stored params and confirm it still matches — catches
silent regressions in the geometry engine or renderer on future changes.
"""
import json
import os

import numpy as np
import pytest
from PIL import Image

from engine.fold_punch.geometry import reconstruct_geometry
from render.fold_punch import render_option_image

GOLDEN_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_PATH = os.path.join(GOLDEN_DIR, "fold_punch_golden.jsonl")
ANSWERS_PATH = os.path.join(GOLDEN_DIR, "fold_punch_golden_answers.jsonl")
IMAGES_DIR = os.path.join(GOLDEN_DIR, "images")


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


RECORDS = _load_jsonl(RECORDS_PATH)
ANSWERS_BY_ID = {a["question_id"]: a["answer_points"] for a in _load_jsonl(ANSWERS_PATH)}


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["question_id"])
def test_golden_record_answer_points_unchanged(record: dict):
    params = record["params"]
    geometry = reconstruct_geometry(
        params["axis_sequence"], params["fold_steps"], params["punch_points"]
    )
    expected = [tuple(p) for p in ANSWERS_BY_ID[record["question_id"]]]
    actual = [tuple(p) for p in geometry.answer_points]
    assert actual == expected


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["question_id"])
def test_golden_record_correct_option_image_unchanged(record: dict):
    question_id = record["question_id"]
    params = record["params"]
    geometry = reconstruct_geometry(
        params["axis_sequence"], params["fold_steps"], params["punch_points"]
    )
    rerendered = render_option_image(geometry.answer_points)

    frozen_path = os.path.join(IMAGES_DIR, question_id, f"{record['correct_option']}.png")
    frozen = Image.open(frozen_path).convert("RGB")

    assert np.array_equal(np.array(rerendered), np.array(frozen))
