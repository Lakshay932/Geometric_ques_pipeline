from schemas.record import ExamStyle, Family, ImagePaths, QuestionText, Record


def test_dummy_record_validates():
    record = Record(
        family=Family.FOLD_PUNCH,
        sub_type="vertical_fold_single_punch",
        difficulty=1,
        params={"fold_axis": "vertical", "fold_count": 1, "punch_count": 1},
        image_paths=ImagePaths(question="data/images/q1/question.png", options={"A": "a.png"}),
        correct_option="A",
        distractor_rules=["wrong_symmetry_axis"],
        exam_style=ExamStyle.SSC,
        text=QuestionText(stem="stem", option_labels={"A": "opt a"}, explanation="expl"),
    )

    assert record.family == Family.FOLD_PUNCH
    assert record.difficulty == 1
    assert record.correct_option == "A"
    assert record.version == 1


def test_difficulty_out_of_range_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Record(
            family=Family.MIRROR_WATER,
            sub_type="45deg_mirror",
            difficulty=6,
            image_paths=ImagePaths(question="q.png"),
            correct_option="A",
            text=QuestionText(stem="s", explanation="e"),
        )
