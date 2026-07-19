import os
import shutil
import tempfile

import numpy as np
import pytest

from providers.stub import StubLLMProvider, StubVLMProvider
from schemas.record import Record
from verify.graph import run_fold_punch_pipeline


@pytest.fixture
def data_dir():
    path = tempfile.mkdtemp(prefix="fold_punch_data_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_pipeline_returns_either_a_record_or_a_flagged_entry(data_dir):
    vlm = StubVLMProvider()
    llm = StubLLMProvider()
    seen_verified = False
    seen_flagged = False

    for seed in range(30):
        rng = np.random.default_rng(seed)
        record, flagged = run_fold_punch_pipeline(rng, vlm, llm, data_dir)

        assert (record is None) != (flagged is None)  # exactly one is set

        if record is not None:
            seen_verified = True
            assert isinstance(record, Record)
            assert record.correct_option in record.image_paths.options
            assert len(record.distractor_rules) == 3
            assert not os.path.isabs(record.image_paths.question)
            assert os.path.exists(os.path.join(data_dir, record.image_paths.question))
            for path in record.image_paths.options.values():
                assert not os.path.isabs(path)
                assert os.path.exists(os.path.join(data_dir, path))
        else:
            seen_flagged = True
            assert "attempts_log" in flagged
            assert not os.path.isabs(flagged["image_dir"])
            assert os.path.isdir(os.path.join(data_dir, flagged["image_dir"]))

    # with the deterministic stub VLM (always answers "A"), both outcomes
    # should occur across 30 random seeds.
    assert seen_verified
    assert seen_flagged
