"""Freeze a stratified sample of verified fold_punch records as the golden
set (TRD Section 9 — golden-set regression).

For each selected record, also captures its answer_points independently
(recomputed from params via reconstruct_geometry) into a companion file,
so the regression test in tests/golden_set/ can catch silent drift in
either the geometry engine or the renderer on future changes.
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import defaultdict

from engine.fold_punch.geometry import reconstruct_geometry

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDS_PATH = os.path.join(REPO_ROOT, "data", "records", "fold_punch.jsonl")
GOLDEN_DIR = os.path.join(REPO_ROOT, "tests", "golden_set")
GOLDEN_RECORDS_PATH = os.path.join(GOLDEN_DIR, "fold_punch_golden.jsonl")
GOLDEN_ANSWERS_PATH = os.path.join(GOLDEN_DIR, "fold_punch_golden_answers.jsonl")
GOLDEN_IMAGES_DIR = os.path.join(GOLDEN_DIR, "images")

TARGET_COUNT = 80


def _load_records() -> list[dict]:
    with open(RECORDS_PATH) as f:
        return [json.loads(line) for line in f]


def _stratified_sample(records: list[dict], target_count: int) -> list[dict]:
    rng = random.Random(0)
    buckets: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        buckets[record["difficulty"]].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict] = []
    difficulties = sorted(buckets.keys())
    while len(selected) < target_count:
        progressed = False
        for difficulty in difficulties:
            if buckets[difficulty]:
                selected.append(buckets[difficulty].pop())
                progressed = True
                if len(selected) >= target_count:
                    break
        if not progressed:
            break
    return selected


def main() -> None:
    os.makedirs(GOLDEN_IMAGES_DIR, exist_ok=True)
    records = _load_records()
    selected = _stratified_sample(records, TARGET_COUNT)

    with open(GOLDEN_RECORDS_PATH, "w") as records_file, open(
        GOLDEN_ANSWERS_PATH, "w"
    ) as answers_file:
        for record in selected:
            question_id = record["question_id"]
            src_dir = os.path.join(REPO_ROOT, "data", "images", question_id)
            dest_dir = os.path.join(GOLDEN_IMAGES_DIR, question_id)
            shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

            records_file.write(json.dumps(record) + "\n")

            params = record["params"]
            geometry = reconstruct_geometry(
                params["axis_sequence"], params["fold_steps"], params["punch_points"]
            )
            answers_file.write(
                json.dumps({"question_id": question_id, "answer_points": geometry.answer_points})
                + "\n"
            )

    difficulty_counts = defaultdict(int)
    for record in selected:
        difficulty_counts[record["difficulty"]] += 1
    print(f"Froze {len(selected)} golden records: {dict(sorted(difficulty_counts.items()))}")


if __name__ == "__main__":
    main()
