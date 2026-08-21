from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
from PIL import Image

from distractors.fold_punch import RULE_POOL_V2, generate_distractors, generate_shape_distractors
from engine.fold_punch.difficulty import compute_difficulty
from engine.fold_punch.geometry import make_start_polygon, serialize_fold_steps
from engine.fold_punch.punches import make_punch_polygon, serialize_punch_shapes
from engine.fold_punch.sampler import (
    sample_fold_punch_geometry_v2,
    sample_fold_punch_geometry_with_shapes,
)
from evaluate.dedup import (
    PHASH_DUPLICATE_THRESHOLD,
    canonical_geometry_hash,
    canonical_param_hash,
    perceptual_hash,
)
from evaluate.quality import correct_answer_unique, options_pairwise_distinct, render_sanity, solvability_check
from evaluate.report import build_scorecard, check_diversity_gate, write_report
from evaluate.store import DedupStore, default_store_path
from providers.factory import get_llm_provider
from render.fold_punch import render_option_image, render_question_image, save_png
from schemas.record import ExamStyle, Family, ImagePaths, Record, Source, Verification
from textgen.generator import assign_option_letters, build_question_text
from uuid import uuid4

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_DATA_DIR, "records", "fold_punch.jsonl")

MAX_ATTEMPTS = 4
TRACK_FOLD_V2 = "fold_v2"
TRACK_SHAPES = "shapes"

def _build_attempt(rng: np.random.Generator, data_dir: str) -> dict:
    track = TRACK_FOLD_V2 if rng.random() < 0.5 else TRACK_SHAPES

    if track == TRACK_FOLD_V2:
        params, geometry = sample_fold_punch_geometry_v2(rng)
        distractors = generate_distractors(
            geometry.answer_points, geometry.punch_points, geometry.steps, rng, rule_pool=RULE_POOL_V2
        )
        punch_shapes_for_question = None
    else:
        params, geometry = sample_fold_punch_geometry_with_shapes(rng)
        punch_shapes_for_question = [make_punch_polygon(p) for p in geometry.punches]
        distractors = generate_shape_distractors(
            geometry.answer_shapes, punch_shapes_for_question, geometry.steps, rng
        )

    letters = assign_option_letters(rng)
    correct_letter, distractor_letters = letters[0], letters[1:]

    if track == TRACK_FOLD_V2:
        option_points_by_letter = {correct_letter: geometry.answer_points}
        option_shapes_by_letter = None
    else:
        option_points_by_letter = {correct_letter: geometry.answer_points}
        option_shapes_by_letter = {correct_letter: geometry.answer_shapes}
    distractor_rule_by_letter: dict[str, str] = {}
    for letter, distractor in zip(distractor_letters, distractors):
        option_points_by_letter[letter] = distractor.points
        if option_shapes_by_letter is not None:
            option_shapes_by_letter[letter] = distractor.shapes
        distractor_rule_by_letter[letter] = distractor.rule

    question_id = str(uuid4())
    image_dir = os.path.join(data_dir, "images", question_id)
    os.makedirs(image_dir, exist_ok=True)

    outline_polygon = make_start_polygon(geometry.start_shape)
    question_path = os.path.join(image_dir, "question.png")
    save_png(
        render_question_image(geometry.steps, geometry.final_polygon, geometry.punch_points, punch_shapes_for_question),
        question_path,
    )
    question_rel = os.path.join("images", question_id, "question.png")

    option_paths = {}
    option_dests = {}
    for letter, points in option_points_by_letter.items():
        shapes = option_shapes_by_letter[letter] if option_shapes_by_letter is not None else None
        path = os.path.join(image_dir, f"{letter}.png")
        save_png(render_option_image(points, shapes=shapes, outline_polygon=outline_polygon), path)
        option_paths[letter] = path
        option_dests[letter] = os.path.join("images", question_id, f"{letter}.png")

    realized_params = {
        **asdict(params),
        "start_shape": geometry.start_shape,
        "fold_steps": serialize_fold_steps(geometry.steps),
        "punch_points": [list(p) for p in geometry.punch_points],
    }
    if track == TRACK_SHAPES:
        realized_params["punch_shapes"] = serialize_punch_shapes(geometry.punches)

    return {
        "track": track,
        "params": params,
        "geometry": geometry,
        "correct_letter": correct_letter,
        "distractor_letters": distractor_letters,
        "option_points_by_letter": option_points_by_letter,
        "distractor_rule_by_letter": distractor_rule_by_letter,
        "question_id": question_id,
        "question_path": question_path,
        "question_rel": question_rel,
        "option_paths": option_paths,
        "option_dests": option_dests,
        "realized_params": realized_params,
    }

def _quality_ok(attempt: dict) -> bool:
    option_points_by_letter = attempt["option_points_by_letter"]
    correct_points = option_points_by_letter[attempt["correct_letter"]]
    distractor_points = [option_points_by_letter[letter] for letter in attempt["distractor_letters"]]
    realized = attempt["realized_params"]
    return (
        correct_answer_unique(correct_points, distractor_points)
        and options_pairwise_distinct(list(option_points_by_letter.values()))
        and render_sanity(attempt["question_path"])
        and all(render_sanity(p) for p in attempt["option_paths"].values())
        and solvability_check(
            attempt["params"].axis_sequence,
            realized["fold_steps"],
            realized["punch_points"],
            attempt["geometry"].answer_points,
            start_shape=attempt["geometry"].start_shape,
        )
    )

def _finalize_record(attempt: dict, llm_provider, exam_style: ExamStyle) -> Record:
    params = attempt["params"]
    geometry = attempt["geometry"]
    track = attempt["track"]
    distractor_rules_in_order = [
        attempt["distractor_rule_by_letter"][letter] for letter in attempt["distractor_letters"]
    ]
    text = build_question_text(
        llm_provider, asdict(params), attempt["correct_letter"], attempt["distractor_letters"], distractor_rules_in_order
    )
    punch_kinds = [p.kind for p in geometry.punches] if track == TRACK_SHAPES else None
    difficulty = compute_difficulty(
        params.fold_count,
        params.punch_count,
        params.axis_sequence,
        distractor_rules_in_order,
        punch_kinds=punch_kinds,
    )
    sub_type_suffix = "shapes" if track == TRACK_SHAPES else geometry.start_shape
    return Record(
        question_id=attempt["question_id"],
        family=Family.FOLD_PUNCH,
        sub_type="_".join(params.axis_sequence) + f"_{params.punch_count}punch_{sub_type_suffix}",
        difficulty=difficulty,
        params=attempt["realized_params"],
        image_paths=ImagePaths(question=attempt["question_rel"], options=attempt["option_dests"]),
        correct_option=attempt["correct_letter"],
        distractor_rules=distractor_rules_in_order,
        exam_style=exam_style,
        text=text,
        tags=["fold_punch", "upgraded_v2", track, geometry.start_shape, *params.axis_sequence],
        embedding_text=f"{text.stem} {' '.join(params.axis_sequence)}",
        verification=Verification(vlm_model=None, vlm_answer=None, agree=None, verified_at=None),
        source=Source.SYNTHETIC,
    )

def generate_upgraded_gated_batch(
    count: int,
    seed: int,
    data_dir: str,
    store: DedupStore,
    llm_provider,
    exam_style: ExamStyle = ExamStyle.GENERIC,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[list[Record], list[dict]]:
    verified: list[Record] = []
    rejected: list[dict] = []

    for i in range(count):
        rng = np.random.default_rng(seed + i)
        last_reason = "quality_gate_failed"
        last_attempt: dict | None = None
        winner: tuple[dict, str, str, int] | None = None

        for _ in range(max_attempts):
            attempt = _build_attempt(rng, data_dir)
            last_attempt = attempt
            param_hash = canonical_param_hash(attempt["realized_params"])
            canonical_hash = canonical_geometry_hash(attempt["geometry"].answer_points)
            correct_image = Image.open(attempt["option_paths"][attempt["correct_letter"]])
            phash = perceptual_hash(correct_image)

            is_duplicate = store.has_param_hash(param_hash) or store.has_canonical_hash(canonical_hash)
            if not is_duplicate:
                closest = store.closest_phash_distance(phash)
                is_duplicate = closest is not None and closest <= PHASH_DUPLICATE_THRESHOLD

            if is_duplicate:
                last_reason = "duplicate"
                continue
            if not _quality_ok(attempt):
                last_reason = "quality_gate_failed"
                continue

            winner = (attempt, param_hash, canonical_hash, phash)
            break

        if winner is not None:
            attempt, param_hash, canonical_hash, phash = winner
            record = _finalize_record(attempt, llm_provider, exam_style)
            store.record(attempt["question_id"], param_hash, canonical_hash, phash)
            verified.append(record)
        else:
            rejected.append(
                {
                    "question_id": last_attempt["question_id"],
                    "family": Family.FOLD_PUNCH.value,
                    "params": last_attempt["realized_params"],
                    "correct_letter": last_attempt["correct_letter"],
                    "rejection_reason": last_reason,
                    "image_dir": os.path.join("images", last_attempt["question_id"]),
                }
            )

    return verified, rejected

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an upgraded (fold-space-v2 + shapes) fold_punch batch")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    llm_provider = get_llm_provider()
    store = DedupStore(default_store_path(DEFAULT_DATA_DIR))
    start = time.time()
    verified, rejected = generate_upgraded_gated_batch(args.count, args.seed, DEFAULT_DATA_DIR, store, llm_provider)
    elapsed = time.time() - start

    with open(args.output, "a") as f:
        for record in verified:
            f.write(record.model_dump_json() + "\n")

    rejected_path = os.path.join(DEFAULT_DATA_DIR, "records", "points_rejected.jsonl")
    with open(rejected_path, "a") as f:
        for entry in rejected:
            f.write(json.dumps(entry, default=str) + "\n")

    difficulty_counts: dict[int, int] = defaultdict(int)
    subtype_difficulty_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    rule_counts: dict[str, int] = defaultdict(int)
    verified_image_paths = []
    verified_dicts = []
    for record in verified:
        difficulty_counts[record.difficulty] += 1
        subtype_difficulty_counts[record.sub_type][record.difficulty] += 1
        for rule in record.distractor_rules:
            rule_counts[rule] += 1
        verified_image_paths.append(os.path.join(DEFAULT_DATA_DIR, record.image_paths.options[record.correct_option]))
        verified_dicts.append(json.loads(record.model_dump_json()))

    rejected_duplicate = sum(1 for r in rejected if r["rejection_reason"] == "duplicate")
    rejected_quality = sum(1 for r in rejected if r["rejection_reason"] == "quality_gate_failed")

    scorecard = build_scorecard(
        attempted=args.count,
        verified=len(verified),
        flagged_vlm_disagreement=0,
        rejected_duplicate=rejected_duplicate,
        rejected_quality=rejected_quality,
        difficulty_counts=dict(difficulty_counts),
        subtype_difficulty_counts={k: dict(v) for k, v in subtype_difficulty_counts.items()},
        rule_counts=dict(rule_counts),
        elapsed_seconds=elapsed,
    )
    reports_dir = os.path.join(DEFAULT_DATA_DIR, "reports")
    json_path, html_path = write_report(scorecard, reports_dir, verified_image_paths=verified_image_paths, label="upgraded")

    gate_passed, gate_reasons = check_diversity_gate(verified_dicts, args.count, len(verified))
    if not gate_passed:
        print(f"DIVERSITY GATE WARNING: {'; '.join(gate_reasons)}")

    print(
        f"Upgraded batch: {len(verified)}/{args.count} verified "
        f"(duplicate={rejected_duplicate}, quality={rejected_quality}) -> {args.output}"
    )
    print(f"Scorecard: {html_path}")

if __name__ == "__main__":
    main()
