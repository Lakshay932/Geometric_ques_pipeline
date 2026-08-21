from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from uuid import uuid4

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from distractors.mirror_water import RULE_POOL_V2, generate_distractors
from engine.mirror_water.difficulty import compute_difficulty
from engine.mirror_water.figures import generate_random_figure
from engine.mirror_water.geometry import (
    AXIS_OFFSET_RANGE,
    FIGURE_IDS,
    MIRROR_AXES,
    generate_from_polygon,
    make_figure_polygon,
    mirror_line_for_axis,
)
from evaluate.dedup import (
    PHASH_DUPLICATE_THRESHOLD,
    canonical_asymmetric_geometry_hash,
    canonical_param_hash,
    perceptual_hash,
)
from evaluate.quality import correct_answer_unique, mirror_water_solvability_check, options_pairwise_distinct, render_sanity
from evaluate.report import build_scorecard, check_diversity_gate, write_report
from evaluate.store import DedupStore
from providers.factory import get_llm_provider
from render.mirror_water import render_option_image, render_question_image, save_png
from schemas.record import ExamStyle, Family, ImagePaths, Record, Source, Verification
from textgen.generator import assign_option_letters, build_question_text

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_OUTPUT = os.path.join(DEFAULT_DATA_DIR, "records", "mirror_water.jsonl")

MIRROR_WATER_STORE_FILENAME = "dedup_store_mirror_water.jsonl"
MAX_ATTEMPTS = 4

PROCEDURAL_FIGURE_PROBABILITY = 0.5

DEFAULT_STEM = "A figure is shown next to a mirror line (dashed). Which option correctly shows its reflection?"
RULE_EXPLANATIONS: dict[str, str] = {
    "identical_copy": "This option shows the original figure, unreflected.",
    "rotated_180": "This option rotates the figure 180 degrees instead of reflecting it.",
    "wrong_axis": "This option reflects the figure across a different mirror line than the one shown.",
    "shifted_reflection": "This option shows the correct reflection shifted slightly out of position.",
    "rotated_90": "This option rotates the figure 90 degrees instead of reflecting it.",
    "scaled_reflection": "This option shows the correct reflection at the wrong size.",
    "glide_reflection": "This option shows the correct reflection shifted along the mirror line itself.",
    "partial_reflection": "This option only reflects part of the figure, leaving the rest unreflected.",
}
_DEFAULT_RULE_EXPLANATION = "This option applies an incorrect reflection."

def default_mirror_water_store_path(data_dir: str) -> str:
    return os.path.join(data_dir, MIRROR_WATER_STORE_FILENAME)

def _diversity_cell_key(record: dict) -> tuple:
    params = record["params"]
    return (params["figure_source"], params["axis"])

def sample_figure(rng: np.random.Generator) -> tuple[str, Polygon, str]:
    if rng.random() < PROCEDURAL_FIGURE_PROBABILITY:
        return "procedural", generate_random_figure(rng), "procedural"
    figure_id = FIGURE_IDS[int(rng.integers(0, len(FIGURE_IDS)))]
    return figure_id, make_figure_polygon(figure_id), "hand_designed"

def sample_axis_and_offset(rng: np.random.Generator) -> tuple[str, float]:
    axis = MIRROR_AXES[int(rng.integers(0, len(MIRROR_AXES)))]
    if axis in ("vertical", "horizontal"):
        offset = float(rng.uniform(*AXIS_OFFSET_RANGE))
    else:
        offset = 0.0
    return axis, offset

def _build_attempt(rng: np.random.Generator, data_dir: str) -> dict:
    figure_id, original_polygon, figure_source = sample_figure(rng)
    axis, offset = sample_axis_and_offset(rng)
    geometry = generate_from_polygon(original_polygon, axis, offset, figure_id=figure_id)
    mirror_line = mirror_line_for_axis(axis, offset)

    distractors = generate_distractors(
        geometry.original, geometry.answer, axis, rng, rule_pool=RULE_POOL_V2, axis_offset=offset
    )
    letters = assign_option_letters(rng)
    correct_letter, distractor_letters = letters[0], letters[1:]

    option_shapes_by_letter = {correct_letter: geometry.answer}
    distractor_rule_by_letter: dict[str, str] = {}
    for letter, distractor in zip(distractor_letters, distractors):
        option_shapes_by_letter[letter] = distractor.shape
        distractor_rule_by_letter[letter] = distractor.rule

    question_id = str(uuid4())
    image_dir = os.path.join(data_dir, "images", question_id)
    os.makedirs(image_dir, exist_ok=True)

    question_path = os.path.join(image_dir, "question.png")
    save_png(render_question_image(geometry.original, mirror_line), question_path)
    question_rel = os.path.join("images", question_id, "question.png")

    option_paths = {}
    option_dests = {}
    for letter, shape in option_shapes_by_letter.items():
        path = os.path.join(image_dir, f"{letter}.png")
        save_png(render_option_image(shape, mirror_line), path)
        option_paths[letter] = path
        option_dests[letter] = os.path.join("images", question_id, f"{letter}.png")

    realized_params = {
        "figure_id": figure_id,
        "figure_source": figure_source,
        "axis": axis,
        "axis_offset": offset,
        "original_points": [list(p) for p in original_polygon.exterior.coords[:-1]],
    }

    return {
        "figure_source": figure_source,
        "axis": axis,
        "offset": offset,
        "geometry": geometry,
        "correct_letter": correct_letter,
        "distractor_letters": distractor_letters,
        "option_shapes_by_letter": option_shapes_by_letter,
        "distractor_rule_by_letter": distractor_rule_by_letter,
        "question_id": question_id,
        "question_path": question_path,
        "question_rel": question_rel,
        "option_paths": option_paths,
        "option_dests": option_dests,
        "realized_params": realized_params,
    }

def _quality_ok(attempt: dict) -> bool:
    option_points_by_letter = {
        letter: list(shape.exterior.coords)[:-1]
        for letter, shape in attempt["option_shapes_by_letter"].items()
    }
    correct_points = option_points_by_letter[attempt["correct_letter"]]
    distractor_points = [option_points_by_letter[letter] for letter in attempt["distractor_letters"]]
    realized = attempt["realized_params"]
    return (
        correct_answer_unique(correct_points, distractor_points)
        and options_pairwise_distinct(list(option_points_by_letter.values()))
        and render_sanity(attempt["question_path"])
        and all(render_sanity(p) for p in attempt["option_paths"].values())
        and mirror_water_solvability_check(
            [tuple(p) for p in realized["original_points"]],
            realized["axis"],
            realized["axis_offset"],
            correct_points,
        )
    )

def _finalize_record(attempt: dict, llm_provider, exam_style: ExamStyle) -> Record:
    figure_id = attempt["realized_params"]["figure_id"]
    figure_source = attempt["figure_source"]
    axis = attempt["axis"]
    distractor_rules_in_order = [
        attempt["distractor_rule_by_letter"][letter] for letter in attempt["distractor_letters"]
    ]
    text = build_question_text(
        llm_provider,
        {"figure_id": figure_id, "axis": axis},
        attempt["correct_letter"],
        attempt["distractor_letters"],
        distractor_rules_in_order,
        default_stem=DEFAULT_STEM,
        rule_explanations=RULE_EXPLANATIONS,
        default_rule_explanation=_DEFAULT_RULE_EXPLANATION,
    )
    difficulty = compute_difficulty(
        axis, distractor_rules_in_order, figure_source=figure_source, axis_offset=attempt["offset"]
    )
    return Record(
        question_id=attempt["question_id"],
        family=Family.MIRROR_WATER,
        sub_type=f"{figure_id}_{axis}",
        difficulty=difficulty,
        params=attempt["realized_params"],
        image_paths=ImagePaths(question=attempt["question_rel"], options=attempt["option_dests"]),
        correct_option=attempt["correct_letter"],
        distractor_rules=distractor_rules_in_order,
        exam_style=exam_style,
        text=text,
        tags=["mirror_water", figure_source, figure_id, axis],
        embedding_text=f"{text.stem} {figure_id} {axis}",
        verification=Verification(vlm_model=None, vlm_answer=None, agree=None, verified_at=None),
        source=Source.SYNTHETIC,
    )

def generate_one(rng: np.random.Generator, llm_provider, data_dir: str, exam_style=ExamStyle.GENERIC) -> Record:
    return _finalize_record(_build_attempt(rng, data_dir), llm_provider, exam_style)

def generate_mirror_water_gated_batch(
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
            correct_points = list(
                attempt["option_shapes_by_letter"][attempt["correct_letter"]].exterior.coords
            )[:-1]
            geometry_hash = canonical_asymmetric_geometry_hash(correct_points)
            correct_image = Image.open(attempt["option_paths"][attempt["correct_letter"]])
            phash = perceptual_hash(correct_image)

            is_duplicate = store.has_param_hash(param_hash) or store.has_canonical_hash(geometry_hash)
            if not is_duplicate:
                closest = store.closest_phash_distance(phash)
                is_duplicate = closest is not None and closest <= PHASH_DUPLICATE_THRESHOLD

            if is_duplicate:
                last_reason = "duplicate"
                continue
            if not _quality_ok(attempt):
                last_reason = "quality_gate_failed"
                continue

            winner = (attempt, param_hash, geometry_hash, phash)
            break

        if winner is not None:
            attempt, param_hash, geometry_hash, phash = winner
            record = _finalize_record(attempt, llm_provider, exam_style)
            store.record(attempt["question_id"], param_hash, geometry_hash, phash)
            verified.append(record)
        else:
            rejected.append(
                {
                    "question_id": last_attempt["question_id"],
                    "family": Family.MIRROR_WATER.value,
                    "params": last_attempt["realized_params"],
                    "correct_letter": last_attempt["correct_letter"],
                    "rejection_reason": last_reason,
                    "image_dir": os.path.join("images", last_attempt["question_id"]),
                }
            )

    return verified, rejected

def _run_ungated(args) -> None:
    llm_provider = get_llm_provider()
    rng = np.random.default_rng(args.seed)
    with open(args.output, "w") as f:
        for i in range(args.count):
            record = generate_one(rng, llm_provider, DEFAULT_DATA_DIR)
            f.write(record.model_dump_json() + "\n")
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{args.count}")
    print(f"Wrote {args.count} mirror_water records to {args.output}")

def _run_gated(args) -> None:
    llm_provider = get_llm_provider()
    store = DedupStore(default_mirror_water_store_path(DEFAULT_DATA_DIR))
    start = time.time()
    verified, rejected = generate_mirror_water_gated_batch(
        args.count, args.seed, DEFAULT_DATA_DIR, store, llm_provider
    )
    elapsed = time.time() - start

    with open(args.output, "a") as f:
        for record in verified:
            f.write(record.model_dump_json() + "\n")

    rejected_path = os.path.join(DEFAULT_DATA_DIR, "records", "mirror_water_rejected.jsonl")
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
    json_path, html_path = write_report(
        scorecard, reports_dir, verified_image_paths=verified_image_paths, label="mirror_water"
    )

    gate_passed, gate_reasons = check_diversity_gate(
        verified_dicts, args.count, len(verified), cell_key_fn=_diversity_cell_key
    )
    if not gate_passed:
        print(f"DIVERSITY GATE WARNING: {'; '.join(gate_reasons)}")

    print(
        f"Gated mirror_water batch: {len(verified)}/{args.count} verified "
        f"(duplicate={rejected_duplicate}, quality={rejected_quality}) -> {args.output}"
    )
    print(f"Scorecard: {html_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mirror_water batch")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gated",
        action="store_true",
        help="Run every record through the Phase M2 dedup+quality gates against the persistent mirror_water store.",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.gated:
        _run_gated(args)
    else:
        _run_ungated(args)

if __name__ == "__main__":
    main()
