"""Generate -> verify -> repair loop for fold_punch (TRD Section 3.1, Section 7).

A LangGraph state machine: sample params/geometry, generate distractors,
render images, ask the (swappable) VLM to independently solve the
question. If it agrees with the geometry engine's answer, hand off to the
LLM text generator and mark verified. If it disagrees, resample and retry
(up to MAX_ATTEMPTS total attempts); if it still disagrees, flag for human
review instead of silently dropping the record (Reliability NFR).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TypedDict
from uuid import uuid4

import numpy as np
from langgraph.graph import END, StateGraph

from distractors.fold_punch import Distractor, generate_distractors
from engine.fold_punch.difficulty import compute_difficulty
from engine.fold_punch.geometry import FoldPunchGeometry, serialize_fold_steps
from engine.fold_punch.sampler import FoldPunchParams, sample_fold_punch_geometry
from providers.base import LLMProvider, VLMProvider
from render.fold_punch import render_option_image, render_question_image, save_png
from schemas.record import (
    ExamStyle,
    Family,
    ImagePaths,
    Record,
    Source,
    Verification,
)
from textgen.generator import assign_option_letters, build_question_text

MAX_ATTEMPTS = 4  # 1 initial attempt + up to 3 retries (Reliability NFR)


class VerifyState(TypedDict, total=False):
    rng: np.random.Generator
    vlm_provider: VLMProvider
    llm_provider: LLMProvider
    work_dir: str

    params: FoldPunchParams
    geometry: FoldPunchGeometry

    distractors: list[Distractor]
    correct_letter: str
    distractor_letters: list[str]
    option_points_by_letter: dict[str, list]
    distractor_rule_by_letter: dict[str, str]

    question_image_path: str
    option_image_paths: dict[str, str]

    vlm_answer: str
    agree: bool
    attempts_log: list[dict]

    status: str
    text: object


def _generate_node(state: VerifyState) -> dict:
    params, geometry = sample_fold_punch_geometry(state["rng"])
    return {"params": params, "geometry": geometry}


def _distractors_node(state: VerifyState) -> dict:
    rng = state["rng"]
    geometry = state["geometry"]
    distractors = generate_distractors(
        geometry.answer_points, geometry.punch_points, geometry.steps, rng
    )
    letters = assign_option_letters(rng)
    correct_letter, distractor_letters = letters[0], letters[1:]

    option_points_by_letter = {correct_letter: geometry.answer_points}
    distractor_rule_by_letter: dict[str, str] = {}
    for letter, distractor in zip(distractor_letters, distractors):
        option_points_by_letter[letter] = distractor.points
        distractor_rule_by_letter[letter] = distractor.rule

    return {
        "distractors": distractors,
        "correct_letter": correct_letter,
        "distractor_letters": distractor_letters,
        "option_points_by_letter": option_points_by_letter,
        "distractor_rule_by_letter": distractor_rule_by_letter,
    }


def _render_node(state: VerifyState) -> dict:
    geometry = state["geometry"]
    work_dir = state["work_dir"]

    question_image = render_question_image(
        geometry.steps, geometry.final_polygon, geometry.punch_points
    )
    question_path = os.path.join(work_dir, "question.png")
    save_png(question_image, question_path)

    option_image_paths = {}
    for letter, points in state["option_points_by_letter"].items():
        path = os.path.join(work_dir, f"{letter}.png")
        save_png(render_option_image(points), path)
        option_image_paths[letter] = path

    return {"question_image_path": question_path, "option_image_paths": option_image_paths}


def _vlm_verify_node(state: VerifyState) -> dict:
    vlm_provider = state["vlm_provider"]
    vlm_answer = vlm_provider.solve(state["question_image_path"], state["option_image_paths"])
    agree = vlm_answer == state["correct_letter"]

    attempts_log = list(state.get("attempts_log", []))
    attempts_log.append(
        {
            "attempt": len(attempts_log) + 1,
            "vlm_model": vlm_provider.name,
            "vlm_answer": vlm_answer,
            "correct_letter": state["correct_letter"],
            "agree": agree,
        }
    )
    return {"vlm_answer": vlm_answer, "agree": agree, "attempts_log": attempts_log}


def _route_after_verify(state: VerifyState) -> str:
    if state["agree"]:
        return "text_gen"
    if len(state.get("attempts_log", [])) >= MAX_ATTEMPTS:
        return "flag"
    return "generate"


def _text_gen_node(state: VerifyState) -> dict:
    llm_provider = state["llm_provider"]
    distractor_rules_in_order = [
        state["distractor_rule_by_letter"][letter] for letter in state["distractor_letters"]
    ]
    text = build_question_text(
        llm_provider,
        asdict(state["params"]),
        state["correct_letter"],
        state["distractor_letters"],
        distractor_rules_in_order,
    )
    return {"status": "verified", "text": text}


def _flag_node(state: VerifyState) -> dict:
    return {"status": "flagged"}


def build_verify_graph():
    graph = StateGraph(VerifyState)
    graph.add_node("generate", _generate_node)
    graph.add_node("distractors", _distractors_node)
    graph.add_node("render", _render_node)
    graph.add_node("vlm_verify", _vlm_verify_node)
    graph.add_node("text_gen", _text_gen_node)
    graph.add_node("flag", _flag_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "distractors")
    graph.add_edge("distractors", "render")
    graph.add_edge("render", "vlm_verify")
    graph.add_conditional_edges(
        "vlm_verify",
        _route_after_verify,
        {"text_gen": "text_gen", "generate": "generate", "flag": "flag"},
    )
    graph.add_edge("text_gen", END)
    graph.add_edge("flag", END)
    return graph.compile()


_COMPILED_GRAPH = None


def _get_compiled_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_verify_graph()
    return _COMPILED_GRAPH


def _sub_type(params: FoldPunchParams) -> str:
    return "_".join(params.axis_sequence) + f"_{params.punch_count}punch"


def run_fold_punch_pipeline(
    rng: np.random.Generator,
    vlm_provider: VLMProvider,
    llm_provider: LLMProvider,
    data_dir: str,
    exam_style: ExamStyle = ExamStyle.GENERIC,
) -> tuple[Record | None, dict | None]:
    """Runs one question through generate -> verify -> repair.

    Returns (record, None) if verified, or (None, flagged_entry) if it was
    flagged for human review after MAX_ATTEMPTS. Never returns (None, None)
    — a result always exists, matching the "no silent drops" reliability
    requirement.
    """
    graph = _get_compiled_graph()
    work_dir = tempfile.mkdtemp(prefix="fold_punch_")
    try:
        initial_state: VerifyState = {
            "rng": rng,
            "vlm_provider": vlm_provider,
            "llm_provider": llm_provider,
            "work_dir": work_dir,
        }
        final_state = graph.invoke(initial_state, config={"recursion_limit": 100})

        question_id = str(uuid4())
        image_dir = os.path.join(data_dir, "images", question_id)
        os.makedirs(image_dir, exist_ok=True)

        # Stored image_paths are relative to data_dir (portable across
        # machines/deployments) — join with the configured data dir at
        # read time, e.g. in the retrieval API.
        question_dest = os.path.join(image_dir, "question.png")
        question_rel = os.path.join("images", question_id, "question.png")
        shutil.copy2(final_state["question_image_path"], question_dest)

        option_dests = {}
        for letter, src in final_state["option_image_paths"].items():
            dest = os.path.join(image_dir, f"{letter}.png")
            shutil.copy2(src, dest)
            option_dests[letter] = os.path.join("images", question_id, f"{letter}.png")

        params: FoldPunchParams = final_state["params"]
        geometry: FoldPunchGeometry = final_state["geometry"]
        attempts_log = final_state["attempts_log"]

        # Full realized params (not just the recipe) so any record can be
        # re-rendered byte-for-byte from `params` alone, without needing the
        # original rng draw sequence (FR-12).
        realized_params = {
            **asdict(params),
            "fold_steps": serialize_fold_steps(geometry.steps),
            "punch_points": [list(p) for p in geometry.punch_points],
        }

        if final_state["status"] == "verified":
            distractor_rules_in_order = [
                final_state["distractor_rule_by_letter"][letter]
                for letter in final_state["distractor_letters"]
            ]
            difficulty = compute_difficulty(
                params.fold_count, params.punch_count, params.axis_sequence, distractor_rules_in_order
            )
            record = Record(
                question_id=question_id,
                family=Family.FOLD_PUNCH,
                sub_type=_sub_type(params),
                difficulty=difficulty,
                params=realized_params,
                image_paths=ImagePaths(question=question_rel, options=option_dests),
                correct_option=final_state["correct_letter"],
                distractor_rules=distractor_rules_in_order,
                exam_style=exam_style,
                text=final_state["text"],
                tags=["fold_punch", *params.axis_sequence],
                embedding_text=f"{final_state['text'].stem} {' '.join(params.axis_sequence)}",
                verification=Verification(
                    vlm_model=vlm_provider.name,
                    vlm_answer=final_state["vlm_answer"],
                    agree=True,
                    verified_at=datetime.now(timezone.utc),
                ),
                source=Source.SYNTHETIC,
            )
            return record, None

        flagged_entry = {
            "question_id": question_id,
            "family": Family.FOLD_PUNCH.value,
            "params": realized_params,
            "correct_letter": final_state["correct_letter"],
            "distractor_rule_by_letter": final_state["distractor_rule_by_letter"],
            "attempts_log": attempts_log,
            "image_dir": os.path.join("images", question_id),
            "flagged_at": datetime.now(timezone.utc).isoformat(),
        }
        return None, flagged_entry
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
