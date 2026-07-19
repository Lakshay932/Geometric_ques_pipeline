"""LLM Text Generator (FR-8): stem, option labels, and explanation from an
already-correct structured record. Never touches the answer itself —
distractor tags become the explanation of each wrong option (TRD Section
3.1, step 6), templated deterministically so explanations stay accurate
and auditable regardless of which LLM backend is swapped in.
"""
from __future__ import annotations

import numpy as np

from providers.base import LLMProvider
from schemas.record import QuestionText

DEFAULT_STEM = (
    "A square sheet of paper is folded as shown and then punched. "
    "Which option correctly shows the paper when fully unfolded?"
)

RULE_EXPLANATIONS: dict[str, str] = {
    "missing_hole": "This option is missing one of the holes that the punch actually produced.",
    "extra_hole": "This option shows an extra hole that the punch would not have produced.",
    "wrong_symmetry_axis": "This option is missing the mirror image created by one of the folds.",
    "mirrored_wrong": "This option mirrors the whole hole pattern across the wrong axis.",
    "shifted_hole": "This option shows the holes shifted slightly away from their correct positions.",
}
_DEFAULT_RULE_EXPLANATION = "This option applies an incorrect fold transformation."

OPTION_LETTERS = ["A", "B", "C", "D"]


def assign_option_letters(rng: np.random.Generator) -> list[str]:
    letters = list(OPTION_LETTERS)
    rng.shuffle(letters)
    return letters


def build_question_text(
    llm_provider: LLMProvider,
    params: dict,
    correct_letter: str,
    distractor_letters: list[str],
    distractor_rules_in_order: list[str],
) -> QuestionText:
    """distractor_rules_in_order[i] corresponds to distractor_letters[i].

    correct_letter/distractor_letters must already be decided (via
    assign_option_letters) before this is called — letter assignment
    happens at render time, since the VLM verifier answers with a letter
    tied to a specific rendered option image, and that mapping must not
    change afterward.
    """
    llm_result = llm_provider.generate_text(params, correct_letter, distractor_rules_in_order)

    option_labels = {letter: f"Option {letter}" for letter in OPTION_LETTERS}
    option_labels.update(llm_result.get("option_labels", {}))

    explanation_parts = [llm_result.get("explanation") or f"Option {correct_letter} is correct."]
    for letter, rule in zip(distractor_letters, distractor_rules_in_order):
        explanation_parts.append(
            f"Option {letter}: {RULE_EXPLANATIONS.get(rule, _DEFAULT_RULE_EXPLANATION)}"
        )

    return QuestionText(
        stem=llm_result.get("stem") or DEFAULT_STEM,
        option_labels=option_labels,
        explanation=" ".join(explanation_parts),
    )
