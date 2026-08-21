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
    default_stem: str = DEFAULT_STEM,
    rule_explanations: dict[str, str] = RULE_EXPLANATIONS,
    default_rule_explanation: str = _DEFAULT_RULE_EXPLANATION,
) -> QuestionText:
    llm_result = llm_provider.generate_text(params, correct_letter, distractor_rules_in_order)

    option_labels = {letter: f"Option {letter}" for letter in OPTION_LETTERS}
    option_labels.update(llm_result.get("option_labels", {}))

    explanation_parts = [llm_result.get("explanation") or f"Option {correct_letter} is correct."]
    for letter, rule in zip(distractor_letters, distractor_rules_in_order):
        explanation_parts.append(
            f"Option {letter}: {rule_explanations.get(rule, default_rule_explanation)}"
        )

    return QuestionText(
        stem=llm_result.get("stem") or default_stem,
        option_labels=option_labels,
        explanation=" ".join(explanation_parts),
    )
