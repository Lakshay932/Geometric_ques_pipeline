"""Canned-output stub providers. Default until a real backend is wired in."""
from __future__ import annotations

from .base import LLMProvider, VLMProvider


class StubVLMProvider(VLMProvider):
    name = "stub-vlm"

    def solve(self, question_image_path: str, option_image_paths: dict[str, str]) -> str:
        return "A"


class StubLLMProvider(LLMProvider):
    name = "stub-llm"

    def generate_text(
        self,
        params: dict,
        correct_option: str,
        distractor_rules: list[str],
    ) -> dict:
        return {
            "stem": "Stub question stem.",
            "option_labels": {label: f"Option {label}" for label in ("A", "B", "C", "D")},
            "explanation": "Stub explanation.",
        }
