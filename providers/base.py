"""Abstract interfaces for the two swappable model roles (TRD Section 7).

Concrete VLM/LLM backends (OpenRouter-hosted models, self-hosted, etc.)
implement these interfaces. Callers (verify/, textgen/) depend only on
this interface, never on a specific provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VLMProvider(ABC):
    """Vision-language model used to independently verify a rendered question."""

    name: str

    @abstractmethod
    def solve(self, question_image_path: str, option_image_paths: dict[str, str]) -> str:
        """Return the option label (e.g. "A") the VLM believes is correct."""


class LLMProvider(ABC):
    """LLM used to write stem, option labels, and explanation from a structured record."""

    name: str

    @abstractmethod
    def generate_text(
        self,
        params: dict,
        correct_option: str,
        distractor_rules: list[str],
    ) -> dict:
        """Return {"stem": str, "option_labels": dict[str, str], "explanation": str}."""
