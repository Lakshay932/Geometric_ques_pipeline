from __future__ import annotations

from abc import ABC, abstractmethod

class VLMProvider(ABC):

    name: str

    @abstractmethod
    def solve(self, question_image_path: str, option_image_paths: dict[str, str]) -> str:
        pass

class LLMProvider(ABC):

    name: str

    @abstractmethod
    def generate_text(
        self,
        params: dict,
        correct_option: str,
        distractor_rules: list[str],
    ) -> dict:
        pass
