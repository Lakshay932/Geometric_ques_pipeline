import numpy as np

from providers.stub import StubLLMProvider
from textgen.generator import OPTION_LETTERS, assign_option_letters, build_question_text


def test_build_question_text_uses_given_letters_and_explanations():
    provider = StubLLMProvider()
    rng = np.random.default_rng(1)
    letters = assign_option_letters(rng)
    correct_letter, distractor_letters = letters[0], letters[1:]
    rules = ["missing_hole", "shifted_hole", "mirrored_wrong"]

    text = build_question_text(provider, {}, correct_letter, distractor_letters, rules)

    assert text.stem
    for letter in distractor_letters:
        assert f"Option {letter}:" in text.explanation


def test_assign_option_letters_is_not_always_the_same_across_seeds():
    seen_first_letters = set()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        letters = assign_option_letters(rng)
        assert set(letters) == set(OPTION_LETTERS)
        seen_first_letters.add(letters[0])
    assert len(seen_first_letters) > 1
