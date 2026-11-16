"""Verifiable reward functions for GRPO."""
from tasks import FORMAT_RE, extract_answer, is_correct


def _text(c):
    return c[0]["content"] if isinstance(c, list) else c


def correctness(completions, gold, **kwargs):
    return [float(is_correct(extract_answer(_text(c)), g)) for c, g in zip(completions, gold)]


def format_bonus(completions, **kwargs):
    return [0.2 if FORMAT_RE.match(_text(c)) else 0.0 for c in completions]


REWARDS = {"correctness": [correctness], "correctness_format": [correctness, format_bonus]}
