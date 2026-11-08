"""Strict four-way ANLS endpoint evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import re
import string
from typing import Iterable


CORRECT = "CORRECT"
WRONG = "WRONG"
AMBIGUOUS = "AMBIGUOUS"
INVALID_OUTPUT = "INVALID_OUTPUT"


@dataclass(frozen=True)
class Evaluation:
    status: str
    anls: float
    normalized_prediction: str
    best_alias: str | None
    invalid_reason: str | None


def normalize(text: str) -> str:
    value = text.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[right_index - 1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def classify_output(raw: object, aliases: Iterable[str], *, terminated_eos: bool | None = None) -> Evaluation:
    if not isinstance(raw, str):
        return Evaluation(INVALID_OUTPUT, 0.0, "", None, "NON_STRING_OUTPUT")
    normalized = normalize(raw)
    if not normalized:
        return Evaluation(INVALID_OUTPUT, 0.0, normalized, None, "EMPTY_AFTER_NORMALIZATION")
    if terminated_eos is False:
        return Evaluation(INVALID_OUTPUT, 0.0, normalized, None, "EXPLICIT_NONTERMINATED_OUTPUT")

    best_score = 0.0
    best_alias: str | None = None
    for alias in aliases:
        normalized_alias = normalize(alias)
        if normalized == normalized_alias:
            return Evaluation(CORRECT, 1.0, normalized, alias, None)
        denominator = max(len(normalized), len(normalized_alias))
        raw_score = 1.0 - levenshtein(normalized, normalized_alias) / denominator
        score = raw_score if raw_score > 0.5 else 0.0
        if score > best_score:
            best_score, best_alias = score, alias
    if best_score == 0.0:
        return Evaluation(WRONG, 0.0, normalized, None, None)
    return Evaluation(AMBIGUOUS, best_score, normalized, best_alias, None)
