"""
Common interface for every ensemble member.

A scorer maps a headline to (score, confidence):
    score      in [-1, +1]   negative .. positive
    confidence in [ 0,  1]   how sure the model is (used for confidence-weighting)

Keeping this interface tiny is what lets the aggregator treat VADER (lexicon +
rules), FinBERT (transformer) and the Loughran-McDonald dictionary uniformly,
and what makes it trivial to bolt on a fourth model (e.g. an LLM API) later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SentimentResult:
    score: float
    confidence: float


class SentimentScorer(ABC):
    name: str = "base"

    @abstractmethod
    def score_batch(self, texts: list[str]) -> list[SentimentResult]: ...

    def score(self, text: str) -> SentimentResult:
        return self.score_batch([text])[0]

    @property
    def available(self) -> bool:
        return True


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))
