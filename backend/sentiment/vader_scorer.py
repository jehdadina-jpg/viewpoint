"""
VADER (Valence Aware Dictionary and sEntiment Reasoner) - ensemble baseline.

Hutto & Gilbert (2014). Rule-based, lexicon-driven, tuned for social media.
Fast (thousands of headlines/sec), no model download, but NOT finance-aware:
"shares fall sharply" and "debt load grows" may score near-neutral because the
lexicon has no notion of what is good or bad for a stock. That bias is exactly
why we ensemble it with FinBERT + a finance dictionary rather than use it alone.

Output: VADER's `compound` score is already in [-1, +1]. Confidence is derived
from how far the compound is from zero and how much of the text was non-neutral
(1 - neu share) - a headline VADER finds entirely neutral gives us little
information, so it should carry little weight in the ensemble.
"""
from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment.base import SentimentResult, SentimentScorer, clamp


class VaderScorer(SentimentScorer):
    name = "vader"

    def __init__(self) -> None:
        self._sia = SentimentIntensityAnalyzer()

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        out = []
        for t in texts:
            p = self._sia.polarity_scores(t or "")
            compound = clamp(p["compound"])
            # confidence: blend of polarity magnitude and non-neutral coverage
            conf = clamp(0.5 * abs(compound) + 0.5 * (1.0 - p["neu"]), 0.0, 1.0)
            out.append(SentimentResult(score=compound, confidence=conf))
        return out
