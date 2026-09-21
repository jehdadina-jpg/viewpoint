"""
Loughran-McDonald finance dictionary scorer - third, fully offline ensemble member.

Loughran & McDonald (2011), "When Is a Liability Not a Liability? Textual
Analysis, Dictionaries, and 10-Ks", Journal of Finance. They showed that
general-purpose sentiment lexicons misclassify ~3/4 of "negative" words in
financial text (e.g. "liability", "tax", "cost" are not bad news). Their
finance-specific word lists remain the standard pre-LLM baseline (and are the
kind of dictionary approach Creamer (2015) builds on).

We embed a compact subset of the LM positive/negative lists here so the model
needs no download. Score = (pos - neg) / (pos + neg + 1) with simple negation
handling; confidence grows with the number of dictionary hits.

Role in the ensemble: a cheap, transparent, finance-aware vote that is
uncorrelated with VADER's general lexicon and provides a sanity check on
FinBERT. If FinBERT is unavailable, VADER + LM still form a two-model ensemble.
"""
from __future__ import annotations

import re

from sentiment.base import SentimentResult, SentimentScorer, clamp

# Compact subset of Loughran-McDonald (2011) lists, lower-cased stems.
_POS = {
    "achieve", "achieved", "achievement", "advance", "advances", "advantage", "advantageous", "attractive",
    "beat", "beats", "benefit", "benefits", "best", "better", "boost", "boosted", "breakthrough", "bullish",
    "cheer", "cheers", "climb", "climbs", "confident", "deliver", "delivered", "delivers", "efficient",
    "enhance", "enhanced", "exceed", "exceeded", "exceeds", "excellent", "expand", "expands", "expansion",
    "favorable", "gain", "gained", "gains", "good", "great", "greater", "grow", "grows", "growth", "high",
    "higher", "hike", "hiked", "improve", "improved", "improvement", "improving", "increase", "increased",
    "innovative", "jump", "jumps", "leader", "leading", "momentum", "opportunity", "optimistic", "outperform",
    "outperformed", "positive", "profit", "profitable", "profits", "progress", "raise", "raised", "raises",
    "rally", "rallies", "rebound", "record", "resilient", "robust", "soar", "soars", "solid", "strength",
    "strong", "stronger", "succeed", "success", "successful", "surge", "surged", "surges", "surpass",
    "surpassed", "upbeat", "upgrade", "upgraded", "upside", "win", "wins", "won", "buyback", "dividend",
}
_NEG = {
    "adverse", "adversely", "against", "alleged", "allegation", "allegations", "bankrupt", "bankruptcy",
    "bearish", "breach", "burden", "collapse", "concern", "concerns", "crisis", "cut", "cuts", "damage",
    "decline", "declined", "declines", "decrease", "decreased", "default", "deficit", "delay", "delayed",
    "deteriorate", "deteriorating", "difficult", "disappoint", "disappointed", "disappointing", "disruption",
    "downgrade", "downgraded", "downturn", "drop", "dropped", "drops", "fail", "failed", "failure", "fall",
    "falls", "fell", "fine", "fined", "fraud", "halt", "halted", "hurt", "impairment", "investigation",
    "lawsuit", "layoff", "layoffs", "litigation", "lose", "loses", "loss", "losses", "lost", "lower",
    "miss", "missed", "misses", "negative", "penalty", "plunge", "plunges", "poor", "pressure", "probe",
    "problem", "problems", "recall", "recession", "risk", "risks", "scandal", "sell", "sharply", "shortfall",
    "slash", "slashed", "slide", "slides", "slow", "slowdown", "slowing", "slump", "struggle", "struggling",
    "sue", "sued", "suspend", "suspended", "tumble", "tumbles", "uncertain", "uncertainty", "underperform",
    "unfavorable", "warn", "warning", "warns", "weak", "weaken", "weaker", "weakness", "worse", "worst",
    "writedown", "write-off",
}
_NEGATORS = {"not", "no", "never", "without", "fails", "failed", "isn't", "wasn't", "don't", "doesn't", "didn't"}
_TOKEN = re.compile(r"[a-z][a-z\-']+")


class LexiconScorer(SentimentScorer):
    name = "lexicon"

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        out = []
        for t in texts:
            toks = _TOKEN.findall((t or "").lower())
            pos = neg = 0.0
            for i, w in enumerate(toks):
                negated = any(x in _NEGATORS for x in toks[max(0, i - 3):i])
                if w in _POS:
                    if negated:
                        neg += 1
                    else:
                        pos += 1
                elif w in _NEG:
                    if negated:
                        pos += 1
                    else:
                        neg += 1
            hits = pos + neg
            score = clamp((pos - neg) / (hits + 1.0))
            conf = clamp(1.0 - 1.0 / (1.0 + hits), 0.0, 1.0)  # 0 hits -> 0, 1 -> .5, 3 -> .75
            out.append(SentimentResult(score=score, confidence=conf))
        return out
