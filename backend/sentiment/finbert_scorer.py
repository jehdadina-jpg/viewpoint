"""
FinBERT (ProsusAI/finbert) - the financial-domain transformer.

Araci (2019), "FinBERT: Financial Sentiment Analysis with Pre-trained Language
Models". BERT fine-tuned on the Financial PhraseBank, so it knows that
"guidance cut" is bad and "buyback" is good - the domain knowledge VADER lacks.

This is the same model Colasanto, Grilli, Santoro & Villani (2022) use to
build their Black-Litterman view ("BERT's sentiment score for portfolio
optimization: a fine-tuned view in Black and Litterman model").

Output mapping (3-class softmax -> scalar):
    score      = P(positive) - P(negative)          in [-1, +1]
    confidence = max(P(pos), P(neg), P(neu))         in [1/3, 1]

Why not argmax? A headline at P=(0.45 pos, 0.40 neg, 0.15 neu) is genuinely
ambiguous; argmax says "+1 positive", the expectation says "+0.05" - which is
what we want feeding a portfolio view.

If torch/transformers or the model weights are unavailable the scorer reports
`available=False` and the ensemble proceeds without it (config:
`sentiment.finbert_fallback_ok`). The API surfaces which members were active.
"""
from __future__ import annotations

import logging

from sentiment.base import SentimentResult, SentimentScorer, clamp

log = logging.getLogger(__name__)


class FinBertScorer(SentimentScorer):
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, max_len: int = 64) -> None:
        self._batch = batch_size
        self._max_len = max_len
        self._ok = False
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
            self._pos = next(i for i, l in id2label.items() if l.startswith("pos"))
            self._neg = next(i for i, l in id2label.items() if l.startswith("neg"))
            self._ok = True
            log.info("FinBERT loaded (%s)", model_name)
        except Exception as e:  # noqa: BLE001
            log.warning("FinBERT unavailable (%s). Ensemble will run without it.", e)

    @property
    def available(self) -> bool:
        return self._ok

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not self._ok:
            raise RuntimeError("FinBERT not available")
        torch = self._torch
        out: list[SentimentResult] = []
        with torch.no_grad():
            for i in range(0, len(texts), self._batch):
                chunk = [t or "" for t in texts[i:i + self._batch]]
                enc = self._tok(chunk, padding=True, truncation=True, max_length=self._max_len, return_tensors="pt")
                probs = torch.softmax(self._model(**enc).logits, dim=-1).cpu().numpy()
                for p in probs:
                    score = clamp(float(p[self._pos] - p[self._neg]))
                    conf = clamp(float(p.max()), 0.0, 1.0)
                    out.append(SentimentResult(score=score, confidence=conf))
        return out
