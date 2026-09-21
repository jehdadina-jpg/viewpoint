"""
Ensemble aggregation: many headlines x several models -> one daily signal per stock.

Pipeline
--------
1. `score_headlines()`   run every enabled model over every unscored headline
                         (cached per (headline, model) in SQLite).
2. `EnsembleCombiner`    collapse the per-model scores of ONE headline into a
                         single score. Default = confidence-weighted average.
3. `aggregate_daily()`   for each (ticker, trading day) collapse the headlines
                         in a trailing window into a score + confidence +
                         dispersion, with exponential time-decay.

Why an ensemble?
----------------
Mantshimuli & Muteba Mwamba (2025), "Enhancing Portfolio Optimization with
Multi-LLM Sentiment Aggregation": any single sentiment model carries its own
bias (VADER ignores finance semantics, FinBERT is over-confident on short
text, dictionaries miss context). Averaging de-correlated models reduces that
noise - they report Sharpe 3.02 for the multi-model portfolio vs. the
benchmark. Their aggregation is a learned LSTM; ours is a weighted average
with the same interface, so a learned combiner can be dropped in (see
`LearnedCombiner`).

Dispersion as a confidence signal
---------------------------------
Two things can make the signal untrustworthy: the models disagree on the
same headline, or different headlines about the same stock disagree with each
other. We measure both and expose `dispersion`; views.py turns high dispersion
into a wide Omega (low view confidence) so the optimizer leans on the market
prior instead. This is the "fine-tuned view" idea in Colasanto et al. (2022).

=============================================================================
LOOK-AHEAD BIAS
=============================================================================
The sentiment for trading day D uses ONLY headlines with
    published_at <= D 21:00 UTC   (~ US cash close, 16:00-17:00 ET)
so a signal "as of close D" never contains anything printed after the close.
The backtest then applies weights chosen at close D to returns from D+1 on.
=============================================================================
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import db
from config import load_config
from sentiment.base import SentimentScorer
from sentiment.finbert_scorer import FinBertScorer
from sentiment.lexicon_scorer import LexiconScorer
from sentiment.vader_scorer import VaderScorer

log = logging.getLogger(__name__)

MARKET_CLOSE_UTC_HOUR = 21


# ----------------------------------------------------------------- scorers
def _has_scores(model: str) -> bool:
    with db.connect() as conn:
        return conn.execute("SELECT 1 FROM headline_scores WHERE model = ? LIMIT 1", (model,)).fetchone() is not None


def build_scorers() -> dict[str, tuple[SentimentScorer, float]]:
    """Instantiate enabled models from config -> {name: (scorer, weight)}."""
    cfg = load_config().sentiment
    scorers: dict[str, tuple[SentimentScorer, float]] = {}
    m = cfg.models
    if m.get("vader", {}).get("enabled", True):
        scorers["vader"] = (VaderScorer(), float(m.vader.get("weight", 1.0)))
    if m.get("lexicon", {}).get("enabled", True):
        scorers["lexicon"] = (LexiconScorer(), float(m.lexicon.get("weight", 0.5)))
    if m.get("finbert", {}).get("enabled", True):
        if db.unscored_news("finbert").empty and _has_scores("finbert"):
            # Nothing new to score: skip the ~40s model load but keep FinBERT
            # registered as an active ensemble member (its scores are cached).
            log.info("finbert: all headlines already scored - skipping model load")
            db.set_meta("sentiment_models", json.dumps(sorted(list(scorers) + ["finbert"])))
            return scorers
        fb = FinBertScorer(m.finbert.get("model_name", "ProsusAI/finbert"),
                           int(m.finbert.get("batch_size", 32)), int(m.finbert.get("max_len", 64)))
        if fb.available:
            scorers["finbert"] = (fb, float(m.finbert.get("weight", 1.5)))
        elif not cfg.get("finbert_fallback_ok", True):
            raise RuntimeError("FinBERT required but unavailable")
    if len(scorers) < 2:
        log.warning("Only %d sentiment model(s) active - ensemble degenerates to single model", len(scorers))
    db.set_meta("sentiment_models", json.dumps(sorted(scorers)))
    return scorers


def score_headlines(scorers: dict[str, tuple[SentimentScorer, float]] | None = None) -> dict[str, int]:
    """Score every headline that each model hasn't seen yet. Returns counts."""
    scorers = scorers or build_scorers()
    counts = {}
    for name, (scorer, _) in scorers.items():
        todo = db.unscored_news(name)
        if todo.empty:
            counts[name] = 0
            continue
        # Dedupe identical strings: cheap for real news, a huge win for sample data.
        uniq = todo["headline"].drop_duplicates().tolist()
        log.info("%s: scoring %d headlines (%d unique)", name, len(todo), len(uniq))
        res = dict(zip(uniq, scorer.score_batch(uniq)))
        rows = [(int(r.id), name, res[r.headline].score, res[r.headline].confidence)
                for r in todo.itertuples(index=False)]
        db.upsert_headline_scores(rows)
        counts[name] = len(rows)
    return counts


# --------------------------------------------------------------- combiners
class EnsembleCombiner(ABC):
    """Collapse per-model (score, confidence) for one headline into one score."""

    @abstractmethod
    def combine(self, scores: pd.DataFrame, confs: pd.DataFrame) -> pd.Series:
        """scores/confs: index=news_id, columns=model. Returns Series index=news_id."""


class WeightedAverageCombiner(EnsembleCombiner):
    """
    score_h = sum_m w_m * c_hm * s_hm / sum_m w_m * c_hm

    w_m  static model weight from config (FinBERT > VADER > lexicon by default)
    c_hm the model's own confidence on THIS headline - a model that shrugs
         ("neutral, 0.4") is down-weighted on that headline only.
    """

    def __init__(self, weights: dict[str, float]):
        self._w = weights

    def combine(self, scores: pd.DataFrame, confs: pd.DataFrame) -> pd.Series:
        w = pd.Series({m: self._w.get(m, 1.0) for m in scores.columns})
        eff = confs.fillna(0.0).mul(w, axis=1)
        num = (scores.fillna(0.0) * eff).sum(axis=1)
        den = eff.sum(axis=1).replace(0.0, np.nan)
        return (num / den).fillna(0.0)


class LearnedCombiner(EnsembleCombiner):
    """
    EXTENSION POINT (paper-backed): replace the static average with a model
    that learns how much to trust each member from realised forward returns -
    e.g. a logistic regression on [s_vader, s_finbert, s_lexicon, confs] or
    the LSTM aggregator of Mantshimuli & Muteba Mwamba (2025).

    Training must be walk-forward (fit only on data before each rebalance) to
    stay look-ahead safe. Not implemented; kept as a typed stub so swapping it
    in is a one-line change in `aggregate_daily()`.
    """

    def combine(self, scores: pd.DataFrame, confs: pd.DataFrame) -> pd.Series:  # pragma: no cover
        raise NotImplementedError("Train a walk-forward combiner and return per-headline scores here")


# --------------------------------------------------------- daily aggregation
def aggregate_daily(trading_days: pd.DatetimeIndex, tickers: list[str],
                    combiner: EnsembleCombiner | None = None) -> pd.DataFrame:
    """
    Build the daily_sentiment table.

    For ticker i and day D, with headlines h in (D - lookback, D 21:00 UTC]:
        age_h   = (D_close - t_h) in days
        w_h     = exp(-ln2 * age_h / half_life)              exponential decay
        score   = sum w_h s_h / sum w_h                       time-decayed mean
        disp_h  = sqrt(sum w_h (s_h - score)^2 / sum w_h)    headline disagreement
        disp_m  = sum w_h * std_m(s_hm) / sum w_h            model disagreement
        dispersion = 0.5 * (disp_h + disp_m)
        N_eff   = sum w_h                                      volume weighting
        confidence = (1 - dispersion) * (1 - exp(-N_eff / 3))
    """
    cfg = load_config().sentiment.aggregation
    half_life = float(cfg.get("half_life_days", 3.0))
    lookback = float(cfg.get("lookback_days", 10))
    min_head = int(cfg.get("min_headlines", 1))
    weights = {k: float(v.get("weight", 1.0)) for k, v in load_config().sentiment.models.items()}
    combiner = combiner or WeightedAverageCombiner(weights)

    hs = db.load_headline_scores()
    if hs.empty:
        raise RuntimeError("No headline scores - run score_headlines() first")

    scores = hs.pivot(index="news_id", columns="model", values="score")
    confs = hs.pivot(index="news_id", columns="model", values="confidence")
    per_head = pd.DataFrame({
        "score": combiner.combine(scores, confs),
        "model_disagreement": scores.std(axis=1, ddof=0).fillna(0.0),
    })
    for m in scores.columns:
        per_head[f"m_{m}"] = scores[m]
    meta = hs.drop_duplicates("news_id").set_index("news_id")[["ticker", "published_at"]]
    per_head = per_head.join(meta)
    model_cols = [c for c in per_head.columns if c.startswith("m_")]

    ln2 = np.log(2.0)
    rows = []
    for t in tickers:
        ph = per_head[per_head.ticker == t].sort_values("published_at")
        times = ph["published_at"].values.astype("datetime64[ns]")
        s = ph["score"].values
        dm = ph["model_disagreement"].values
        mvals = ph[model_cols].values
        for d in trading_days:
            # --- NO LOOK-AHEAD: only headlines at or before this day's close
            close = (pd.Timestamp(d) + pd.Timedelta(hours=MARKET_CLOSE_UTC_HOUR)).tz_localize("UTC")
            close64 = np.datetime64(close.tz_convert("UTC").tz_localize(None), "ns")
            start64 = close64 - np.timedelta64(int(lookback * 24 * 3600), "s")
            lo = np.searchsorted(times, start64, side="right")
            hi = np.searchsorted(times, close64, side="right")
            n = hi - lo
            if n < min_head:
                rows.append((t, d, np.nan, np.nan, np.nan, int(n), "{}"))
                continue
            age_days = (close64 - times[lo:hi]).astype("timedelta64[s]").astype(float) / 86400.0
            w = np.exp(-ln2 * age_days / half_life)
            wsum = w.sum()
            sc = float((w * s[lo:hi]).sum() / wsum)
            disp_h = float(np.sqrt((w * (s[lo:hi] - sc) ** 2).sum() / wsum))
            disp_m = float((w * dm[lo:hi]).sum() / wsum)
            dispersion = 0.5 * (disp_h + disp_m)
            n_eff = float(wsum)
            conf = float(max(0.0, 1.0 - min(dispersion, 1.0)) * (1.0 - np.exp(-n_eff / 3.0)))
            per_model = {}
            for j, c in enumerate(model_cols):
                col = mvals[lo:hi, j]
                mask = ~np.isnan(col)
                if mask.any():
                    per_model[c[2:]] = round(float((w[mask] * col[mask]).sum() / w[mask].sum()), 4)
            rows.append((t, d, sc, conf, dispersion, int(n), json.dumps(per_model)))

    out = pd.DataFrame(rows, columns=["ticker", "date", "score", "confidence", "dispersion", "n_headlines", "model_scores"])
    db.replace_daily_sentiment(out)
    log.info("daily_sentiment: %d rows (%d tickers x %d days)", len(out), len(tickers), len(trading_days))
    return out


def sentiment_matrix(daily: pd.DataFrame, col: str = "score") -> pd.DataFrame:
    return daily.pivot(index="date", columns="ticker", values=col).sort_index()
