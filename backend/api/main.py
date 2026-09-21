"""
FastAPI layer.

    uvicorn api.main:app --reload --port 8000      (from backend/)

Batch results (backtest, current weights) come from storage/results.json,
built by `python -m pipeline.run`. If that file is missing the API builds it
on first start (slow the very first time: model download + scoring).

Row-level data (price history, news feed, daily sentiment) is read from
SQLite on demand. The WebSocket streams SIMULATED intraday ticks - see
`ws_ticks` - because we only hold end-of-day data.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import db
from config import load_config
from data.sample_data import COMPANY_NAMES, SECTORS
from sentiment.aggregator import WeightedAverageCombiner

log = logging.getLogger("api")

# --------------------------------------------------------------------- state
class State:
    results: dict = {}
    prices: pd.DataFrame = pd.DataFrame()      # long
    px: pd.DataFrame = pd.DataFrame()          # wide adj_close
    ohlc: dict[str, pd.DataFrame] = {}
    sentiment: pd.DataFrame = pd.DataFrame()   # daily_sentiment long
    tickers: list[str] = []


S = State()


def _results_path() -> Path:
    from pipeline.run import results_path
    return results_path()


def load_state() -> None:
    cfg = load_config()
    p = _results_path()
    if not p.exists():
        log.warning("results.json missing - running pipeline (first run can take several minutes)")
        from pipeline.run import run
        run()
    S.results = json.loads(p.read_text(encoding="utf-8"))
    S.tickers = list(S.results.get("universe") or cfg.universe.tickers)
    S.prices = db.load_prices(S.tickers)
    S.px = S.prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    S.ohlc = {t: g.set_index("date").sort_index() for t, g in S.prices.groupby("ticker")}
    S.sentiment = db.load_daily_sentiment()
    log.info("state loaded: %d tickers, prices to %s", len(S.tickers), S.px.index[-1].date())


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    load_state()
    yield


app = FastAPI(title="VIEWPOINT API - sentiment-driven Black-Litterman", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ------------------------------------------------------------------ helpers
def _f(x) -> float | None:
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return None


def _latest_sentiment(ticker: str) -> dict:
    s = S.sentiment[S.sentiment.ticker == ticker].dropna(subset=["score"])
    if s.empty:
        return {"score": None, "confidence": None, "dispersion": None, "n_headlines": 0, "model_scores": {}}
    r = s.iloc[-1]
    return {
        "score": _f(r.score), "confidence": _f(r.confidence), "dispersion": _f(r.dispersion),
        "n_headlines": int(r.n_headlines), "model_scores": json.loads(r.model_scores or "{}"),
        "date": r.date.strftime("%Y-%m-%d"),
    }


def _label(score: float | None) -> str:
    if score is None:
        return "NEU"
    return "POS" if score > 0.15 else "NEG" if score < -0.15 else "NEU"


def _require_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t not in S.tickers:
        raise HTTPException(404, f"{t} not in universe")
    return t


# ---------------------------------------------------------------- endpoints
@app.get("/meta")
def meta():
    cfg = load_config()
    return {
        "built_at": S.results.get("built_at"),
        "universe": S.tickers,
        "data_sources": S.results.get("data_sources"),
        "last_price_date": S.px.index[-1].strftime("%Y-%m-%d"),
        "backtest_window": {"start": cfg.universe.start, "end": cfg.universe.end},
        "black_litterman": dict(cfg.black_litterman),
        "optimizer": dict(cfg.optimizer),
        "backtest": dict(cfg.backtest),
        "sentiment": {"aggregation": dict(cfg.sentiment.aggregation),
                      "model_weights": {k: v.get("weight") for k, v in cfg.sentiment.models.items() if v.get("enabled")}},
    }


@app.get("/stocks")
def stocks():
    weights = S.results.get("portfolio", {}).get("weights", {})
    bench = S.results.get("portfolio", {}).get("benchmark_weights", {})
    out = []
    for t in S.tickers:
        c = S.px[t].dropna()
        last, prev = float(c.iloc[-1]), float(c.iloc[-2])
        sent = _latest_sentiment(t)
        out.append({
            "ticker": t, "name": COMPANY_NAMES.get(t, t), "sector": SECTORS.get(t, "n/a"),
            "last": last, "prev_close": prev, "change": last - prev, "pct_change": (last / prev - 1.0) * 100.0,
            "date": c.index[-1].strftime("%Y-%m-%d"),
            "sentiment": sent["score"], "confidence": sent["confidence"], "dispersion": sent["dispersion"],
            "n_headlines": sent["n_headlines"], "label": _label(sent["score"]),
            "weight": _f(weights.get(t)), "benchmark_weight": _f(bench.get(t)),
        })
    return {"as_of": S.px.index[-1].strftime("%Y-%m-%d"), "stocks": out}


@app.get("/stock/{ticker}/history")
def stock_history(ticker: str, days: int = Query(365, ge=30, le=2000)):
    t = _require_ticker(ticker)
    o = S.ohlc[t].tail(days)
    sent = S.sentiment[S.sentiment.ticker == t].set_index("date").reindex(o.index)
    candles = [
        {"time": d.strftime("%Y-%m-%d"), "open": _f(r.open), "high": _f(r.high), "low": _f(r.low),
         "close": _f(r.close), "volume": _f(r.volume)}
        for d, r in o.iterrows()
    ]
    sentiment = [
        {"time": d.strftime("%Y-%m-%d"), "score": _f(r.score), "confidence": _f(r.confidence),
         "dispersion": _f(r.dispersion), "n_headlines": int(r.n_headlines) if not pd.isna(r.n_headlines) else 0}
        for d, r in sent.iterrows()
    ]
    return {"ticker": t, "name": COMPANY_NAMES.get(t, t), "candles": candles, "sentiment": sentiment,
            "price_source": str(o.source.iloc[-1])}


@app.get("/stock/{ticker}/news")
def stock_news(ticker: str, limit: int = Query(40, ge=1, le=200)):
    t = _require_ticker(ticker)
    news = db.load_news(t, limit=limit)
    if news.empty:
        return {"ticker": t, "news": []}
    scores = db.load_headline_scores(news_ids=news["id"].tolist())
    cfg = load_config().sentiment.models
    weights = {k: float(v.get("weight", 1.0)) for k, v in cfg.items()}
    per_model = {}
    ens = pd.Series(dtype=float)
    if not scores.empty:
        sp = scores.pivot(index="news_id", columns="model", values="score")
        cp = scores.pivot(index="news_id", columns="model", values="confidence")
        ens = WeightedAverageCombiner(weights).combine(sp, cp)
        per_model = {int(i): {m: _f(v) for m, v in row.items() if not pd.isna(v)} for i, row in sp.iterrows()}
    out = []
    for r in news.itertuples(index=False):
        e = _f(ens.get(r.id)) if len(ens) else None
        out.append({
            "id": int(r.id), "headline": r.headline, "source": r.source, "url": r.url,
            "published_at": r.published_at.isoformat(), "provider": r.provider,
            "is_synthetic": bool(r.is_synthetic), "score": e, "label": _label(e),
            "model_scores": per_model.get(int(r.id), {}),
        })
    return {"ticker": t, "news": out}


@app.get("/portfolio/current")
def portfolio_current():
    p = S.results.get("portfolio")
    if not p:
        raise HTTPException(503, "portfolio not built")
    return p


@app.get("/portfolio/backtest")
def portfolio_backtest():
    bt = S.results.get("backtest")
    if not bt:
        raise HTTPException(503, "backtest not built")
    return bt


@app.get("/portfolio/stats")
def portfolio_stats():
    bt = S.results.get("backtest")
    if not bt:
        raise HTTPException(503, "backtest not built")
    return {
        "config": bt["config"],
        "net": bt["metrics"]["net"],
        "gross": bt["metrics"]["gross"],
        "regimes": {k: bt["regimes"][k] for k in ("high_vol", "low_vol", "n_days")},
    }


# ---------------------------------------------------------------- websocket
@app.websocket("/ws/ticks")
async def ws_ticks(ws: WebSocket):
    """
    *** SIMULATED LIVE TICKS ***
    We only have end-of-day prices. To give the dashboard a live feel each
    tick nudges every name with an Ornstein-Uhlenbeck-style jitter around its
    last real close:
        p_{t+1} = p_t + kappa (close - p_t) + close * sigma_tick * eps
    Every message carries `"simulated": true`. Do NOT read anything into these
    numbers; they exist purely for the cell-flash / ticker-tape demo.
    """
    await ws.accept()
    cfg = load_config()
    interval = float(cfg.api.get("tick_interval_seconds", 1.0))
    rng = np.random.default_rng()
    close = {t: float(S.px[t].dropna().iloc[-1]) for t in S.tickers}
    prev = {t: float(S.px[t].dropna().iloc[-2]) for t in S.tickers}
    cur = dict(close)
    # per-name tick vol proportional to realised daily vol (~1/25 of a day)
    dvol = {t: float(S.px[t].pct_change().tail(60).std()) / 25.0 for t in S.tickers}
    try:
        while True:
            ticks = []
            for t in S.tickers:
                cur[t] = cur[t] + 0.05 * (close[t] - cur[t]) + close[t] * dvol[t] * float(rng.standard_normal())
                ticks.append({
                    "ticker": t, "price": round(cur[t], 2), "change": round(cur[t] - prev[t], 2),
                    "pct_change": round((cur[t] / prev[t] - 1.0) * 100.0, 3),
                })
            await ws.send_json({"type": "ticks", "simulated": True,
                                "ts": pd.Timestamp.now(tz="UTC").isoformat(), "ticks": ticks})
            await asyncio.sleep(interval)
    except (WebSocketDisconnect, RuntimeError):
        return


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    c = load_config().api
    uvicorn.run("api.main:app", host=c.host, port=int(c.port), reload=False)
