"""
End-to-end build: data -> sentiment -> optimisation -> backtest -> results.json

    python -m pipeline.run            # incremental (uses SQLite caches)
    python -m pipeline.run --force    # re-pull prices/news
    python -m pipeline.run --skip-backtest

The API reads `storage/results.json` at startup; running this script is the
only "batch" step. Everything else (news feed, price history, live ticks) is
served straight from SQLite / memory.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

import db
from backtest.engine import run_backtest
from backtest.metrics import clean
from config import load_config, resolve_path
from data.news_fetcher import ensure_news
from data.price_fetcher import ensure_market_caps, ensure_prices, price_matrix
from optimization.optimizer import optimize_portfolio
from sentiment.aggregator import aggregate_daily, build_scorers, score_headlines, sentiment_matrix

log = logging.getLogger("pipeline")


def results_path() -> Path:
    return resolve_path(load_config().api.results_cache)


def current_portfolio(px: pd.DataFrame, caps: dict, sent: pd.DataFrame, disp: pd.DataFrame,
                      w_prev: pd.Series | None) -> dict:
    """Optimise as of the last available close = 'what we'd hold tomorrow'."""
    as_of = px.index[-1]
    rets = px.pct_change().dropna(how="all").loc[:as_of]
    res = optimize_portfolio(rets, caps, sent.loc[:as_of], disp.loc[:as_of], as_of,
                             last_prices=px.loc[as_of], use_views=True, seed=999, w_prev=w_prev)
    bench = optimize_portfolio(rets, caps, None, None, as_of, use_views=False)
    tickers = list(px.columns)
    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "weights": {t: float(res.weights[t]) for t in tickers},
        "benchmark_weights": {t: float(res.w_mkt[t]) for t in tickers},          # market cap prior
        "bl_no_view_weights": {t: float(bench.weights[t]) for t in tickers},     # constrained MV on pi only
        "prior_returns": {t: float(res.prior[t]) for t in tickers},
        "posterior_returns": {t: float(res.posterior[t]) for t in tickers},
        "views": [
            {"ticker": t, "z": res.views.z[t], "q": float(q), "confidence": res.views.confidence[t],
             "target_price": res.views.target_price.get(t), "last_price": float(px.loc[as_of, t])}
            for t, q in zip(res.views.view_tickers, res.views.Q)
        ],
        "sentiment_z": res.views.z,
    }


def run(force: bool = False, skip_backtest: bool = False) -> dict:
    t0 = time.time()
    cfg = load_config()
    db.init_db()

    log.info("[1/6] prices")
    px_long = ensure_prices(force=force)
    px = price_matrix(px_long)
    caps = ensure_market_caps()

    log.info("[2/6] news")
    ensure_news(px_long, force=force)

    log.info("[3/6] sentiment models")
    scorers = build_scorers()
    counts = score_headlines(scorers)
    log.info("scored: %s", counts)

    log.info("[4/6] daily aggregation")
    trading_days = px.loc[cfg.universe.start:].index
    daily = aggregate_daily(trading_days, list(cfg.universe.tickers))
    sent, disp = sentiment_matrix(daily, "score"), sentiment_matrix(daily, "dispersion")

    out: dict = {
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "universe": list(cfg.universe.tickers),
        "data_sources": {
            "prices": db.get_meta("price_source"),
            "market_caps": db.get_meta("market_cap_source"),
            "news": db.get_meta("news_provider"),
            "sentiment_models": json.loads(db.get_meta("sentiment_models") or "[]"),
            "synthetic_news_share": float(db.load_news()["is_synthetic"].mean()) if len(db.load_news()) else None,
        },
    }

    prev = None
    if not skip_backtest:
        log.info("[5/6] walk-forward backtest")
        bt = run_backtest(px, caps, sent, disp)
        out["backtest"] = bt
        last_w = bt["weights_history"][-1]
        prev = pd.Series({t: last_w[t] for t in cfg.universe.tickers})

    log.info("[6/6] current portfolio")
    out["portfolio"] = current_portfolio(px, caps, sent, disp, prev)

    p = results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(out)), encoding="utf-8")
    log.info("results -> %s  (%.1fs)", p, time.time() - t0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-pull prices and news")
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(force=a.force, skip_backtest=a.skip_backtest)
