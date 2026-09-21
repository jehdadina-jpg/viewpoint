"""
Walk-forward backtest: sentiment-tilted Black-Litterman vs. two benchmarks.

Portfolios
----------
  bl_sentiment  Black-Litterman with sentiment views (the strategy)
  market_cap    cap-weighted prior portfolio, rebalanced monthly
  equal_weight  1/N, rebalanced monthly

Mechanics
---------
* Rebalance on the last trading day of each month, D_k.
* Weights for D_k are computed from data with date <= D_k ONLY:
    - covariance from trailing 252 daily returns ending D_k
    - sentiment / dispersion matrices sliced `.loc[:D_k]`
    - the sentiment for D_k itself contains only headlines up to the close
      (see sentiment/aggregator.py)
  and applied to returns from D_k + 1 through D_{k+1}.  <- NO LOOK-AHEAD
* Within a holding period weights drift with prices (buy-and-hold), so the
  turnover at D_{k+1} is sum |w_target - w_drifted|, exactly what a trader
  would have to transact.
* Transaction costs: `transaction_cost_bps` (default 10 bp) charged on
  one-way turnover at each rebalance. Reported gross AND net. Recent work on
  sentiment-driven optimisation (e.g. the 2026 ScienceDirect study the club
  brief references - fill exact citation in README) stresses that monthly
  sentiment tilts can churn the book enough for costs to eat a large share of
  the paper alpha, so the net numbers are the ones that matter.

Regime split
------------
Realised 21-day volatility of the market-cap benchmark, annualised, split at
its median over the backtest -> each day is "high_vol" or "low_vol". Metrics
are recomputed on each subset. Sentiment signals are widely reported to work
differently in stressed vs. calm markets (Creamer 2015 finds the news effect
is high-frequency and regime-dependent), so a single full-sample Sharpe hides
the interesting part.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtest.metrics import compute_metrics, equity_curve
from config import load_config
from optimization.optimizer import cap_weights, optimize_portfolio

log = logging.getLogger(__name__)
PORTFOLIOS = ["bl_sentiment", "market_cap", "equal_weight"]


def month_end_rebalance_dates(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    return [pd.Timestamp(x) for x in idx.to_series().groupby(idx.to_period("M")).max().values]


def _drift(w: pd.Series, r: pd.Series) -> pd.Series:
    g = w * (1.0 + r.fillna(0.0))
    s = g.sum()
    return g / s if s > 0 else w


def run_backtest(px: pd.DataFrame, market_caps: dict[str, float],
                 sent: pd.DataFrame, disp: pd.DataFrame) -> dict:
    """
    px   : adj-close wide matrix (date x ticker), including warm-up history
    sent : daily ensemble sentiment (date x ticker)
    disp : daily dispersion (date x ticker)
    Returns a JSON-serialisable dict consumed by the API.
    """
    cfg = load_config()
    bt = cfg.backtest
    tickers = list(px.columns)
    rets = px.pct_change().dropna(how="all")
    cost = float(bt.transaction_cost_bps) / 1e4
    rf = float(bt.risk_free_annual)
    start, end = cfg.universe.start, cfg.universe.end

    rebal_dates = month_end_rebalance_dates(rets.index, start, end)
    if len(rebal_dates) < 3:
        raise RuntimeError("Not enough rebalance dates - check universe.start/end and price coverage")
    log.info("backtest: %d monthly rebalances %s .. %s", len(rebal_dates), rebal_dates[0].date(), rebal_dates[-1].date())

    # state per portfolio
    cur: dict[str, pd.Series | None] = {p: None for p in PORTFOLIOS}
    daily_gross = {p: [] for p in PORTFOLIOS}
    daily_net = {p: [] for p in PORTFOLIOS}
    turnover = {p: [] for p in PORTFOLIOS}
    weights_hist: list[dict] = []
    views_hist: list[dict] = []

    for k, d in enumerate(rebal_dates):
        as_of = pd.Timestamp(d)
        # ---------- everything below is sliced to <= as_of (NO LOOK-AHEAD) ----------
        hist_rets = rets.loc[:as_of]
        res = optimize_portfolio(hist_rets, market_caps, sent.loc[:as_of], disp.loc[:as_of],
                                 as_of, last_prices=px.loc[as_of], use_views=True, seed=k,
                                 w_prev=cur["bl_sentiment"])
        targets = {
            "bl_sentiment": res.weights,
            "market_cap": cap_weights(market_caps, tickers),
            "equal_weight": pd.Series(1.0 / len(tickers), index=tickers),
        }
        weights_hist.append({"date": as_of.strftime("%Y-%m-%d"),
                             **{t: round(float(res.weights[t]), 5) for t in tickers}})
        views_hist.append({
            "date": as_of.strftime("%Y-%m-%d"), "n_views": res.views.k,
            "views": [{"ticker": t, "z": round(res.views.z[t], 3), "q": round(float(q), 4),
                       "confidence": round(res.views.confidence[t], 3)}
                      for t, q in zip(res.views.view_tickers, res.views.Q)],
        })

        # ---------- holding period: (as_of, next rebalance] ----------
        nxt = pd.Timestamp(rebal_dates[k + 1]) if k + 1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[as_of:nxt].iloc[1:]           # strictly after as_of
        if period.empty:
            continue

        for p in PORTFOLIOS:
            w_target = targets[p]
            w_prev = cur[p]
            to = float((w_target - w_prev).abs().sum()) if w_prev is not None else float(w_target.abs().sum())
            turnover[p].append((as_of, to))
            w = w_target.copy()
            first = True
            for day, r in period.iterrows():
                pr = float((w * r.fillna(0.0)).sum())
                daily_gross[p].append((day, pr))
                # cost hits the first day of the new holding period
                daily_net[p].append((day, pr - (cost * to if first else 0.0)))
                first = False
                w = _drift(w, r)
            cur[p] = w  # drifted weights entering next rebalance

    def _series(pairs):
        s = pd.Series({d: v for d, v in pairs}).sort_index()
        s.index = pd.DatetimeIndex(s.index)
        return s

    gross = {p: _series(daily_gross[p]) for p in PORTFOLIOS}
    net = {p: _series(daily_net[p]) for p in PORTFOLIOS}
    to_s = {p: pd.Series({d: v for d, v in turnover[p]}) for p in PORTFOLIOS}
    bench = net["market_cap"]

    # ---------- regime split on realised vol of the market-cap benchmark ----------
    win = int(bt.regime_vol_window)
    rv = bench.rolling(win).std() * np.sqrt(252)
    rv = rv.bfill()
    med = float(rv.median())
    regime = pd.Series(np.where(rv > med, "high_vol", "low_vol"), index=rv.index)

    def metrics_block(returns: dict[str, pd.Series], mask: pd.Series | None = None) -> dict:
        out = {}
        for p in PORTFOLIOS:
            r = returns[p] if mask is None else returns[p][mask.reindex(returns[p].index).fillna(False)]
            b = bench if mask is None else bench[mask.reindex(bench.index).fillna(False)]
            out[p] = compute_metrics(r, rf, benchmark=b if p != "market_cap" else None,
                                     turnover=to_s[p] if mask is None else None)
        return out

    results = {
        "config": {
            "start": start, "end": end, "rebalance": "monthly",
            "transaction_cost_bps": float(bt.transaction_cost_bps), "risk_free_annual": rf,
            "n_rebalances": len(rebal_dates), "regime_vol_window": win,
            "regime_vol_median_annualized": med,
        },
        "dates": [d.strftime("%Y-%m-%d") for d in gross["bl_sentiment"].index],
        "equity": {
            "gross": {p: equity_curve(gross[p]).round(6).tolist() for p in PORTFOLIOS},
            "net": {p: equity_curve(net[p]).round(6).tolist() for p in PORTFOLIOS},
        },
        "metrics": {"gross": metrics_block(gross), "net": metrics_block(net)},
        "regimes": {
            "labels": regime.tolist(),
            "realized_vol": rv.round(5).tolist(),
            "high_vol": metrics_block(net, regime == "high_vol"),
            "low_vol": metrics_block(net, regime == "low_vol"),
            "n_days": {"high_vol": int((regime == "high_vol").sum()), "low_vol": int((regime == "low_vol").sum())},
        },
        "turnover": {p: [{"date": d.strftime("%Y-%m-%d"), "turnover": round(float(v), 5)} for d, v in to_s[p].items()]
                     for p in PORTFOLIOS},
        "weights_history": weights_hist,
        "views_history": views_hist,
    }
    m = results["metrics"]["net"]
    log.info("NET  Sharpe  BL=%.2f  MCAP=%.2f  EW=%.2f | CumRet BL=%.1f%% MCAP=%.1f%% EW=%.1f%%",
             m["bl_sentiment"]["sharpe"], m["market_cap"]["sharpe"], m["equal_weight"]["sharpe"],
             100 * m["bl_sentiment"]["cumulative_return"], 100 * m["market_cap"]["cumulative_return"],
             100 * m["equal_weight"]["cumulative_return"])
    return results
