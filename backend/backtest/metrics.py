"""
Performance metrics on a daily return series.

All annualisation uses 252 trading days. `rf` is an annual risk-free rate;
Sharpe/Sortino are computed on excess returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(daily: pd.Series) -> pd.Series:
    return (1.0 + daily.fillna(0.0)).cumprod()


def max_drawdown(daily: pd.Series) -> float:
    eq = equity_curve(daily)
    dd = eq / eq.cummax() - 1.0
    return float(dd.min())


def compute_metrics(daily: pd.Series, rf: float = 0.0, benchmark: pd.Series | None = None,
                    turnover: pd.Series | None = None) -> dict:
    """
    daily      portfolio daily simple returns
    rf         annual risk-free rate (for Sharpe / Sortino / alpha)
    benchmark  daily returns of the benchmark (for alpha / beta)
    turnover   per-rebalance turnover (sum |dw|) to annualise
    """
    d = daily.dropna()
    if len(d) < 2:
        return {}
    n = len(d)
    years = n / TRADING_DAYS
    eq = equity_curve(d)
    cum = float(eq.iloc[-1] - 1.0)
    ann_ret = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(d.std(ddof=1) * np.sqrt(TRADING_DAYS))
    rf_d = (1.0 + rf) ** (1.0 / TRADING_DAYS) - 1.0
    ex = d - rf_d
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS)) if ex.std(ddof=1) > 0 else np.nan
    downside = ex[ex < 0]
    dd_dev = float(np.sqrt((downside**2).sum() / n) * np.sqrt(TRADING_DAYS))  # full-sample downside deviation
    sortino = float(ex.mean() * TRADING_DAYS / dd_dev) if dd_dev > 0 else np.nan
    mdd = max_drawdown(d)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else np.nan

    out = {
        "n_days": int(n),
        "cumulative_return": cum,
        "annualized_return": ann_ret,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "best_day": float(d.max()),
        "worst_day": float(d.min()),
        "hit_rate": float((d > 0).mean()),
    }
    if benchmark is not None:
        b = benchmark.reindex(d.index).dropna()
        dd_ = d.reindex(b.index)
        if len(b) > 10 and b.var() > 0:
            # OLS: (r_p - rf) = alpha + beta (r_b - rf) + e
            x, y = (b - rf_d).values, (dd_ - rf_d).values
            beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
            alpha_d = float(y.mean() - beta * x.mean())
            out["beta"] = beta
            out["alpha_annualized"] = float(alpha_d * TRADING_DAYS)
            te = float((dd_ - b).std(ddof=1) * np.sqrt(TRADING_DAYS))
            out["tracking_error"] = te
            out["information_ratio"] = float((dd_ - b).mean() * TRADING_DAYS / te) if te > 0 else np.nan
    if turnover is not None and len(turnover):
        # one-way turnover per rebalance, annualised (12 rebalances / year)
        out["avg_turnover_per_rebalance"] = float(turnover.mean())
        out["annual_turnover"] = float(turnover.sum() / years) if years > 0 else np.nan
    return out


def clean(obj):
    """Recursively replace NaN/inf with None so the API can JSON-encode results."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
