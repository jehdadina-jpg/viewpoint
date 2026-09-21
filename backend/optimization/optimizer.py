r"""
Black-Litterman posterior + constrained mean-variance optimisation.

Implemented by hand with numpy/scipy (no PyPortfolioOpt) so every step of the
math is visible and explainable in a Q&A. See views.py for the derivation of
the formulas referenced below as (1)-(3).

Flow per rebalance:
    Sigma  = shrunk sample covariance of trailing daily returns, annualised
    w_mkt  = market-cap weights (the equilibrium / prior portfolio)
    pi     = delta * Sigma * w_mkt                                  (1)
    views  = build_views(...)  -> P, Q, Omega
    mu_BL  = BL master formula                                      (2)
    Sigma_BL                                                        (3)
    w*     = argmax_w  mu_BL'w - (delta/2) w' Sigma_BL w
             s.t. sum w = 1, 0 <= w_i <= max_weight                (long-only, capped)

Benchmarks use the same machinery with no views (w_mkt directly) or 1/N.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import load_config
from optimization.views import TRADING_DAYS, ViewSet, build_views


# --------------------------------------------------------------- covariance
def shrunk_covariance(returns: pd.DataFrame, shrink: float = 0.10) -> pd.DataFrame:
    """
    Annualised covariance with linear shrinkage toward the diagonal:
        Sigma = (1 - a) * S + a * diag(S)
    A poor-man's Ledoit-Wolf; with 16 names and 252 obs the sample covariance is
    reasonably conditioned, but shrinkage keeps the BL inverse stable.
    """
    S = returns.cov().values * TRADING_DAYS
    D = np.diag(np.diag(S))
    Sig = (1.0 - shrink) * S + shrink * D
    return pd.DataFrame(Sig, index=returns.columns, columns=returns.columns)


# --------------------------------------------------------------- BL pieces
def market_prior(Sigma: pd.DataFrame, w_mkt: pd.Series, delta: float) -> pd.Series:
    """Equation (1): reverse-optimised equilibrium excess returns pi = delta Sigma w."""
    w = w_mkt.reindex(Sigma.columns).fillna(0.0).values
    return pd.Series(delta * Sigma.values @ w, index=Sigma.columns)


def black_litterman_posterior(Sigma: pd.DataFrame, pi: pd.Series, views: ViewSet, tau: float
                              ) -> tuple[pd.Series, pd.DataFrame]:
    """
    Equations (2) and (3).

        A      = (tau Sigma)^-1 + P' Omega^-1 P            posterior precision
        mu_BL  = A^-1 [ (tau Sigma)^-1 pi + P' Omega^-1 Q ]
        Sig_BL = Sigma + A^-1
    With zero views A^-1 = tau Sigma and mu_BL = pi exactly.
    """
    S = Sigma.values
    n = S.shape[0]
    tS_inv = np.linalg.inv(tau * S)
    if views.k == 0:
        return pi.copy(), Sigma + tau * Sigma
    P, Q, Om = views.P, views.Q, views.Omega
    Om_inv = np.linalg.inv(Om)
    A = tS_inv + P.T @ Om_inv @ P
    A_inv = np.linalg.inv(A)
    mu = A_inv @ (tS_inv @ pi.values + P.T @ Om_inv @ Q)
    Sig_bl = S + A_inv
    return pd.Series(mu, index=Sigma.columns), pd.DataFrame(Sig_bl, index=Sigma.columns, columns=Sigma.columns)


# --------------------------------------------------------- mean-variance
def mean_variance_weights(mu: pd.Series, Sigma: pd.DataFrame, delta: float,
                          max_weight: float = 0.15, min_weight: float = 0.0,
                          long_only: bool = True, w_prev: pd.Series | None = None,
                          turnover_penalty: float = 0.0) -> pd.Series:
    """
    max_w  mu'w - (delta/2) w'Sigma w - lambda ||w - w_prev||^2
           s.t.  sum(w)=1,  lo <= w <= hi

    Solved with SLSQP. Quadratic utility with the same delta as the prior keeps
    the "no views -> hold the market" property (up to the constraints).

    The turnover term (lambda = `turnover_penalty`, w_prev = drifted weights
    entering the rebalance) is the standard fix for MV optimisers that flip
    names between 0 and the cap every month: it makes the book move only when
    the posterior return change justifies the transaction cost. Without it the
    unpenalised strategy turned over ~8x/year in testing.
    """
    n = len(mu)
    m, S = mu.values, Sigma.values
    lo = min_weight if long_only else -max_weight
    bounds = [(lo, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    wp = w_prev.reindex(mu.index).fillna(0.0).values if w_prev is not None else None
    lam = float(turnover_penalty) if wp is not None else 0.0

    def neg_utility(w):
        pen = lam * ((w - wp) ** 2).sum() if lam else 0.0
        return -(m @ w - 0.5 * delta * w @ S @ w) + pen

    def grad(w):
        g = -(m - delta * S @ w)
        if lam:
            g = g + 2.0 * lam * (w - wp)
        return g

    x0 = wp.copy() if wp is not None else np.full(n, 1.0 / n)
    res = minimize(neg_utility, x0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if res.success else x0
    w = np.clip(w, lo, max_weight)
    w = w / w.sum()
    return pd.Series(w, index=mu.index)


# -------------------------------------------------------------- top level
@dataclass
class OptimizationResult:
    weights: pd.Series
    prior: pd.Series          # pi
    posterior: pd.Series      # mu_BL
    w_mkt: pd.Series
    views: ViewSet
    as_of: pd.Timestamp


def cap_weights(market_caps: dict[str, float], tickers: list[str], max_weight: float | None = None) -> pd.Series:
    """
    Market-cap weights. NOTE: we only have *current* caps (yfinance gives no
    history). Using today's caps throughout the backtest introduces a mild
    survivorship/size drift in the prior; documented in the README as a
    known limitation. Optionally capped to keep the prior comparable to the
    constrained BL portfolio.
    """
    w = pd.Series({t: float(market_caps.get(t, 0.0)) for t in tickers})
    w = w / w.sum()
    if max_weight:
        # iterative cap-and-redistribute
        for _ in range(50):
            over = w > max_weight
            if not over.any():
                break
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            under = ~over
            w[under] += excess * w[under] / w[under].sum()
    return w


def optimize_portfolio(returns: pd.DataFrame, market_caps: dict[str, float],
                       sent: pd.DataFrame | None, disp: pd.DataFrame | None,
                       as_of: pd.Timestamp, last_prices: pd.Series | None = None,
                       use_views: bool = True, seed: int = 0,
                       w_prev: pd.Series | None = None) -> OptimizationResult:
    """
    Run the full BL pipeline as of one rebalance date.

    NO LOOK-AHEAD: `returns`, `sent`, `disp` must already be truncated to
    dates <= as_of by the caller (backtest engine does this).
    """
    cfg = load_config()
    bl, oc = cfg.black_litterman, cfg.optimizer
    tickers = list(returns.columns)
    delta, tau = float(bl.risk_aversion), float(bl.tau)

    Sigma = shrunk_covariance(returns.tail(int(bl.cov_lookback_days)), float(bl.cov_shrinkage))
    w_mkt = cap_weights(market_caps, tickers)
    pi = market_prior(Sigma, w_mkt, delta)

    if use_views and sent is not None and disp is not None:
        views = build_views(sent, disp, pi, Sigma, as_of, last_prices, seed=seed)
    else:
        views = ViewSet(tickers=tickers, P=np.zeros((0, len(tickers))), Q=np.zeros(0), Omega=np.zeros((0, 0)))

    mu_bl, Sig_bl = black_litterman_posterior(Sigma, pi, views, tau)
    w = mean_variance_weights(mu_bl, Sig_bl, delta, float(oc.max_weight), float(oc.min_weight), bool(oc.long_only),
                              w_prev=w_prev, turnover_penalty=float(oc.get("turnover_penalty", 0.0)))
    return OptimizationResult(weights=w, prior=pi, posterior=mu_bl, w_mkt=w_mkt, views=views, as_of=as_of)
