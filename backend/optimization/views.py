r"""
Sentiment -> Black-Litterman views.

=============================================================================
THE BLACK-LITTERMAN MODEL IN ONE PAGE (Black & Litterman, 1992; He & Litterman, 1999)
=============================================================================
Notation (N assets, K views):

  Sigma  (N x N)  covariance of excess returns (annualised)
  w_mkt  (N)      market-cap weights            <- the equilibrium portfolio
  delta  scalar   risk aversion
  pi     (N)      implied equilibrium excess returns
                    pi = delta * Sigma * w_mkt                          (1)
                  i.e. "what returns would make a mean-variance investor
                  hold the market portfolio?" - reverse optimisation.
  tau    scalar   uncertainty in the prior; prior on mu ~ N(pi, tau*Sigma)

  P      (K x N)  picking matrix. Row k selects the assets that view k is about.
                  Absolute view on asset i: a unit row e_i.
                  Relative view "i beats j": +1 at i, -1 at j.
  Q      (K)      the view values: P mu ~ N(Q, Omega)
  Omega  (K x K)  view uncertainty (diagonal): small = confident view.

Posterior expected returns (the BL "master formula"):

  mu_BL = [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1 [ (tau Sigma)^-1 pi + P' Omega^-1 Q ]   (2)

Intuition: a precision-weighted average of the prior pi and the views Q. If
Omega -> infinity (no confidence) mu_BL -> pi and the optimizer holds the
market. If Omega -> 0 the views are imposed exactly.

Posterior covariance:

  Sigma_BL = Sigma + [ (tau Sigma)^-1 + P' Omega^-1 P ]^-1                          (3)

=============================================================================
HOW SENTIMENT BECOMES (P, Q, Omega)  -  this file
=============================================================================
Following Colasanto, Grilli, Santoro & Villani (2022) we treat the ensemble
sentiment score as an *absolute* view on each stock:

  1. z-score the sentiment per stock over a trailing window (using only data
     up to `as_of` - no look-ahead):
        z_i = (s_i - mean_{t<=as_of}(s_i)) / std_{t<=as_of}(s_i)
     This removes per-name coverage bias (a stock whose press is always
     upbeat should not get a permanent overweight).

  2. Only stocks with |z_i| >= min_abs_z get a view -> P has one unit row per
     such stock. Fewer, stronger views is both more robust and closer to how a
     PM would actually express conviction.

  3. Q_i via Monte Carlo (Colasanto et al. convert sentiment into a *price*
     view by simulating paths). We simulate GBM over the rebalance horizon
     with a sentiment-tilted drift:
        mu_i^view = pi_i + view_scale * z_i                 (annualised)
        dS/S = mu_i^view dt + sigma_i dW
     and set Q_i = annualised mean simulated return, plus a target price for
     the dashboard. (For pure GBM the MC mean is analytic; we keep the MC so
     the framework extends to fat-tailed / jump models without touching BL.)

  4. Omega_i inversely proportional to view confidence. The aggregator gives
     us `dispersion_i` = disagreement across models and headlines. We inflate
     the canonical He-Litterman choice Omega = diag(P tau Sigma P') by it:
        Omega_ii = tau * Sigma_ii * (1 + k * d_i) + omega_floor,
        d_i = clip(dispersion_i / d_max, 0, 1)
     Noisy sentiment -> wide Omega -> the optimizer trusts pi more. This is
     the mechanism by which the ensemble's *uncertainty*, not just its point
     estimate, reaches the portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import load_config

TRADING_DAYS = 252
DISPERSION_MAX = 0.6  # empirical upper range of aggregator dispersion; maps to d_i = 1


@dataclass
class ViewSet:
    tickers: list[str]                       # universe order (columns of P)
    P: np.ndarray                            # K x N
    Q: np.ndarray                            # K
    Omega: np.ndarray                        # K x K diagonal
    view_tickers: list[str] = field(default_factory=list)
    z: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)   # 1/(1+k d)
    target_price: dict[str, float] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.view_tickers)


def sentiment_zscores(sent: pd.DataFrame, as_of: pd.Timestamp, window: int) -> pd.Series:
    """
    Trailing-window z-score per ticker. `sent` is date x ticker.
    NO LOOK-AHEAD: slice strictly `<= as_of` before computing any statistic.
    """
    hist = sent.loc[:as_of].tail(window)
    if len(hist) < max(10, window // 3):
        return pd.Series(0.0, index=sent.columns)
    mu = hist.mean()
    sd = hist.std().clip(lower=1e-3)
    latest = hist.iloc[-1]
    return ((latest - mu) / sd).fillna(0.0)


def monte_carlo_view(mu_annual: float, sigma_annual: float, horizon_days: int,
                     n_paths: int, rng: np.random.Generator, s0: float = 1.0) -> tuple[float, float]:
    """
    Simulate GBM S_T for `horizon_days` and return
      (annualised mean simple return, mean terminal price).
    Colasanto et al. (2022) style: sentiment enters as the drift.
    """
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal((n_paths, horizon_days))
    log_paths = (mu_annual - 0.5 * sigma_annual**2) * dt + sigma_annual * np.sqrt(dt) * z
    s_t = s0 * np.exp(log_paths.sum(axis=1))
    horizon_ret = s_t.mean() / s0 - 1.0
    annualised = (1.0 + horizon_ret) ** (TRADING_DAYS / horizon_days) - 1.0
    return float(annualised), float(s_t.mean())


def build_views(sent: pd.DataFrame, disp: pd.DataFrame, pi: pd.Series, Sigma: pd.DataFrame,
                as_of: pd.Timestamp, last_prices: pd.Series | None = None,
                seed: int = 0) -> ViewSet:
    """
    sent, disp : date x ticker matrices (ensemble score, dispersion)
    pi         : equilibrium returns (from optimizer.market_prior)
    Sigma      : annualised covariance, ticker x ticker
    as_of      : rebalance date - only information <= as_of is used
    """
    cfg = load_config().black_litterman
    tickers = list(Sigma.columns)
    n = len(tickers)
    tau = float(cfg.tau)
    view_scale = float(cfg.view_scale_annual)
    min_z = float(cfg.min_abs_z_for_view)
    k_disp = float(cfg.omega_dispersion_mult)
    omega_floor = float(cfg.get("omega_floor", 1e-4))
    horizon = int(cfg.horizon_days)
    n_paths = int(cfg.monte_carlo_paths)

    z = sentiment_zscores(sent[tickers], as_of, int(cfg.view_z_window_days))
    # latest dispersion as of the rebalance date (NaN -> treat as maximally noisy)
    d_hist = disp[tickers].loc[:as_of]
    d_latest = d_hist.iloc[-1] if len(d_hist) else pd.Series(np.nan, index=tickers)
    d_norm = (d_latest / DISPERSION_MAX).clip(0.0, 1.0).fillna(1.0)

    rng = np.random.default_rng(seed)
    rows, Q, Om, vt, conf, tgt = [], [], [], [], {}, {}
    for i, t in enumerate(tickers):
        zi = float(z[t])
        if abs(zi) < min_z or np.isnan(zi):
            continue
        sigma_i = float(np.sqrt(Sigma.loc[t, t]))
        mu_view = float(pi[t]) + view_scale * zi
        q_i, s_T = monte_carlo_view(mu_view, sigma_i, horizon, n_paths, rng,
                                    s0=float(last_prices[t]) if last_prices is not None else 1.0)
        row = np.zeros(n); row[i] = 1.0                       # absolute view on asset i
        omega_i = tau * float(Sigma.loc[t, t]) * (1.0 + k_disp * float(d_norm[t])) + omega_floor
        rows.append(row); Q.append(q_i); Om.append(omega_i); vt.append(t)
        conf[t] = float(1.0 / (1.0 + k_disp * float(d_norm[t])))
        if last_prices is not None:
            tgt[t] = s_T

    if not rows:
        P = np.zeros((0, n)); Qa = np.zeros(0); Oma = np.zeros((0, 0))
    else:
        P = np.vstack(rows); Qa = np.array(Q); Oma = np.diag(Om)
    return ViewSet(tickers=tickers, P=P, Q=Qa, Omega=Oma, view_tickers=vt,
                   z={t: float(z[t]) for t in tickers}, confidence=conf, target_price=tgt)
