"""
Deterministic SAMPLE data generators.

Two jobs:
  1. `sample_prices()`   - synthetic daily OHLCV when yfinance is unavailable.
  2. `sample_headlines()` - synthetic headlines for the historical backtest window,
                            because no free news API offers multi-year archives.

Everything produced here is flagged `source='sample'` / `is_synthetic=1` in the
database and surfaced in the API so the UI never presents it as real.

=============================================================================
LOOK-AHEAD BIAS NOTE (important - read before touching this file)
=============================================================================
Synthetic headline sentiment is tied to the stock's *trailing* 5-day return,
i.e. news on day t reflects what happened up to t-1. It is NEVER tied to
forward returns. If we seeded headline tone from future prices the backtest
would "discover" a fake edge. The generator therefore only ever looks at
`returns.iloc[:i]` when producing headlines for index i.

Real-world analogue: press coverage overwhelmingly *reacts* to price moves
(momentum-of-narrative), which is exactly what this mimics - and it's why a
sentiment strategy on sample data behaves roughly like a mild momentum tilt.
=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- prices
# Rough sector blocks used to give the synthetic covariance matrix a realistic
# block structure (same-sector names co-move more).
SECTORS = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMZN": "tech", "GOOGL": "tech", "META": "tech",
    "JPM": "fin", "GS": "fin",
    "XOM": "energy", "CVX": "energy",
    "JNJ": "health", "UNH": "health",
    "PG": "staples", "KO": "staples",
    "HD": "cyclical", "CAT": "cyclical",
}
_SECTOR_VOL = {"tech": 0.32, "fin": 0.26, "energy": 0.28, "health": 0.18, "staples": 0.15, "cyclical": 0.24}
_START_PRICE = {
    "AAPL": 175, "MSFT": 330, "NVDA": 24, "AMZN": 165, "GOOGL": 140, "META": 330,
    "JPM": 155, "GS": 380, "XOM": 62, "CVX": 118, "JNJ": 170, "UNH": 500,
    "PG": 160, "KO": 58, "HD": 400, "CAT": 205,
}
# Approximate market caps (USD bn) used ONLY when live caps are unavailable.
SAMPLE_MARKET_CAPS_BN = {
    "AAPL": 3400, "MSFT": 3300, "NVDA": 3000, "AMZN": 2000, "GOOGL": 2100, "META": 1400,
    "JPM": 650, "GS": 170, "XOM": 500, "CVX": 280, "JNJ": 380, "UNH": 480,
    "PG": 400, "KO": 280, "HD": 380, "CAT": 170,
}


def sample_prices(tickers: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """
    Correlated geometric Brownian motion with a market factor + sector factor +
    idiosyncratic noise. Returns long-format OHLCV with an `adj_close` column.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    sectors = sorted(set(SECTORS.get(t, "tech") for t in tickers))
    mkt = rng.normal(0, 1, n)
    sec = {s: rng.normal(0, 1, n) for s in sectors}

    frames = []
    for t in tickers:
        s = SECTORS.get(t, "tech")
        ann_vol = _SECTOR_VOL[s] * rng.uniform(0.85, 1.25)
        dvol = ann_vol / np.sqrt(252)
        drift = 0.08 / 252
        idio = rng.normal(0, 1, n)
        # factor loadings: ~55% market, ~25% sector, rest idiosyncratic
        z = 0.74 * mkt + 0.5 * sec[s] + 0.45 * idio
        rets = drift - 0.5 * dvol**2 + dvol * z
        close = _START_PRICE.get(t, 100) * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.normal(0, dvol * 0.2, n))
        hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, dvol * 0.5, n)))
        lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, dvol * 0.5, n)))
        vol = rng.lognormal(mean=np.log(2.5e7), sigma=0.4, size=n)
        frames.append(pd.DataFrame({
            "ticker": t, "date": dates, "open": open_, "high": hi, "low": lo,
            "close": close, "adj_close": close, "volume": vol,
        }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------- headlines
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet",
    "META": "Meta", "JPM": "JPMorgan", "GS": "Goldman Sachs", "XOM": "Exxon", "CVX": "Chevron",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PG": "Procter & Gamble", "KO": "Coca-Cola",
    "HD": "Home Depot", "CAT": "Caterpillar",
}
_SOURCES = ["Reuters", "Bloomberg", "WSJ", "CNBC", "FT", "MarketWatch", "Barron's", "r/stocks", "r/wallstreetbets"]

_POS = [
    "{c} beats earnings estimates, raises full-year guidance",
    "{c} shares climb after analyst upgrade to Buy",
    "{c} announces record quarterly revenue on strong demand",
    "{c} expands buyback program, dividend hiked",
    "Strong outlook: {c} sees accelerating growth in core segment",
    "{c} wins major contract; margins expected to improve",
    "Investors cheer {c} product launch, stock rallies",
    "{c} upgraded at Morgan Stanley on robust free cash flow",
    "{c} profit surges as costs fall and pricing holds",
    "Bullish on {c}: momentum and fundamentals aligned, says strategist",
]
_NEG = [
    "{c} misses on revenue, cuts guidance; shares slide",
    "{c} downgraded to Sell amid margin pressure",
    "Regulators open probe into {c} business practices",
    "{c} shares tumble after weak quarterly results",
    "{c} faces lawsuit over alleged disclosure failures",
    "Supply chain disruption hits {c}; outlook uncertain",
    "{c} warns of slowing demand, plans layoffs",
    "Analysts slash {c} price targets after disappointing update",
    "{c} loses key customer; revenue at risk",
    "Bearish sentiment builds around {c} as debt load grows",
]
_NEU = [
    "{c} to report quarterly results next week",
    "{c} schedules annual shareholder meeting",
    "{c} names new chief financial officer",
    "{c} files 10-Q with the SEC",
    "{c} holds investor day; reiterates existing targets",
    "What to watch when {c} reports earnings",
    "{c} trading flat ahead of Fed decision",
    "{c} shares little changed in quiet session",
]


def sample_headlines(prices_long: pd.DataFrame, start: str, end: str,
                     seed: int = 42, rate_per_day: float = 2.5) -> list[dict]:
    """
    Generate synthetic headlines for every ticker/trading-day in [start, end].

    Tone probability is a logistic function of the TRAILING 5-day return plus
    noise. See the module docstring for why this is look-ahead safe.
    """
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    px = prices_long.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = px.pct_change()
    dates = px.loc[start:end].index

    for t in px.columns:
        c = COMPANY_NAMES.get(t, t)
        r = rets[t]
        for i, d in enumerate(dates):
            # --- NO LOOK-AHEAD: only returns strictly before day `d` are visible
            pos = r.index.get_loc(d)
            trailing = r.iloc[max(0, pos - 5):pos].sum() if pos > 0 else 0.0
            # scale: 5-day move of +/-3% pushes tone probability noticeably
            tilt = np.tanh(trailing / 0.03)                # in (-1, 1)
            noise = rng.normal(0, 0.6)
            p_pos = 1 / (1 + np.exp(-(1.2 * tilt + noise)))  # logistic
            n_head = rng.poisson(rate_per_day)
            for k in range(n_head):
                u = rng.random()
                if u < 0.25:                                 # 25% neutral filler
                    tmpl = rng.choice(_NEU)
                elif rng.random() < p_pos:
                    tmpl = rng.choice(_POS)
                else:
                    tmpl = rng.choice(_NEG)
                # spread publish times across the trading day (UTC)
                hour = int(rng.integers(11, 22))
                minute = int(rng.integers(0, 60))
                ts = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=hour, minutes=minute)
                out.append({
                    "ticker": t,
                    "published_at": ts.isoformat(),
                    "headline": tmpl.format(c=c),
                    "source": str(rng.choice(_SOURCES)),
                    "url": None,
                    "provider": "sample",
                    "is_synthetic": 1,
                })
    return out
