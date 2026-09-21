"""
Price data: daily OHLCV for the configured universe, cached in SQLite.

Provider is selected in config.yaml (`prices.provider`):
  yfinance -> real data (no API key needed)
  sample   -> synthetic GBM paths (offline demo)

Repeated runs are served from SQLite; we only hit yfinance for dates we don't
already have, which keeps us well inside Yahoo's informal rate limits.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

import db
from config import load_config
from data.sample_data import SAMPLE_MARKET_CAPS_BN, sample_prices

log = logging.getLogger(__name__)

# Extra history pulled before the backtest start so the first rebalance has a
# full covariance lookback (252 trading days ~ 365 calendar days + buffer).
WARMUP_CALENDAR_DAYS = 420


def _cfg():
    return load_config()


def _needed_range() -> tuple[str, str]:
    cfg = _cfg()
    start = pd.Timestamp(cfg.universe.start) - timedelta(days=WARMUP_CALENDAR_DAYS)
    end = pd.Timestamp(cfg.universe.end)
    # never ask for the future
    today = pd.Timestamp.today().normalize()
    if end > today:
        end = today
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ------------------------------------------------------------------ yfinance
def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False,
        group_by="ticker", progress=False, threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    frames = []
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            log.warning("yfinance: no data for %s", t)
            continue
        sub = sub.dropna(subset=["Close"])
        if sub.empty:
            continue
        adj = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
        frames.append(pd.DataFrame({
            "ticker": t,
            "date": pd.to_datetime(sub.index).tz_localize(None),
            "open": sub["Open"].values, "high": sub["High"].values, "low": sub["Low"].values,
            "close": sub["Close"].values, "adj_close": adj.values, "volume": sub["Volume"].values,
        }))
    if not frames:
        raise RuntimeError("yfinance returned no usable tickers")
    return pd.concat(frames, ignore_index=True)


def _fetch_market_caps_yfinance(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    caps: dict[str, float] = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            cap = fi.get("marketCap") if hasattr(fi, "get") else getattr(fi, "market_cap", None)
            if cap:
                caps[t] = float(cap)
        except Exception as e:  # noqa: BLE001
            log.warning("market cap fetch failed for %s: %s", t, e)
    return caps


# ---------------------------------------------------------------- public API
def ensure_prices(force: bool = False) -> pd.DataFrame:
    """
    Make sure SQLite holds prices for the full needed range and return them
    (long format). Uses the cache unless `force` or the cache is incomplete.
    """
    cfg = _cfg()
    tickers = list(cfg.universe.tickers)
    start, end = _needed_range()
    provider = cfg.prices.provider

    cached = db.load_prices(tickers, start, end)
    have_all = (not cached.empty and set(cached.ticker.unique()) == set(tickers)
                and cached.date.min() <= pd.Timestamp(start) + timedelta(days=7)
                and cached.date.max() >= pd.Timestamp(end) - timedelta(days=7))
    if have_all and not force:
        log.info("prices: cache hit (%d rows, source=%s)", len(cached), cached.source.iloc[0])
        return cached

    source = provider
    df: pd.DataFrame | None = None
    if provider == "yfinance":
        try:
            df = _fetch_yfinance(tickers, start, end)
            log.info("prices: fetched %d rows from yfinance", len(df))
        except Exception as e:  # noqa: BLE001
            if not cfg.prices.get("fallback_to_sample", True):
                raise
            log.warning("yfinance failed (%s); falling back to SAMPLE prices", e)
            source = "sample"
    if df is None:
        df = sample_prices(tickers, start, end)
        source = "sample"

    db.upsert_prices(df, source=source)
    db.set_meta("price_source", source)
    return db.load_prices(tickers, start, end)


def ensure_market_caps() -> dict[str, float]:
    cfg = _cfg()
    tickers = list(cfg.universe.tickers)
    caps = db.load_market_caps()
    if set(caps) >= set(tickers):
        return {t: caps[t] for t in tickers}

    source = "sample"
    if cfg.prices.provider == "yfinance":
        try:
            live = _fetch_market_caps_yfinance(tickers)
            if len(live) >= len(tickers) * 0.8:
                caps, source = live, "yfinance"
        except Exception as e:  # noqa: BLE001
            log.warning("market caps via yfinance failed: %s", e)
    if source == "sample":
        caps = {t: SAMPLE_MARKET_CAPS_BN.get(t, 200) * 1e9 for t in tickers}
    # fill any single missing name from the sample table so the prior is complete
    for t in tickers:
        caps.setdefault(t, SAMPLE_MARKET_CAPS_BN.get(t, 200) * 1e9)
    db.upsert_market_caps(caps, source)
    db.set_meta("market_cap_source", source)
    return {t: caps[t] for t in tickers}


def price_matrix(prices_long: pd.DataFrame, col: str = "adj_close") -> pd.DataFrame:
    """Wide matrix: index=date, columns=ticker."""
    return prices_long.pivot(index="date", columns="ticker", values=col).sort_index()
