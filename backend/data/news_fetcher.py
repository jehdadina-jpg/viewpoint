"""
News / social text ingestion with pluggable providers.

Provider is selected in config.yaml (`news.provider`). Each provider implements
`fetch(ticker, start, end) -> list[dict]` with keys:
    ticker, published_at (ISO UTC), headline, source, url, provider, is_synthetic

Adding a new provider = add a class here + a line in config.yaml. Nothing
downstream (sentiment, optimizer, API) knows or cares which provider ran.

=============================================================================
LOOK-AHEAD BIAS
=============================================================================
Headlines carry a `published_at` timestamp. Downstream code (aggregator.py,
backtest/engine.py) only ever uses rows with `published_at <= as_of`. This
module just needs to preserve the true publish time - never "round" it to a
later date, and never backfill a headline with a fabricated earlier time.
=============================================================================
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import pandas as pd

import db
from config import env_secret, load_config
from data.sample_data import sample_headlines

log = logging.getLogger(__name__)


class NewsProvider(ABC):
    name: str = "base"
    is_synthetic: int = 0
    # how far back this provider can actually see (None = unlimited)
    history_limit_days: int | None = None

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> list[dict]: ...


# ------------------------------------------------------------------- sample
class SampleProvider(NewsProvider):
    name = "sample"
    is_synthetic = 1

    def __init__(self, prices_long: pd.DataFrame, seed: int, rate: float):
        self._prices = prices_long
        self._seed = seed
        self._rate = rate
        self._cache: dict[tuple[str, str], list[dict]] = {}

    def fetch(self, ticker: str, start: str, end: str) -> list[dict]:
        key = (start, end)
        if key not in self._cache:
            self._cache[key] = sample_headlines(self._prices, start, end,
                                                seed=self._seed, rate_per_day=self._rate)
        return [r for r in self._cache[key] if r["ticker"] == ticker]


# ------------------------------------------------------------------ newsapi
class NewsAPIProvider(NewsProvider):
    """https://newsapi.org - free tier: last ~30 days, 100 requests/day."""
    name = "newsapi"
    history_limit_days = 29

    def __init__(self, api_key: str, page_size: int = 50, language: str = "en"):
        self._key = api_key
        self._page_size = page_size
        self._lang = language

    def fetch(self, ticker: str, start: str, end: str) -> list[dict]:
        import requests

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f'"{ticker}" OR "{_company_name(ticker)}"',
            "from": start, "to": end, "language": self._lang,
            "sortBy": "publishedAt", "pageSize": self._page_size, "apiKey": self._key,
        }
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 429:
            log.warning("NewsAPI rate limited; sleeping 5s"); time.sleep(5)
            r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        out = []
        for a in r.json().get("articles", []):
            if not a.get("title"):
                continue
            out.append({
                "ticker": ticker,
                "published_at": pd.Timestamp(a["publishedAt"]).tz_convert("UTC").isoformat(),
                "headline": a["title"],
                "source": (a.get("source") or {}).get("name"),
                "url": a.get("url"),
                "provider": self.name, "is_synthetic": 0,
            })
        time.sleep(0.5)  # be polite
        return out


# ------------------------------------------------------------------- reddit
class RedditProvider(NewsProvider):
    """Reddit via PRAW. Titles only (bodies are noisy). Needs client id/secret."""
    name = "reddit"
    history_limit_days = 30  # search endpoint is not a reliable archive

    def __init__(self, client_id: str, client_secret: str, user_agent: str, subreddits: list[str]):
        import praw  # lazy so the dependency is optional

        self._reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        self._subs = subreddits

    def fetch(self, ticker: str, start: str, end: str) -> list[dict]:
        out = []
        s_ts, e_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        for sub in self._subs:
            for post in self._reddit.subreddit(sub).search(f"${ticker} OR {ticker}", sort="new", limit=100):
                ts = pd.Timestamp(post.created_utc, unit="s", tz="UTC")
                if not (s_ts <= ts <= e_ts):
                    continue
                out.append({
                    "ticker": ticker, "published_at": ts.isoformat(), "headline": post.title,
                    "source": f"r/{sub}", "url": f"https://reddit.com{post.permalink}",
                    "provider": self.name, "is_synthetic": 0,
                })
            time.sleep(1.0)
        return out


# ----------------------------------------------------------------- yfinance
class YFinanceNewsProvider(NewsProvider):
    """Ticker(...).news - recent stories only, no key. Good for the live panel."""
    name = "yfinance"
    history_limit_days = 14

    def fetch(self, ticker: str, start: str, end: str) -> list[dict]:
        import yfinance as yf

        out = []
        s_ts, e_ts = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        try:
            items = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance news failed for %s: %s", ticker, e)
            return out
        for it in items:
            c = it.get("content", it)  # new-style payload nests under 'content'
            title = c.get("title")
            pub = c.get("pubDate") or it.get("providerPublishTime")
            if not title or pub is None:
                continue
            ts = pd.Timestamp(pub, unit="s", tz="UTC") if isinstance(pub, (int, float)) else pd.Timestamp(pub).tz_convert("UTC")
            if not (s_ts <= ts <= e_ts):
                continue
            prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else it.get("publisher")
            link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else it.get("link")
            out.append({
                "ticker": ticker, "published_at": ts.isoformat(), "headline": title,
                "source": prov, "url": link, "provider": self.name, "is_synthetic": 0,
            })
        time.sleep(0.3)
        return out


# ------------------------------------------------------------------ factory
def _company_name(ticker: str) -> str:
    from data.sample_data import COMPANY_NAMES
    return COMPANY_NAMES.get(ticker, ticker)


def build_provider(prices_long: pd.DataFrame) -> NewsProvider:
    cfg = load_config()
    n = cfg.news
    p = n.provider
    if p == "newsapi":
        key = env_secret(n.newsapi.api_key_env)
        if not key:
            log.warning("news.provider=newsapi but %s not set; using sample", n.newsapi.api_key_env)
        else:
            return NewsAPIProvider(key, n.newsapi.get("page_size", 50), n.newsapi.get("language", "en"))
    elif p == "reddit":
        cid, sec = env_secret(n.reddit.client_id_env), env_secret(n.reddit.client_secret_env)
        if not (cid and sec):
            log.warning("news.provider=reddit but credentials not set; using sample")
        else:
            return RedditProvider(cid, sec, n.reddit.user_agent, list(n.reddit.subreddits))
    elif p == "yfinance":
        return YFinanceNewsProvider()
    return SampleProvider(prices_long, seed=n.sample.get("seed", 42), rate=n.sample.get("headlines_per_day_mean", 2.5))


# --------------------------------------------------------------- public API
def ensure_news(prices_long: pd.DataFrame, force: bool = False) -> int:
    """
    Populate the `news` table for the backtest window.

    Live providers can only see a short trailing window, so the history that
    they cannot cover is filled by the sample generator (flagged synthetic).
    Cache-aware: skips tickers that already have coverage for the range.
    """
    cfg = load_config()
    tickers = list(cfg.universe.tickers)
    start, end = cfg.universe.start, cfg.universe.end
    end = min(pd.Timestamp(end), pd.Timestamp.today().normalize()).strftime("%Y-%m-%d")

    provider = build_provider(prices_long)
    lo, hi = db.news_date_range()
    if lo and hi and not force and lo[:10] <= start and hi[:10] >= pd.Timestamp(end).strftime("%Y-%m-%d")[:7]:
        log.info("news: cache hit (%s .. %s)", lo[:10], hi[:10])
        return 0

    inserted = 0
    # 1) Historical window from the sample generator if the live provider can't reach it
    if provider.history_limit_days is not None:
        cutoff = (pd.Timestamp(end) - pd.Timedelta(days=provider.history_limit_days)).strftime("%Y-%m-%d")
        sample = SampleProvider(prices_long, cfg.news.sample.get("seed", 42), cfg.news.sample.get("headlines_per_day_mean", 2.5))
        for t in tickers:
            inserted += db.insert_news(sample.fetch(t, start, cutoff))
        live_start = cutoff
    else:
        live_start = start

    # 2) Provider window
    for t in tickers:
        try:
            rows = provider.fetch(t, live_start, end)
        except Exception as e:  # noqa: BLE001
            log.warning("news provider %s failed for %s: %s", provider.name, t, e)
            rows = []
        inserted += db.insert_news(rows)
    db.set_meta("news_provider", provider.name)
    log.info("news: inserted %d headlines via %s", inserted, provider.name)
    return inserted
