"""
SQLite persistence layer.

Everything the pipeline pulls from an external API is cached here so that
repeated runs never re-hit rate-limited endpoints. Tables:

  prices            daily OHLCV per ticker
  market_caps       latest market cap per ticker (Black-Litterman prior weights)
  news              one row per headline; `is_synthetic` flags sample data
  headline_scores   one row per (headline, model) - raw ensemble member outputs
  daily_sentiment   aggregated per (ticker, date) - the signal fed to the optimizer
  meta              key/value (data_source flags, last fetch timestamps, ...)
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from config import load_config, resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,           -- ISO yyyy-mm-dd
    open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    source TEXT NOT NULL,           -- 'yfinance' | 'sample'
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS market_caps (
    ticker TEXT PRIMARY KEY,
    market_cap REAL,
    source TEXT NOT NULL,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker       TEXT NOT NULL,
    published_at TEXT NOT NULL,     -- ISO datetime, UTC
    headline     TEXT NOT NULL,
    source       TEXT,
    url          TEXT,
    provider     TEXT NOT NULL,     -- 'sample' | 'newsapi' | 'reddit' | 'yfinance'
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    UNIQUE (ticker, published_at, headline)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news (ticker, published_at);
CREATE TABLE IF NOT EXISTS headline_scores (
    news_id    INTEGER NOT NULL,
    model      TEXT NOT NULL,       -- 'vader' | 'finbert' | 'lexicon'
    score      REAL NOT NULL,       -- [-1, 1]
    confidence REAL NOT NULL,       -- [0, 1]
    PRIMARY KEY (news_id, model),
    FOREIGN KEY (news_id) REFERENCES news(id)
);
CREATE TABLE IF NOT EXISTS daily_sentiment (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    score       REAL,               -- ensemble, time-decayed, [-1, 1]
    confidence  REAL,               -- [0, 1]; 1 - normalised dispersion
    dispersion  REAL,               -- std-dev across models x headlines
    n_headlines INTEGER,
    model_scores TEXT,              -- json {model: score} for transparency
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def db_path() -> Path:
    cfg = load_config()
    p = resolve_path(cfg.get_path("prices.cache_db", "storage/market.db"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- meta
def set_meta(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def get_meta(key: str, default: str | None = None) -> str | None:
    with connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


# ------------------------------------------------------------------------- prices
def upsert_prices(df: pd.DataFrame, source: str) -> int:
    """df columns: ticker, date, open, high, low, close, adj_close, volume."""
    if df.empty:
        return 0
    rows = [
        (r.ticker, str(r.date)[:10], float(r.open), float(r.high), float(r.low),
         float(r.close), float(r.adj_close), float(r.volume), source)
        for r in df.itertuples(index=False)
    ]
    with connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def load_prices(tickers: Iterable[str] | None = None,
                start: str | None = None, end: str | None = None) -> pd.DataFrame:
    q = "SELECT ticker, date, open, high, low, close, adj_close, volume, source FROM prices WHERE 1=1"
    params: list = []
    if tickers:
        tl = list(tickers)
        placeholders = ",".join(["?"] * len(tl))
        q += f" AND ticker IN ({placeholders})"
        params += tl
    if start:
        q += " AND date >= ?"
        params.append(start)
    if end:
        q += " AND date <= ?"
        params.append(end)
    q += " ORDER BY ticker, date"
    with connect() as conn:
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def upsert_market_caps(caps: dict[str, float], source: str) -> None:
    now = pd.Timestamp.now(tz="UTC").isoformat()
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO market_caps VALUES (?,?,?,?)",
            [(t, float(c), source, now) for t, c in caps.items()],
        )


def load_market_caps() -> dict[str, float]:
    with connect() as conn:
        rows = conn.execute("SELECT ticker, market_cap FROM market_caps").fetchall()
    return {t: c for t, c in rows}


# --------------------------------------------------------------------------- news
def insert_news(rows: list[dict]) -> int:
    """rows: dicts with ticker, published_at, headline, source, url, provider, is_synthetic."""
    if not rows:
        return 0
    with connect() as conn:
        cur = conn.executemany(
            """INSERT OR IGNORE INTO news
               (ticker, published_at, headline, source, url, provider, is_synthetic)
               VALUES (:ticker, :published_at, :headline, :source, :url, :provider, :is_synthetic)""",
            rows,
        )
        return cur.rowcount


def load_news(ticker: str | None = None, start: str | None = None,
              end: str | None = None, limit: int | None = None) -> pd.DataFrame:
    q = "SELECT id, ticker, published_at, headline, source, url, provider, is_synthetic FROM news WHERE 1=1"
    params: list = []
    if ticker:
        q += " AND ticker = ?"
        params.append(ticker)
    if start:
        q += " AND published_at >= ?"
        params.append(start)
    if end:
        q += " AND published_at <= ?"
        params.append(end)
    q += " ORDER BY published_at DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with connect() as conn:
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def news_date_range(ticker: str | None = None) -> tuple[str | None, str | None]:
    q = "SELECT MIN(published_at), MAX(published_at) FROM news"
    params: list = []
    if ticker:
        q += " WHERE ticker = ?"
        params.append(ticker)
    with connect() as conn:
        row = conn.execute(q, params).fetchone()
    return (row[0], row[1]) if row else (None, None)


def unscored_news(model: str) -> pd.DataFrame:
    q = """SELECT n.id, n.ticker, n.headline FROM news n
           LEFT JOIN headline_scores s ON s.news_id = n.id AND s.model = ?
           WHERE s.news_id IS NULL ORDER BY n.id"""
    with connect() as conn:
        return pd.read_sql_query(q, conn, params=[model])


def upsert_headline_scores(rows: list[tuple[int, str, float, float]]) -> None:
    if not rows:
        return
    with connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO headline_scores VALUES (?,?,?,?)", rows)


def load_headline_scores(ticker: str | None = None, news_ids: Iterable[int] | None = None) -> pd.DataFrame:
    q = """SELECT n.id AS news_id, n.ticker, n.published_at, s.model, s.score, s.confidence
           FROM headline_scores s JOIN news n ON n.id = s.news_id WHERE 1=1"""
    params: list = []
    if ticker:
        q += " AND n.ticker = ?"
        params.append(ticker)
    if news_ids is not None:
        ids = [int(i) for i in news_ids]
        if not ids:
            return pd.DataFrame(columns=["news_id", "ticker", "published_at", "model", "score", "confidence"])
        q += f" AND n.id IN ({','.join(['?'] * len(ids))})"
        params += ids
    with connect() as conn:
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


# ----------------------------------------------------------------- daily sentiment
def replace_daily_sentiment(df: pd.DataFrame) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM daily_sentiment")
        conn.executemany(
            "INSERT INTO daily_sentiment VALUES (?,?,?,?,?,?,?)",
            [
                (r.ticker, str(r.date)[:10],
                 None if pd.isna(r.score) else float(r.score),
                 None if pd.isna(r.confidence) else float(r.confidence),
                 None if pd.isna(r.dispersion) else float(r.dispersion),
                 int(r.n_headlines), r.model_scores)
                for r in df.itertuples(index=False)
            ],
        )


def load_daily_sentiment(ticker: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM daily_sentiment"
    params: list = []
    if ticker:
        q += " WHERE ticker = ?"
        params.append(ticker)
    q += " ORDER BY ticker, date"
    with connect() as conn:
        df = pd.read_sql_query(q, conn, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
