"""
VIEWPOINT serve-mode API for serverless hosting (Vercel).

Serves the JSON snapshot produced by `python -m pipeline.export_snapshot`
from the real backend. No torch, no pandas, no SQLite, no WebSocket: the
whole thing is dictionary lookups, so cold starts are fast and the bundle
is tiny. Routes mirror backend/api/main.py exactly, and are mounted both at
`/` and `/api` so they work regardless of whether the platform rewrite
strips the prefix.

The frontend detects the missing WebSocket and simulates ticks client-side.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

SNAP = json.loads((Path(__file__).parent / "snapshot.json").read_text(encoding="utf-8"))
UNIVERSE: list[str] = SNAP["meta"]["universe"]

app = FastAPI(title="VIEWPOINT API (snapshot)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
r = APIRouter()


def _t(ticker: str) -> str:
    t = ticker.upper()
    if t not in UNIVERSE:
        raise HTTPException(404, f"{t} not in universe")
    return t


@r.get("/meta")
def meta():
    return {**SNAP["meta"], "serve_mode": "snapshot"}


@r.get("/stocks")
def stocks():
    return SNAP["stocks"]


@r.get("/stock/{ticker}/history")
def history(ticker: str, days: int = Query(365, ge=30, le=2000)):
    h = SNAP["history"][_t(ticker)]
    return {**h, "candles": h["candles"][-days:], "sentiment": h["sentiment"][-days:]}


@r.get("/stock/{ticker}/news")
def news(ticker: str, limit: int = Query(40, ge=1, le=200)):
    n = SNAP["news"][_t(ticker)]
    return {**n, "news": n["news"][:limit]}


@r.get("/portfolio/current")
def portfolio_current():
    return SNAP["portfolio"]["current"]


@r.get("/portfolio/backtest")
def portfolio_backtest():
    return SNAP["portfolio"]["backtest"]


@r.get("/portfolio/stats")
def portfolio_stats():
    return SNAP["portfolio"]["stats"]


@r.get("/health")
def health():
    return {"ok": True, "built_at": SNAP["meta"].get("built_at")}


app.include_router(r)
app.include_router(r, prefix="/api")
