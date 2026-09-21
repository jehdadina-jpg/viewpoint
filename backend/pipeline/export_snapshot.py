"""
Export a static snapshot of every API response for serverless hosting.

    python -m pipeline.export_snapshot          # -> ../deploy/backend/snapshot.json

Why: Vercel (and similar) cannot run torch/FinBERT, keep SQLite on disk, or
hold a WebSocket open. So we run the full pipeline locally, then freeze the
exact JSON each endpoint returns and let a dependency-free FastAPI in
deploy/backend/ serve it. Re-run this after every `pipeline.run` you want
published, then commit the snapshot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from config import BACKEND_DIR

log = logging.getLogger("snapshot")
OUT = BACKEND_DIR.parent / "deploy" / "backend" / "snapshot.json"


def export(out: Path = OUT) -> Path:
    snap: dict = {"meta": None, "stocks": None, "portfolio": {}, "history": {}, "news": {}}
    with TestClient(app) as c:
        snap["meta"] = c.get("/meta").json()
        snap["stocks"] = c.get("/stocks").json()
        for k in ("current", "backtest", "stats"):
            snap["portfolio"][k] = c.get(f"/portfolio/{k}").json()
        for t in snap["meta"]["universe"]:
            snap["history"][t] = c.get(f"/stock/{t}/history?days=2000").json()
            snap["news"][t] = c.get(f"/stock/{t}/news?limit=200").json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    log.info("snapshot -> %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export()
