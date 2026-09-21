import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api/client";
import type { StockRow, Tick, TickMsg } from "../types";

export type TickMap = Record<string, Tick>;
export type WsStatus = "connecting" | "live" | "local" | "down";

const MAX_WS_ATTEMPTS = 2;

/**
 * SIMULATED tick stream. Two sources, same shape:
 *   "live"  - the backend WebSocket (/ws/ticks), which jitters around the last close
 *   "local" - the same Ornstein-Uhlenbeck jitter computed in the browser, used when
 *             the WebSocket is unavailable (serverless hosting has no WS)
 * Either way the numbers are for the cell-flash / tape demo only.
 */
export function useTicks(stocks: StockRow[]): { ticks: TickMap; status: WsStatus } {
  const [ticks, setTicks] = useState<TickMap>({});
  const [status, setStatus] = useState<WsStatus>("connecting");
  const attempts = useRef(0);
  const local = useRef(false);

  // --- WebSocket with limited retries, then hand over to local simulation
  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      if (attempts.current >= MAX_WS_ATTEMPTS) { local.current = true; setStatus("local"); return; }
      attempts.current += 1;
      setStatus("connecting");
      try { ws = new WebSocket(wsUrl()); } catch { local.current = true; setStatus("local"); return; }
      ws.onopen = () => { attempts.current = 0; local.current = false; setStatus("live"); };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data) as TickMsg;
        if (m.type !== "ticks") return;
        setTicks((prev) => { const next = { ...prev }; for (const t of m.ticks) next[t.ticker] = t; return next; });
      };
      ws.onclose = () => { if (!closed) timer = window.setTimeout(connect, 800); };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { closed = true; window.clearTimeout(timer); ws?.close(); };
  }, []);

  // --- local simulation: p' = p + 0.05 (close - p) + close * sigma * eps
  useEffect(() => {
    if (status !== "local" || stocks.length === 0) return;
    const cur: Record<string, number> = {};
    for (const s of stocks) cur[s.ticker] = s.last;
    const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const id = window.setInterval(() => {
      setTicks(() => {
        const next: TickMap = {};
        for (const s of stocks) {
          const sigma = Math.max(0.0002, Math.abs(s.pct_change) / 100 / 8) || 0.0004;
          cur[s.ticker] = cur[s.ticker] + 0.05 * (s.last - cur[s.ticker]) + s.last * sigma * gauss();
          const p = cur[s.ticker];
          next[s.ticker] = { ticker: s.ticker, price: +p.toFixed(2), change: +(p - s.prev_close).toFixed(2), pct_change: +((p / s.prev_close - 1) * 100).toFixed(3) };
        }
        return next;
      });
    }, 1000);
    return () => window.clearInterval(id);
  }, [status, stocks]);

  return { ticks, status };
}
