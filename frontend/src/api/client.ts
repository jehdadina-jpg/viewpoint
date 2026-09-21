import type { Backtest, HistoryResponse, Meta, NewsResponse, PortfolioCurrent, Stats, StocksResponse } from "../types";

// In dev, Vite proxies /api -> http://127.0.0.1:8000 (vite.config.ts).
// For a static build, set VITE_API_BASE to the backend origin.
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} - ${path}`);
  return (await r.json()) as T;
}

export const api = {
  meta: () => get<Meta>("/meta"),
  stocks: () => get<StocksResponse>("/stocks"),
  history: (t: string, days = 400) => get<HistoryResponse>(`/stock/${t}/history?days=${days}`),
  news: (t: string, limit = 60) => get<NewsResponse>(`/stock/${t}/news?limit=${limit}`),
  portfolio: () => get<PortfolioCurrent>("/portfolio/current"),
  backtest: () => get<Backtest>("/portfolio/backtest"),
  stats: () => get<Stats>("/portfolio/stats"),
};

export function wsUrl(): string {
  const explicit = import.meta.env.VITE_WS_BASE as string | undefined;
  if (explicit) return `${explicit}/ws/ticks`;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/ticks`;
}

// ------------------------------------------------------------ formatting
export const fmt = {
  px: (v: number | null | undefined, d = 2) => (v == null ? "--" : v.toFixed(d)),
  pct: (v: number | null | undefined, d = 2, sign = true) =>
    v == null ? "--" : `${sign && v > 0 ? "+" : ""}${v.toFixed(d)}%`,
  num: (v: number | null | undefined, d = 2, sign = false) =>
    v == null ? "--" : `${sign && v > 0 ? "+" : ""}${v.toFixed(d)}`,
  ratio: (v: number | null | undefined) => (v == null ? "--" : v.toFixed(2)),
  sent: (v: number | null | undefined) => (v == null ? "--" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`),
  ts: (iso: string) => {
    const d = new Date(iso);
    const p = (n: number) => String(n).padStart(2, "0");
    return `${d.getMonth() + 1}/${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  },
};

export const dirClass = (v: number | null | undefined) => (v == null || v === 0 ? "flat" : v > 0 ? "up" : "down");
