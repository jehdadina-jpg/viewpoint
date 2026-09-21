import { useState } from "react";
import { fmt } from "../api/client";
import type { PortfolioCurrent, Stats } from "../types";
import { Panel } from "./Panel";

interface Props {
  portfolio: PortfolioCurrent | null;
  stats: Stats | null;
  selected: string;
  onSelect: (t: string) => void;
}

type Mode = "tilt" | "bench";

/**
 * Right-hand portfolio panel.
 *  - toggle Sentiment-Tilted (BL posterior weights) vs Benchmark (market-cap prior)
 *  - horizontal amber bars; in tilt mode a thin ghost marker shows the benchmark weight
 *    so the *tilt* itself is visible per name
 *  - active BL views with z-score, view return Q and confidence
 *  - headline stats (net of costs) vs benchmark
 */
export function PortfolioPanel({ portfolio, stats, selected, onSelect }: Props) {
  const [mode, setMode] = useState<Mode>("tilt");
  if (!portfolio) return <Panel title="Portfolio" className="area-port"><div className="loading">loading portfolio</div></Panel>;

  const w = mode === "tilt" ? portfolio.weights : portfolio.benchmark_weights;
  const ghost = portfolio.benchmark_weights;
  const rows = Object.entries(w).sort((a, b) => b[1] - a[1]);
  const maxW = Math.max(...rows.map(([, v]) => v), 0.01);
  const bl = stats?.net.bl_sentiment;
  const bm = stats?.net.market_cap;
  const viewsByTicker = new Map(portfolio.views.map((v) => [v.ticker, v]));

  return (
    <Panel title="Portfolio" className="area-port" sub={`as of ${portfolio.as_of}`}
           tools={<>
             <button className={mode === "tilt" ? "active" : ""} onClick={() => setMode("tilt")}>Sentiment-Tilted</button>
             <button className={mode === "bench" ? "active" : ""} onClick={() => setMode("bench")}>Benchmark</button>
           </>}>
      <div className="pad" style={{ padding: "4px 8px" }}>
        <div className="dim" style={{ fontSize: "var(--fs-xs)", marginBottom: 3, letterSpacing: "0.06em" }}>
          {mode === "tilt" ? "BL POSTERIOR WEIGHTS  (tick = mkt-cap prior)" : "MARKET-CAP PRIOR WEIGHTS (w_mkt)"}
        </div>
        {rows.map(([t, v]) => {
          const view = viewsByTicker.get(t);
          const z = portfolio.sentiment_z[t] ?? 0;
          return (
            <div className="bar-row" key={t} style={{ cursor: "pointer" }} onClick={() => onSelect(t)}>
              <span className={t === selected ? "amber bold" : "amber"}>{t}</span>
              <div className="bar-track">
                <div className={`bar-fill ${mode === "bench" ? "bench" : ""}`} style={{ width: `${(v / maxW) * 100}%` }} />
                {mode === "tilt" && <div className="bar-fill ghost" style={{ width: `${(ghost[t] / maxW) * 100}%` }} />}
              </div>
              <span className="right">
                {(v * 100).toFixed(1)}%
                {mode === "tilt" && <span className={view ? (z > 0 ? "green" : "red") : "mute"} style={{ fontSize: "var(--fs-xs)" }}> {view ? (z > 0 ? "▲" : "▼") : "·"}</span>}
              </span>
            </div>
          );
        })}
      </div>
      <div className="hr" />
      <div style={{ padding: "2px 8px" }}>
        <div className="dim" style={{ fontSize: "var(--fs-xs)", letterSpacing: "0.06em" }}>ACTIVE BL VIEWS ({portfolio.views.length}): z / Q ann / conf / MC target</div>
        <table className="grid">
          <tbody>
            {portfolio.views.sort((a, b) => Math.abs(b.z) - Math.abs(a.z)).map((v) => (
              <tr key={v.ticker} className={`clickable ${v.ticker === selected ? "selected" : ""}`} onClick={() => onSelect(v.ticker)}>
                <td className="amber">{v.ticker}</td>
                <td className={v.z > 0 ? "green" : "red"}>{fmt.num(v.z, 2, true)}</td>
                <td className={v.q > 0 ? "green" : "red"}>{fmt.pct(v.q * 100, 1)}</td>
                <td className="dim">{(v.confidence * 100).toFixed(0)}%</td>
                <td className="dim">{v.target_price ? `→${fmt.px(v.target_price, 0)}` : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="hr" />
      <div style={{ padding: "2px 8px 6px" }}>
        <div className="dim" style={{ fontSize: "var(--fs-xs)", letterSpacing: "0.06em" }}>BACKTEST, NET OF {stats?.config.transaction_cost_bps ?? 10}BP | BL vs MKT-CAP</div>
        <div className="kv">
          <span className="k">Sharpe</span><span className="v"><span className="amber bold">{fmt.ratio(bl?.sharpe)}</span> <span className="dim">/ {fmt.ratio(bm?.sharpe)}</span></span>
          <span className="k">Sortino</span><span className="v"><span className="amber bold">{fmt.ratio(bl?.sortino)}</span> <span className="dim">/ {fmt.ratio(bm?.sortino)}</span></span>
          <span className="k">Max DD</span><span className="v"><span className="red bold">{fmt.pct((bl?.max_drawdown ?? 0) * 100, 1, false)}</span> <span className="dim">/ {fmt.pct((bm?.max_drawdown ?? 0) * 100, 1, false)}</span></span>
          <span className="k">Ann. Ret</span><span className="v"><span className="amber bold">{fmt.pct((bl?.annualized_return ?? 0) * 100, 1)}</span> <span className="dim">/ {fmt.pct((bm?.annualized_return ?? 0) * 100, 1)}</span></span>
          <span className="k">Alpha (ann)</span><span className="v"><span className={(bl?.alpha_annualized ?? 0) >= 0 ? "green bold" : "red bold"}>{fmt.pct((bl?.alpha_annualized ?? 0) * 100, 2)}</span></span>
          <span className="k">Beta</span><span className="v">{fmt.ratio(bl?.beta)}</span>
          <span className="k">Turnover /yr</span><span className="v">{fmt.pct((bl?.annual_turnover ?? 0) * 100, 0, false)}</span>
        </div>
      </div>
    </Panel>
  );
}
