import { ColorType, CrosshairMode, HistogramSeries, LineSeries, createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import { fmt } from "../api/client";
import type { Backtest, Metrics, MetricsBlock, PortfolioKey } from "../types";
import { Panel } from "./Panel";

interface Props { bt: Backtest | null }

const SERIES: { key: PortfolioKey; label: string; abbr: string; color: string }[] = [
  { key: "bl_sentiment", label: "SENTIMENT-TILTED BL", abbr: "BL", color: "#ff9900" },
  { key: "market_cap", label: "MARKET-CAP", abbr: "MCAP", color: "#d9d9d9" },
  { key: "equal_weight", label: "EQUAL-WEIGHT", abbr: "EW", color: "#2ec4e6" },
];

const METRIC_ROWS: { k: keyof Metrics; label: string; f: (v?: number) => string; cls?: (v?: number) => string }[] = [
  { k: "annualized_return", label: "ANN RET", f: (v) => fmt.pct((v ?? 0) * 100, 1) },
  { k: "annualized_vol", label: "ANN VOL", f: (v) => fmt.pct((v ?? 0) * 100, 1, false) },
  { k: "sharpe", label: "SHARPE", f: (v) => fmt.ratio(v) },
  { k: "sortino", label: "SORTINO", f: (v) => fmt.ratio(v) },
  { k: "max_drawdown", label: "MAX DD", f: (v) => fmt.pct((v ?? 0) * 100, 1, false), cls: () => "red" },
  { k: "calmar", label: "CALMAR", f: (v) => fmt.ratio(v) },
  { k: "hit_rate", label: "HIT RATE", f: (v) => fmt.pct((v ?? 0) * 100, 0, false) },
  { k: "n_days", label: "DAYS", f: (v) => String(v ?? "--") },
];

function MetricsTable({ block, title }: { block: MetricsBlock; title: string }) {
  return (
    <table className="grid regime" style={{ width: "auto" }}>
      <thead><tr><th>{title}</th>{SERIES.map((s) => <th key={s.key} style={{ color: s.color }}>{s.abbr}</th>)}</tr></thead>
      <tbody>
        {METRIC_ROWS.map((m) => (
          <tr key={m.k}>
            <td className="dim">{m.label}</td>
            {SERIES.map((s) => <td key={s.key} className={m.cls?.(block[s.key]?.[m.k]) ?? ""}>{m.f(block[s.key]?.[m.k])}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/**
 * Full-width backtest section: three overlaid equity curves (growth of 100),
 * high-vol regime shading, gross/net toggle, regime-split table, and a date
 * range selector that zooms the time scale.
 */
export function BacktestPanel({ bt }: Props) {
  const [net, setNet] = useState(true);
  const [showRegime, setShowRegime] = useState(false);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const lines = useRef<Record<string, ISeriesApi<"Line">>>({});
  const regimeBand = useRef<ISeriesApi<"Histogram"> | null>(null);
  const [hover, setHover] = useState<Record<string, number> | null>(null);

  useEffect(() => {
    if (!host.current) return;
    const el = host.current;
    const ch = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: "#0c0c0c" }, textColor: "#8a8a8a", fontFamily: "JetBrains Mono, IBM Plex Mono, monospace", fontSize: 10 },
      grid: { vertLines: { color: "#161616" }, horzLines: { color: "#161616" } },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "#ff9900", width: 1, style: 3, labelBackgroundColor: "#ff9900" }, horzLine: { color: "#ff9900", width: 1, style: 3, labelBackgroundColor: "#ff9900" } },
      rightPriceScale: { borderColor: "#262626", scaleMargins: { top: 0.06, bottom: 0.04 } },
      timeScale: { borderColor: "#262626", rightOffset: 2 },
    });
    const band = ch.addSeries(HistogramSeries, { priceScaleId: "regime", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false, base: 0 });
    ch.priceScale("regime").applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
    regimeBand.current = band;
    for (const s of SERIES) {
      lines.current[s.key] = ch.addSeries(LineSeries, { color: s.color, lineWidth: s.key === "bl_sentiment" ? 2 : 1, priceLineVisible: false, lastValueVisible: true, title: s.label.split(" ")[0] });
    }
    chart.current = ch;
    const ro = new ResizeObserver(() => { ch.applyOptions({ width: el.clientWidth, height: el.clientHeight }); ch.timeScale().fitContent(); });
    ro.observe(el);
    ch.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    return () => { ro.disconnect(); ch.remove(); chart.current = null; };
  }, []);

  useEffect(() => {
    if (!bt || !chart.current) return;
    const eq = net ? bt.equity.net : bt.equity.gross;
    for (const s of SERIES) {
      lines.current[s.key].setData(bt.dates.map((d, i) => ({ time: d as Time, value: eq[s.key][i] * 100 })));
    }
    regimeBand.current?.setData(bt.dates.map((d, i) => ({ time: d as Time, value: 1, color: bt.regimes.labels[i] === "high_vol" ? "rgba(255,51,51,0.10)" : "rgba(0,0,0,0)" })));
    chart.current.timeScale().fitContent();
    if (!from) { setFrom(bt.dates[0]); setTo(bt.dates[bt.dates.length - 1]); }
    const ch = chart.current;
    const onMove = (p: { time?: Time; seriesData: Map<unknown, unknown> }) => {
      if (!p.time) { setHover(null); return; }
      const out: Record<string, number> = {};
      for (const s of SERIES) { const v = p.seriesData.get(lines.current[s.key]) as { value?: number } | undefined; if (v?.value != null) out[s.key] = v.value; }
      out.__t = Number(String(p.time).replace(/-/g, ""));
      setHover(out);
    };
    ch.subscribeCrosshairMove(onMove);
    return () => ch.unsubscribeCrosshairMove(onMove);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bt, net]);

  const applyRange = () => {
    if (!chart.current || !from || !to) return;
    chart.current.timeScale().setVisibleRange({ from: from as Time, to: to as Time });
  };
  const resetRange = () => { if (bt) { setFrom(bt.dates[0]); setTo(bt.dates[bt.dates.length - 1]); } chart.current?.timeScale().fitContent(); };

  const m = useMemo(() => (bt ? (net ? bt.metrics.net : bt.metrics.gross) : null), [bt, net]);
  const last = bt ? (net ? bt.equity.net : bt.equity.gross) : null;

  return (
    <Panel title="Backtest" className="area-bt"
           sub={bt ? `${bt.config.start} to ${bt.config.end} | monthly rebal, ${bt.config.n_rebalances} rebalances | ${bt.config.transaction_cost_bps}bp/side | rf ${(bt.config.risk_free_annual * 100).toFixed(0)}%` : undefined}
           tools={<>
             <button className={net ? "active" : ""} onClick={() => setNet(true)}>Net</button>
             <button className={!net ? "active" : ""} onClick={() => setNet(false)}>Gross</button>
             <span className="mute">|</span>
             <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} style={{ height: 16, padding: "0 4px", fontSize: 10 }} />
             <input type="date" value={to} onChange={(e) => setTo(e.target.value)} style={{ height: 16, padding: "0 4px", fontSize: 10 }} />
             <button onClick={applyRange}>Zoom</button>
             <button onClick={resetRange}>Reset</button>
             <span className="mute">|</span>
             <button className={showRegime ? "active" : ""} onClick={() => setShowRegime((v) => !v)}>Regime Split</button>
           </>}>
      <div style={{ display: "grid", gridTemplateColumns: showRegime ? "minmax(0,1fr) auto" : "minmax(0,1fr)", height: "100%", minHeight: 0 }}>
        <div style={{ position: "relative", minHeight: 0 }}>
          <div className="chart-legend">
            {SERIES.map((s) => {
              const v = hover?.[s.key] ?? (last ? last[s.key][last[s.key].length - 1] * 100 : undefined);
              return <span key={s.key}><span className="legend-swatch" style={{ background: s.color }} />{s.label} <span style={{ color: s.color }}>{v == null ? "--" : v.toFixed(1)}</span></span>;
            })}
            <span><span className="legend-band" style={{ background: "rgba(255,51,51,0.18)" }} />HIGH-VOL REGIME</span>
            <span className="dim">GROWTH OF 100, {net ? "NET" : "GROSS"}</span>
          </div>
          <div ref={host} className="chart-wrap" />
        </div>
        {showRegime && bt && m && (
          <div style={{ overflow: "auto", borderLeft: "1px solid var(--border)", padding: "2px 4px", display: "flex", gap: 6, maxWidth: 620 }}>
            <MetricsTable block={m} title={net ? "FULL NET" : "FULL GROSS"} />
            <MetricsTable block={bt.regimes.high_vol} title={`HI-VOL>${(bt.config.regime_vol_median_annualized * 100).toFixed(0)}%`} />
            <MetricsTable block={bt.regimes.low_vol} title="LO-VOL" />
          </div>
        )}
      </div>
    </Panel>
  );
}
