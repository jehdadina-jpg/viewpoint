import { CandlestickSeries, ColorType, CrosshairMode, HistogramSeries, LineSeries, createChart, type IChartApi, type ISeriesApi, type Time } from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { api, fmt } from "../api/client";
import { useApi } from "../hooks/useApi";
import type { StockRow } from "../types";
import { Panel } from "./Panel";

interface Props { ticker: string; stock?: StockRow; livePrice?: number }

const RANGES: [string, number][] = [["3M", 63], ["6M", 126], ["1Y", 252], ["2Y", 504], ["MAX", 2000]];

/**
 * Candlestick chart with sentiment overlay:
 *   - background bands: a full-height histogram series on a hidden price scale,
 *     coloured green/red with opacity proportional to |sentiment|
 *   - sentiment line on the LEFT axis (-1..+1) with a zero line
 *   - price candles on the RIGHT axis
 */
export function PriceChart({ ticker, stock, livePrice }: Props) {
  const [days, setDays] = useState(252);
  const { data, error, loading } = useApi(() => api.history(ticker, days), [ticker, days]);
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const candles = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const bands = useRef<ISeriesApi<"Histogram"> | null>(null);
  const sentLine = useRef<ISeriesApi<"Line"> | null>(null);
  const [hover, setHover] = useState<{ c?: number; s?: number | null; n?: number; time?: string } | null>(null);

  // build chart once
  useEffect(() => {
    if (!host.current) return;
    const el = host.current;
    const ch = createChart(el, {
      layout: { background: { type: ColorType.Solid, color: "#0c0c0c" }, textColor: "#8a8a8a",
                fontFamily: "JetBrains Mono, IBM Plex Mono, monospace", fontSize: 10 },
      grid: { vertLines: { color: "#161616" }, horzLines: { color: "#161616" } },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: "#ff9900", width: 1, style: 3, labelBackgroundColor: "#ff9900" },
                   horzLine: { color: "#ff9900", width: 1, style: 3, labelBackgroundColor: "#ff9900" } },
      rightPriceScale: { borderColor: "#262626", scaleMargins: { top: 0.08, bottom: 0.08 } },
      leftPriceScale: { visible: true, borderColor: "#262626", scaleMargins: { top: 0.55, bottom: 0.02 } },
      timeScale: { borderColor: "#262626", timeVisible: false, rightOffset: 3 },
      handleScroll: true, handleScale: true,
    });
    // bands first so they render behind everything else
    const b = ch.addSeries(HistogramSeries, { priceScaleId: "bands", priceFormat: { type: "volume" }, lastValueVisible: false, priceLineVisible: false, base: 0 });
    ch.priceScale("bands").applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
    const c = ch.addSeries(CandlestickSeries, {
      upColor: "#39ff14", downColor: "#ff3333", borderUpColor: "#39ff14", borderDownColor: "#ff3333",
      wickUpColor: "#39ff14", wickDownColor: "#ff3333", priceLineColor: "#ff9900",
    });
    const s = ch.addSeries(LineSeries, { priceScaleId: "left", color: "#ff9900", lineWidth: 1, lastValueVisible: true, priceLineVisible: false,
                                         crosshairMarkerVisible: true, title: "SENT",
                                         autoscaleInfoProvider: () => ({ priceRange: { minValue: -1, maxValue: 1 } }) });
    s.createPriceLine({ price: 0, color: "#3a3a3a", lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "" });
    chart.current = ch; candles.current = c; bands.current = b; sentLine.current = s;

    const ro = new ResizeObserver(() => { ch.applyOptions({ width: el.clientWidth, height: el.clientHeight }); ch.timeScale().fitContent(); });
    ro.observe(el);
    ch.applyOptions({ width: el.clientWidth, height: el.clientHeight });
    return () => { ro.disconnect(); ch.remove(); chart.current = null; };
  }, []);

  // load data
  useEffect(() => {
    if (!data || !chart.current || !candles.current || !bands.current || !sentLine.current) return;
    candles.current.setData(data.candles.map((k) => ({ time: k.time as Time, open: k.open, high: k.high, low: k.low, close: k.close })));
    const sentByTime = new Map(data.sentiment.map((p) => [p.time, p]));
    bands.current.setData(data.candles.map((k) => {
      const p = sentByTime.get(k.time);
      const s = p?.score ?? null;
      const a = s == null ? 0 : Math.min(0.28, 0.06 + 0.32 * Math.min(1, Math.abs(s) / 0.5));
      const color = s == null ? "rgba(0,0,0,0)" : s >= 0 ? `rgba(57,255,20,${a})` : `rgba(255,51,51,${a})`;
      return { time: k.time as Time, value: 1, color };
    }));
    sentLine.current.setData(data.sentiment.filter((p) => p.score != null).map((p) => ({ time: p.time as Time, value: p.score as number })));
    chart.current.timeScale().fitContent();

    const ch = chart.current;
    const onMove = (param: { time?: Time; seriesData: Map<unknown, unknown> }) => {
      if (!param.time) { setHover(null); return; }
      const cd = param.seriesData.get(candles.current) as { close?: number } | undefined;
      const p = sentByTime.get(String(param.time));
      setHover({ c: cd?.close, s: p?.score ?? null, n: p?.n_headlines, time: String(param.time) });
    };
    ch.subscribeCrosshairMove(onMove);
    return () => ch.unsubscribeCrosshairMove(onMove);
  }, [data]);

  const lastSent = data?.sentiment.filter((p) => p.score != null).at(-1);
  const legend = hover ?? { c: livePrice ?? stock?.last, s: lastSent?.score ?? null, n: lastSent?.n_headlines, time: undefined };

  return (
    <Panel title={`${ticker}`} sub={<>{stock?.name} | {stock?.sector} | {data?.price_source ?? ""}</>}
           tools={RANGES.map(([l, d]) => <button key={l} className={d === days ? "active" : ""} onClick={() => setDays(d)}>{l}</button>)}>
      {error && <div className="err-msg">{error}</div>}
      {loading && !data && <div className="loading">loading {ticker}</div>}
      <div className="chart-legend">
        <b>{ticker}</b>
        <span>PX <span className="amber">{fmt.px(legend.c)}</span></span>
        <span>SENT <span style={{ color: legend.s == null ? "var(--fg-mute)" : legend.s >= 0 ? "var(--green)" : "var(--red)" }}>{fmt.sent(legend.s)}</span></span>
        <span>N <span className="dim">{legend.n ?? "--"}</span></span>
        {legend.time && <span className="dim">{legend.time}</span>}
        <span><span className="legend-band" style={{ background: "rgba(57,255,20,0.25)" }} />POS BAND</span>
        <span><span className="legend-band" style={{ background: "rgba(255,51,51,0.25)" }} />NEG BAND</span>
        <span><span className="legend-swatch" style={{ background: "#ff9900" }} />ENSEMBLE SENT (L)</span>
      </div>
      <div ref={host} className="chart-wrap" />
    </Panel>
  );
}
