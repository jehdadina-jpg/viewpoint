import { useMemo } from "react";
import { dirClass, fmt } from "../api/client";
import type { TickMap } from "../hooks/useTicks";
import type { StockRow } from "../types";

interface Props {
  stocks: StockRow[];
  ticks: TickMap;
  selected: string;
  onSelect: (t: string) => void;
}

/** Scrolling strip of the whole universe. Content is duplicated so the CSS
 *  translateX(-50%) loop is seamless. Hover pauses it; click jumps the app. */
export function TickerTape({ stocks, ticks, selected, onSelect }: Props) {
  const items = useMemo(() => stocks.map((s) => {
    const t = ticks[s.ticker];
    const price = t?.price ?? s.last;
    const pct = t?.pct_change ?? s.pct_change;
    return { ticker: s.ticker, price, pct };
  }), [stocks, ticks]);

  const duration = `${Math.max(30, items.length * 4)}s`;
  const row = (k: string) => items.map((it) => (
    <span key={`${k}-${it.ticker}`} className={`tape-item ${it.ticker === selected ? "selected" : ""}`}
          onClick={() => onSelect(it.ticker)}>
      <span className="t">{it.ticker}</span>
      <span>{fmt.px(it.price)}</span>
      <span className={dirClass(it.pct)}>{it.pct > 0 ? "▲" : it.pct < 0 ? "▼" : "•"} {fmt.pct(it.pct)}</span>
    </span>
  ));

  return (
    <div className="tape">
      <div className="tape-label">VIEWPOINT // SBLV</div>
      <div className="tape-track">
        <div className="tape-scroll" style={{ ["--tape-duration" as string]: duration }}>
          {row("a")}{row("b")}
        </div>
      </div>
    </div>
  );
}
