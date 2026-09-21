import { dirClass, fmt } from "../api/client";
import { useFlash } from "../hooks/useFlash";
import type { TickMap } from "../hooks/useTicks";
import type { StockRow } from "../types";
import { Panel } from "./Panel";

interface Props {
  stocks: StockRow[];
  ticks: TickMap;
  selected: string;
  onSelect: (t: string) => void;
  asOf?: string;
}

function FlashTd({ value, className, children }: { value: number | null | undefined; className?: string; children: React.ReactNode }) {
  const flash = useFlash(value);
  return <td className={`${className ?? ""} ${flash}`}>{children}</td>;
}

/** Sentiment -1..+1 rendered as a colour ramp between red and green. */
export function sentColor(s: number | null | undefined): string {
  if (s == null) return "var(--fg-mute)";
  const a = Math.min(1, Math.abs(s) / 0.6);
  return s >= 0 ? `rgba(57,255,20,${0.35 + 0.65 * a})` : `rgba(255,51,51,${0.35 + 0.65 * a})`;
}

function Row({ s, tick, selected, onSelect }: { s: StockRow; tick?: TickMap[string]; selected: boolean; onSelect: (t: string) => void }) {
  const last = tick?.price ?? s.last;
  const chg = tick?.change ?? s.change;
  const pct = tick?.pct_change ?? s.pct_change;
  return (
    <tr className={`clickable ${selected ? "selected" : ""}`} onClick={() => onSelect(s.ticker)}>
      <td className="amber bold">{s.ticker}</td>
      <FlashTd value={last}>{fmt.px(last)}</FlashTd>
      <FlashTd value={chg} className={dirClass(chg)}>{fmt.num(chg, 2, true)}</FlashTd>
      <FlashTd value={pct} className={dirClass(pct)}>{fmt.pct(pct)}</FlashTd>
      <td style={{ color: sentColor(s.sentiment) }} className="bold">{fmt.sent(s.sentiment)}</td>
      <td className="dim">{s.confidence == null ? "--" : (s.confidence * 100).toFixed(0)}</td>
      <td className="dim">{s.weight == null ? "--" : (s.weight * 100).toFixed(1)}</td>
    </tr>
  );
}

export function Watchlist({ stocks, ticks, selected, onSelect, asOf }: Props) {
  return (
    <Panel title="Watchlist" sub={asOf ? `EOD ${asOf}` : undefined} className="area-watch">
      <table className="grid">
        <thead>
          <tr><th>Ticker</th><th>Last</th><th>Chg</th><th>%Chg</th><th>Sent</th><th>Conf</th><th>Wt%</th></tr>
        </thead>
        <tbody>
          {stocks.length === 0 && <tr><td colSpan={7} className="loading">waiting for /stocks</td></tr>}
          {stocks.map((s) => <Row key={s.ticker} s={s} tick={ticks[s.ticker]} selected={s.ticker === selected} onSelect={onSelect} />)}
        </tbody>
      </table>
    </Panel>
  );
}
