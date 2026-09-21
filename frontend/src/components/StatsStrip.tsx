import { fmt } from "../api/client";
import type { Stats } from "../types";

interface Props { stats: Stats | null }

/** Bottom strip: SHARPE | SORTINO | MAX DD | ALPHA | BETA | TURNOVER for the
 *  sentiment-tilted portfolio (net), with the market-cap benchmark in grey. */
export function StatsStrip({ stats }: Props) {
  const bl = stats?.net.bl_sentiment;
  const bm = stats?.net.market_cap;
  const cells: { k: string; v: string; b?: string; cls?: string }[] = [
    { k: "SHARPE", v: fmt.ratio(bl?.sharpe), b: fmt.ratio(bm?.sharpe) },
    { k: "SORTINO", v: fmt.ratio(bl?.sortino), b: fmt.ratio(bm?.sortino) },
    { k: "MAX DD", v: fmt.pct((bl?.max_drawdown ?? 0) * 100, 1, false), b: fmt.pct((bm?.max_drawdown ?? 0) * 100, 1, false), cls: "red" },
    { k: "ALPHA", v: fmt.pct((bl?.alpha_annualized ?? 0) * 100, 2), cls: (bl?.alpha_annualized ?? 0) >= 0 ? "green" : "red" },
    { k: "BETA", v: fmt.ratio(bl?.beta) },
    { k: "TURNOVER", v: fmt.pct((bl?.annual_turnover ?? 0) * 100, 0, false), b: fmt.pct((bm?.annual_turnover ?? 0) * 100, 0, false) },
    { k: "CALMAR", v: fmt.ratio(bl?.calmar), b: fmt.ratio(bm?.calmar) },
    { k: "ANN RET", v: fmt.pct((bl?.annualized_return ?? 0) * 100, 1), b: fmt.pct((bm?.annualized_return ?? 0) * 100, 1) },
    { k: "CUM RET", v: fmt.pct((bl?.cumulative_return ?? 0) * 100, 0), b: fmt.pct((bm?.cumulative_return ?? 0) * 100, 0) },
  ];
  return (
    <div className="strip area-strip">
      <div className="cell lead">SBLV NET</div>
      {cells.map((c) => (
        <div className="cell" key={c.k}>
          <span className="k">{c.k}</span>
          <span className="v" style={c.cls ? { color: `var(--${c.cls})` } : undefined}>{c.v}</span>
          {c.b && <span className="b dim">MCAP {c.b}</span>}
        </div>
      ))}
      <div className="cell dim" style={{ marginLeft: "auto", borderRight: 0 }}>
        <span className="k">Colasanto et al. 2022 | Mantshimuli & Muteba Mwamba 2025 | Creamer 2015</span>
      </div>
    </div>
  );
}
