import { api, fmt } from "../api/client";
import { useApi } from "../hooks/useApi";
import { Panel } from "./Panel";

interface Props { ticker: string }

const MODEL_ABBR: Record<string, string> = { finbert: "FB", vader: "VD", lexicon: "LM" };

/** Headline list with ensemble badge + per-model breakdown. */
export function NewsFeed({ ticker }: Props) {
  const { data, error, loading } = useApi(() => api.news(ticker, 80), [ticker]);
  const synthetic = data?.news.some((n) => n.is_synthetic);

  return (
    <Panel title={`News / Sentiment`} sub={<>{ticker} | {data?.news.length ?? 0} items{synthetic && <> | <span className="badge sim">SAMPLE</span></>}</>}>
      {error && <div className="err-msg">{error}</div>}
      {loading && !data && <div className="loading">loading news</div>}
      {data && data.news.length === 0 && <div className="loading" style={{ color: "var(--fg-dim)" }}>NO HEADLINES FOR {ticker} IN CACHE. Run python -m pipeline.run to populate.</div>}
      {data?.news.map((n) => (
        <div className="news-row" key={n.id} title={n.headline}>
          <span className="time">{fmt.ts(n.published_at)}</span>
          <span className={`badge ${n.label.toLowerCase()}`}>{n.label}</span>
          <span className="h">
            {n.url ? <a href={n.url} target="_blank" rel="noreferrer">{n.headline}</a> : n.headline}
            <span className="models dim">
              {fmt.sent(n.score)}{" "}
              {Object.entries(n.model_scores).map(([m, v]) => `${MODEL_ABBR[m] ?? m}:${v >= 0 ? "+" : ""}${v.toFixed(2)}`).join(" ")}
            </span>
          </span>
          <span className="src">{n.source ?? n.provider}</span>
        </div>
      ))}
    </Panel>
  );
}
