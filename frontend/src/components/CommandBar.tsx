import { useEffect, useRef, useState } from "react";
import type { WsStatus } from "../hooks/useTicks";
import type { Meta } from "../types";

interface Props {
  universe: string[];
  selected: string;
  onSelect: (t: string) => void;
  onCommand: (cmd: string) => boolean;   // returns true if handled
  wsStatus: WsStatus;
  meta: Meta | null;
}

/**
 * Bloomberg-style command line. Type a ticker + <GO> (Enter) to jump the whole
 * dashboard. Also accepts a few function-key style commands (see fkeys).
 * Global shortcut: "/" focuses the input from anywhere.
 */
export function CommandBar({ universe, selected, onSelect, onCommand, wsStatus, meta }: Props) {
  const [val, setVal] = useState("");
  const [msg, setMsg] = useState<{ text: string; err?: boolean }>({ text: "SBLV <GO> ready | TICKER <GO> | BT PORT NEWS TOP HELP" });
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== ref.current) { e.preventDefault(); ref.current?.focus(); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const run = (raw: string) => {
    const cmd = raw.trim().toUpperCase();
    if (!cmd) return;
    if (universe.includes(cmd)) { onSelect(cmd); setMsg({ text: `${cmd} <GO> loaded` }); }
    else if (onCommand(cmd)) setMsg({ text: `${cmd} <GO>` });
    else if (cmd === "HELP") setMsg({ text: `TICKERS: ${universe.join(" ")} | BT PORT NEWS TOP` });
    else setMsg({ text: `UNKNOWN: ${cmd}  (try HELP)`, err: true });
    setVal("");
  };

  const fkeys: [string, string][] = [["F1", "HELP"], ["F2", "PORT"], ["F3", "BT"], ["F4", "NEWS"], ["F5", "TOP"]];
  const ds = meta?.data_sources;

  return (
    <div className="cmd">
      <span className="prompt">&gt;</span>
      <input ref={ref} value={val} placeholder={`${selected}  <GO>`} spellCheck={false}
             onChange={(e) => setVal(e.target.value)}
             onKeyDown={(e) => { if (e.key === "Enter") run(val); if (e.key === "Escape") setVal(""); }} />
      <div className="fkeys">
        {fkeys.map(([k, c]) => <button key={k} onClick={() => run(c)}><span className="k">{k}</span>{c}</button>)}
      </div>
      <span className={`msg ${msg.err ? "err" : ""}`}>{msg.text}</span>
      <div className="right">
        {ds && (
          <span>PX:<span className="amber">{ds.prices}</span> NEWS:<span className={ds.news === "sample" ? "amber" : "green"}>{ds.news}</span> MODELS:<span className="amber">{ds.sentiment_models.join("+")}</span></span>
        )}
        {meta && <span>EOD {meta.last_price_date}</span>}
        <span><span className={`status-dot ${wsStatus === "live" || wsStatus === "local" ? "live" : wsStatus === "down" ? "err" : ""}`} />{wsStatus === "live" ? "SIM FEED (WS)" : wsStatus === "local" ? "SIM FEED (LOCAL)" : wsStatus.toUpperCase()}</span>
      </div>
    </div>
  );
}
