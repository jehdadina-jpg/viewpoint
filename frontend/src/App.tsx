import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { BacktestPanel } from "./components/BacktestPanel";
import { CommandBar } from "./components/CommandBar";
import { NewsFeed } from "./components/NewsFeed";
import { PortfolioPanel } from "./components/PortfolioPanel";
import { PriceChart } from "./components/PriceChart";
import { StatsStrip } from "./components/StatsStrip";
import { TickerTape } from "./components/TickerTape";
import { Watchlist } from "./components/Watchlist";
import { useApi } from "./hooks/useApi";
import { useTicks } from "./hooks/useTicks";

export default function App() {
  const meta = useApi(() => api.meta());
  const stocks = useApi(() => api.stocks());
  const portfolio = useApi(() => api.portfolio());
  const backtest = useApi(() => api.backtest());
  const stats = useApi(() => api.stats());
  // URL params for deep-linking / demos: ?t=NVDA&regime=1
  const params = new URLSearchParams(location.search);
  const [selected, setSelected] = useState<string>((params.get("t") ?? "NVDA").toUpperCase());

  // default to the largest BL weight once the portfolio loads
  useEffect(() => {
    if (portfolio.data && stocks.data && !stocks.data.stocks.some((s) => s.ticker === selected)) {
      const top = Object.entries(portfolio.data.weights).sort((a, b) => b[1] - a[1])[0]?.[0];
      if (top) setSelected(top);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portfolio.data, stocks.data]);

  const universe = meta.data?.universe ?? stocks.data?.stocks.map((s) => s.ticker) ?? [];
  const rows = stocks.data?.stocks ?? [];
  const { ticks, status } = useTicks(rows);
  const cur = rows.find((s) => s.ticker === selected);

  const onCommand = useCallback((cmd: string): boolean => {
    const scrollTo = (sel: string) => document.querySelector(sel)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    switch (cmd) {
      case "BT": scrollTo(".area-bt"); return true;
      case "PORT": scrollTo(".area-port"); return true;
      case "NEWS": scrollTo(".area-main"); return true;
      case "TOP": {
        const top = portfolio.data ? Object.entries(portfolio.data.weights).sort((a, b) => b[1] - a[1])[0]?.[0] : undefined;
        if (top) setSelected(top);
        return !!top;
      }
      default: return false;
    }
  }, [portfolio.data]);

  const firstError = meta.error || stocks.error || portfolio.error || backtest.error || stats.error;

  return (
    <div className="app">
      <div className="area-tape"><TickerTape stocks={rows} ticks={ticks} selected={selected} onSelect={setSelected} /></div>
      <div className="area-cmd"><CommandBar universe={universe} selected={selected} onSelect={setSelected} onCommand={onCommand} wsStatus={status} meta={meta.data} /></div>

      <Watchlist stocks={rows} ticks={ticks} selected={selected} onSelect={setSelected} asOf={stocks.data?.as_of} />

      <div className="area-main">
        {firstError && rows.length === 0
          ? <div className="panel"><div className="err-msg">BACKEND UNREACHABLE: {firstError}<br />start it with: <span className="amber">cd backend && uvicorn api.main:app --port 8000</span></div></div>
          : <PriceChart ticker={selected} stock={cur} livePrice={ticks[selected]?.price} />}
        <NewsFeed ticker={selected} />
      </div>

      <PortfolioPanel portfolio={portfolio.data} stats={stats.data} selected={selected} onSelect={setSelected} />
      <BacktestPanel bt={backtest.data} initialRegime={params.get("regime") === "1"} />
      <StatsStrip stats={stats.data} />
    </div>
  );
}
