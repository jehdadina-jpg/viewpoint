// API response shapes - mirror backend/api/main.py

export interface StockRow {
  ticker: string;
  name: string;
  sector: string;
  last: number;
  prev_close: number;
  change: number;
  pct_change: number;
  date: string;
  sentiment: number | null;
  confidence: number | null;
  dispersion: number | null;
  n_headlines: number;
  label: "POS" | "NEG" | "NEU";
  weight: number | null;
  benchmark_weight: number | null;
}
export interface StocksResponse { as_of: string; stocks: StockRow[] }

export interface Candle { time: string; open: number; high: number; low: number; close: number; volume: number }
export interface SentimentPoint { time: string; score: number | null; confidence: number | null; dispersion: number | null; n_headlines: number }
export interface HistoryResponse { ticker: string; name: string; candles: Candle[]; sentiment: SentimentPoint[]; price_source: string }

export interface NewsItem {
  id: number; headline: string; source: string | null; url: string | null; published_at: string;
  provider: string; is_synthetic: boolean; score: number | null; label: "POS" | "NEG" | "NEU";
  model_scores: Record<string, number>;
}
export interface NewsResponse { ticker: string; news: NewsItem[] }

export interface ViewRow { ticker: string; z: number; q: number; confidence: number; target_price: number | null; last_price: number }
export interface PortfolioCurrent {
  as_of: string;
  weights: Record<string, number>;
  benchmark_weights: Record<string, number>;
  bl_no_view_weights: Record<string, number>;
  prior_returns: Record<string, number>;
  posterior_returns: Record<string, number>;
  views: ViewRow[];
  sentiment_z: Record<string, number>;
}

export type PortfolioKey = "bl_sentiment" | "market_cap" | "equal_weight";
export interface Metrics {
  n_days?: number; cumulative_return?: number; annualized_return?: number; annualized_vol?: number;
  sharpe?: number; sortino?: number; max_drawdown?: number; calmar?: number; best_day?: number; worst_day?: number;
  hit_rate?: number; beta?: number; alpha_annualized?: number; tracking_error?: number; information_ratio?: number;
  avg_turnover_per_rebalance?: number; annual_turnover?: number;
}
export type MetricsBlock = Record<PortfolioKey, Metrics>;

export interface Backtest {
  config: { start: string; end: string; rebalance: string; transaction_cost_bps: number; risk_free_annual: number;
            n_rebalances: number; regime_vol_window: number; regime_vol_median_annualized: number };
  dates: string[];
  equity: { gross: Record<PortfolioKey, number[]>; net: Record<PortfolioKey, number[]> };
  metrics: { gross: MetricsBlock; net: MetricsBlock };
  regimes: { labels: ("high_vol" | "low_vol")[]; realized_vol: number[]; high_vol: MetricsBlock; low_vol: MetricsBlock;
             n_days: { high_vol: number; low_vol: number } };
  turnover: Record<PortfolioKey, { date: string; turnover: number }[]>;
  weights_history: ({ date: string } & Record<string, number | string>)[];
  views_history: { date: string; n_views: number; views: { ticker: string; z: number; q: number; confidence: number }[] }[];
}

export interface Stats {
  config: Backtest["config"];
  net: MetricsBlock; gross: MetricsBlock;
  regimes: { high_vol: MetricsBlock; low_vol: MetricsBlock; n_days: { high_vol: number; low_vol: number } };
}

export interface Meta {
  built_at: string; universe: string[]; last_price_date: string;
  data_sources: { prices: string; market_caps: string; news: string; sentiment_models: string[]; synthetic_news_share: number | null };
  backtest_window: { start: string; end: string };
  black_litterman: Record<string, number>; optimizer: Record<string, number | boolean>; backtest: Record<string, number | string>;
  sentiment: { aggregation: Record<string, number>; model_weights: Record<string, number> };
}

export interface Tick { ticker: string; price: number; change: number; pct_change: number }
export interface TickMsg { type: "ticks"; simulated: boolean; ts: string; ticks: Tick[] }
