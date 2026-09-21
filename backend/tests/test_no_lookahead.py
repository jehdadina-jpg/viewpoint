"""
Look-ahead guards. If any of these fail, the backtest is cheating.
"""
import numpy as np
import pandas as pd

from backtest.engine import month_end_rebalance_dates
from data.sample_data import sample_headlines, sample_prices
from optimization.views import sentiment_zscores


def test_zscore_ignores_future_data():
    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(0)
    sent = pd.DataFrame(rng.normal(0, 0.3, (120, 2)), index=idx, columns=["A", "B"])
    as_of = idx[80]
    z1 = sentiment_zscores(sent, as_of, 60)
    tampered = sent.copy()
    tampered.iloc[81:] = 5.0                        # blow up the future
    z2 = sentiment_zscores(tampered, as_of, 60)
    assert np.allclose(z1.values, z2.values)


def test_sample_headlines_depend_only_on_past_prices():
    px = sample_prices(["AAPL", "MSFT"], "2024-01-01", "2024-06-30", seed=1)
    h1 = sample_headlines(px, "2024-03-01", "2024-03-31", seed=3)
    # perturb prices strictly AFTER March: headline stream for March must be identical
    px2 = px.copy()
    mask = px2["date"] > "2024-03-31"
    px2.loc[mask, ["open", "high", "low", "close", "adj_close"]] *= 1.5
    h2 = sample_headlines(px2, "2024-03-01", "2024-03-31", seed=3)
    assert [(r["ticker"], r["published_at"], r["headline"]) for r in h1] == \
           [(r["ticker"], r["published_at"], r["headline"]) for r in h2]


def test_rebalance_dates_are_month_ends_inside_window():
    idx = pd.bdate_range("2023-11-01", "2024-03-31")
    d = month_end_rebalance_dates(idx, "2024-01-01", "2024-02-29")
    assert [x.strftime("%Y-%m-%d") for x in d] == ["2024-01-31", "2024-02-29"]
