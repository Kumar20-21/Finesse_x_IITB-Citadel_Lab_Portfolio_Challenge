"""
Tests whether screening out illiquid names before scoring improves the strategy. Motivation:
thinly-traded stocks can have stale prices, which artificially understates measured volatility
-- so the Low-Vol factor and inverse-vol weighting could be systematically favouring and
over-sizing illiquid names rather than genuinely low-risk ones.

Screens out, at each rebalance, any stock whose trailing 20-day average daily traded value
(price x volume) falls below a given percentile of that day's eligible universe, then re-runs
the full 2021-2025 backtest and the 2025 blind test at several percentile thresholds.

Usage (run from the repo root):
    python validation/liquidity_filter_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
THRESHOLDS = [None, 0.05, 0.10, 0.20, 0.30]


def run_one(close, sma200, volume, start, end, pctile):
    cfg = dict(**BASE)
    if pctile is not None:
        cfg.update(volume=volume, min_adv_pctile=pctile)
    eq, tl, ct = run_backtest(close, sma200, start, end, **cfg)
    return summarize(eq, tl, ct)


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    volume = pd.read_pickle("data/prices_volume.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    rows = []
    for pctile in THRESHOLDS:
        full = run_one(close, sma200, volume, "2021-01-01", "2025-12-31", pctile)
        test = run_one(close, sma200, volume, "2025-01-01", "2025-12-31", pctile)
        rows.append({
            "min_adv_pctile": pctile if pctile is not None else "off",
            "Full21-25 PnL (Rs cr)": round(full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(full["sharpe"], 3),
            "Full21-25 MDD (%)": round(full["mdd_pct"], 2),
            "2025 Return (%)": round(test["abs_ret_pct"], 2),
        })

    print(pd.DataFrame(rows).set_index("min_adv_pctile").to_string())
