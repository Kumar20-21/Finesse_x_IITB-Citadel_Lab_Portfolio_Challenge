"""
Grid search for the quality-factor weight (0.001 resolution) using only 2021-2024 training
data and the full 2021-2025 backtest, excluding the 2025 holdout from selection.

Usage:
    python validation/quality_weight_search.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weight_cap=0.15)


def run_grid(close, sma200, weights):
    rows = []
    for qw in weights:
        cfg = dict(**BASE, quality_weight=round(qw, 3), redeploy_leftover=True)
        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        rows.append({
            "quality_weight": round(qw, 3),
            "Train Sharpe": round(s_tr["sharpe"], 3),
            "Full PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full Sharpe": round(s_full["sharpe"], 3),
        })
    return pd.DataFrame(rows).set_index("quality_weight")


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    print("=== Fine search (0.001 steps, 0.49 to 0.51) ===")
    fine = run_grid(close, sma200, np.arange(0.49, 0.511, 0.001))
    print(fine.to_string())
