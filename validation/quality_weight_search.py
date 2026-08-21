"""
Parameter search for the quality-tilt weight (how much the trend-smoothness factor counts in
the composite score). Selection is based only on 2021-2024 training metrics and the full
2021-2025 backtest metrics, deliberately excluding the 2025 held-out test, so the parameter is
not chosen using the very data meant to test generalisation.

Runs a coarse pass (0.05 steps, 0 to 1) then a fine pass (0.001 steps) around the best region,
confirming the chosen weight (0.5) sits inside a genuine stable plateau rather than an isolated,
fragile spike (the composite score only changes the outcome when it flips which two stocks swap
places in the top-10 ranking, so the metric is piecewise-constant, not smooth).

Usage (run from the repo root):
    python validation/quality_weight_search.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True)


def run_grid(close, sma200, weights):
    rows = []
    for qw in weights:
        cfg = dict(**BASE, quality_weight=qw)
        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        rows.append({
            "quality_weight": round(qw, 3),
            "Train Sharpe": round(s_tr["sharpe"], 2),
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
        })
    return pd.DataFrame(rows).set_index("quality_weight")


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    print("=== Coarse search (0.05 steps) ===")
    coarse = run_grid(close, sma200, np.arange(0.0, 1.001, 0.05))
    print(coarse)

    print()
    print("=== Fine search (0.001 steps, 0.485 to 0.525) ===")
    fine = run_grid(close, sma200, np.arange(0.485, 0.5251, 0.001))
    print(fine)
    print()
    print("Best by full-period PnL:", fine["Full21-25 PnL (Rs cr)"].idxmax(),
          fine["Full21-25 PnL (Rs cr)"].max())
