"""
Tests a "staged" version of the 200-DMA buffer, suggested for comparison: instead of a binary
switch that requires the full buffer move before doing anything (trend_buffer), invest only a
partial fraction (staged_split) immediately at the ordinary crossing, and commit the rest only
once price moves the full buffer beyond the average -- symmetrically for exits, sell a partial
fraction first and the remainder only once price breaks the full buffer below.

Concretely: exposure is 0% while price is more than staged_buffer below the 200-DMA,
staged_split while within +/-staged_buffer of it, and 100% once more than staged_buffer above.
A name only trades when the zone itself changes, not on every daily price wiggle within a zone.

Sweeps staged_split at buffer=0.08, then staged_buffer at the best split found. Selection uses
only 2021-2024 training data and full-backtest metrics, deliberately excluding the 2025 held-out
test, matching quality_weight_search.py's discipline.

Usage (run from the repo root):
    python validation/staged_buffer_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True,
            quality_weight=0.5, redeploy_leftover=True)


def run_one(close, sma200, buf, split):
    cfg = dict(**BASE, staged_buffer=buf, staged_split=split)
    eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
    s_tr = summarize(eq_tr, tl_tr, ct_tr)
    eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
    s_full = summarize(eq_full, tl_full, ct_full)
    return s_tr, s_full


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    print("=== Split sweep at buffer=0.08 ===")
    rows = []
    for split in [0.25, 0.4, 0.5, 0.6, 0.75]:
        s_tr, s_full = run_one(close, sma200, 0.08, split)
        rows.append({"split": split, "Train CAGR (%)": round(s_tr["cagr_pct"], 2),
                     "Full PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
                     "Full CAGR (%)": round(s_full["cagr_pct"], 2),
                     "Full Sharpe": round(s_full["sharpe"], 3),
                     "Full MDD (%)": round(s_full["mdd_pct"], 2),
                     "Orders": s_full["n_orders"]})
    df1 = pd.DataFrame(rows).set_index("split")
    print(df1.to_string())
    best_split = df1["Train CAGR (%)"].idxmax()
    print(f"\nBest split by train CAGR: {best_split}")

    print(f"\n=== Buffer sweep at split={best_split} ===")
    rows = []
    for buf in [0.03, 0.05, 0.08, 0.12, 0.15]:
        s_tr, s_full = run_one(close, sma200, buf, best_split)
        rows.append({"buffer": buf, "Train CAGR (%)": round(s_tr["cagr_pct"], 2),
                     "Full PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
                     "Full CAGR (%)": round(s_full["cagr_pct"], 2),
                     "Full Sharpe": round(s_full["sharpe"], 3),
                     "Full MDD (%)": round(s_full["mdd_pct"], 2),
                     "Orders": s_full["n_orders"]})
    df2 = pd.DataFrame(rows).set_index("buffer")
    print(df2.to_string())
