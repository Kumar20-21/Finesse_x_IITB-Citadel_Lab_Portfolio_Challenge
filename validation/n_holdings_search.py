"""
Tests whether a smaller basket (fewer than the maximum 10 stocks permitted by the Round 2
rules) improves returns. Motivated by the observation that the strategy already ends up
holding only ~9 names in 16 of 20 quarters regardless of n_holdings=10 (see the report's
Limitations section) -- this checks whether explicitly targeting a smaller, more concentrated
basket from the start (rather than arriving there by execution accident) does any better.

Selection uses only 2021-2024 training data and full-backtest metrics, deliberately excluding
the 2025 held-out test, matching the discipline in quality_weight_search.py.

Usage (run from the repo root):
    python validation/n_holdings_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
N_GRID = [5, 6, 7, 8, 9, 10]


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    rows = []
    for n in N_GRID:
        cfg = dict(**BASE, n_holdings=n)
        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        eq_test, tl_test, ct_test = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_test = summarize(eq_test, tl_test, ct_test)
        rows.append({
            "n_holdings": n,
            "Train21-24 CAGR (%)": round(s_tr["cagr_pct"], 2),
            "Train21-24 Sharpe": round(s_tr["sharpe"], 3),
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "2025 Return (%)": round(s_test["abs_ret_pct"], 2),
        })

    print(pd.DataFrame(rows).set_index("n_holdings").to_string())
