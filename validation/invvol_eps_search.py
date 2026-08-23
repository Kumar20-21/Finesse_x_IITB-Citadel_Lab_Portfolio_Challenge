"""
Tests whether raising the inverse-volatility weighting's shift constant (eps in
capped_weights()) fixes the near-zero-weight artifact identified in the report's Limitations
section: with the default eps=0.1, the single highest-volatility name in every quarter's top-10
gets a target weight averaging 0.09% (verified: below 0.5% in all 20 of 20 quarters), which
floors to 0-1 shares and leaves the capital as unused cash rather than genuinely deployed.

A larger eps lifts that worst name's weight (since shifted = raw_affinity - min + eps, and eps
is added uniformly, a bigger eps matters proportionally most to the smallest affinity). This
searches a grid of eps values against the full 2021-2025 backtest and the 2021-2024 training
window (2025 kept as a held-out check, not used to pick eps), matching the same discipline as
quality_weight_search.py.

Usage (run from the repo root):
    python validation/invvol_eps_search.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize, composite_scores, \
    select_top_n_with_sector_cap, capped_weights  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
EPS_GRID = [0.1, 1, 3, 5, 10, 20, 30, 50, 100]


def min_weight_stats(close, eps):
    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    rebal_dates = [close.index[close.index >= qs][0] for qs in q_starts]
    mins = []
    for d in rebal_dates:
        scores, vol = composite_scores(close, d, 0.5, quality_weight=0.5)
        target_list = select_top_n_with_sector_cap(scores, {}, 10)
        inv = 1.0 / vol.loc[target_list]
        w = capped_weights(inv, cap=0.15, eps=eps)
        mins.append(w.min())
    return np.mean(mins) * 100, np.min(mins) * 100


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    rows = []
    for eps in EPS_GRID:
        cfg = dict(**BASE, invvol_eps=eps)
        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        eq_test, tl_test, ct_test = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_test = summarize(eq_test, tl_test, ct_test)
        avg_min_w, worst_min_w = min_weight_stats(close, eps)
        rows.append({
            "eps": eps,
            "Avg min weight (%)": round(avg_min_w, 3),
            "Worst min weight (%)": round(worst_min_w, 3),
            "Train21-24 CAGR (%)": round(s_tr["cagr_pct"], 2),
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "2025 Return (%)": round(s_test["abs_ret_pct"], 2),
        })

    print(pd.DataFrame(rows).set_index("eps").to_string())
