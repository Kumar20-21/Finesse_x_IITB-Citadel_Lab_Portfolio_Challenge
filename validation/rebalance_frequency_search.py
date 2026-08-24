"""
Tests three alternatives to the daily 200-DMA trend-filter check, suggested as a comparison
point: (1) quarterly-only rebalancing with no trend filter at all, (2) the trend filter checked
only weekly/monthly instead of daily, and (3) the 200-DMA with a buffer/delta around it (already
implemented as trend_buffer and tested in trend_filter_search.py -- reused here for a single
side-by-side table rather than re-run).

Tests against the full 2021-2025 backtest, the 2021-2024 training window, and the 2025 held-out
test (never used for selection).

Usage (run from the repo root):
    python validation/rebalance_frequency_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, quality_weight=0.5,
            redeploy_leftover=True, reentry=True)

VARIANTS = {
    "Submitted (daily trend check)": {},
    "Quarterly-only (no trend filter)": dict(use_trend_filter=False),
    "Trend check weekly (every 5 days)": dict(trend_check_every=5),
    "Trend check monthly (every 21 days)": dict(trend_check_every=21),
    "200-DMA + 5% buffer": dict(trend_buffer=0.05),
    "200-DMA + 8% buffer": dict(trend_buffer=0.08),
    "200-DMA + 12% buffer": dict(trend_buffer=0.12),
}


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    rows = []
    for label, kw in VARIANTS.items():
        cfg = dict(**BASE, **kw)

        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)

        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)

        eq_test, tl_test, ct_test = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_test = summarize(eq_test, tl_test, ct_test)

        rows.append({
            "Variant": label,
            "Train21-24 CAGR (%)": round(s_tr["cagr_pct"], 2),
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "Orders": s_full["n_orders"],
            "2025 Return (%)": round(s_test["abs_ret_pct"], 2),
        })

    print(pd.DataFrame(rows).set_index("Variant").to_string())
