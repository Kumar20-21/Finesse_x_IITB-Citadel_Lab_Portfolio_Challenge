"""
Runs the same seven variants compared in rebalance_frequency_search.py (submitted daily trend
check, quarterly-only/no filter, weekly/monthly check frequency, and 200-DMA buffer deltas)
against the real 1 January - 30 June 2026 window. Real price data through August 2026 is already
in data/prices_close.pkl; none of this session's experiments used this window for any decision,
so it remains a genuine, uncontaminated out-of-sample check.

Internal validation only -- per instruction, 2026 H1 results are not reported in report.tex.

Usage (run from the repo root):
    python validation/rebalance_frequency_2026h1.py
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

START, END = "2026-01-01", "2026-06-30"


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    b = bench.loc[START:END]
    n500 = (b["^CRSLDX"].iloc[-1] / b["^CRSLDX"].iloc[0] - 1) * 100
    n100 = (b["^CNX100"].iloc[-1] / b["^CNX100"].iloc[0] - 1) * 100
    print(f"Nifty 500, {START} to {END}: {n500:+.2f}%")
    print(f"Nifty 100, {START} to {END}: {n100:+.2f}%")
    print()

    rows = []
    for label, kw in VARIANTS.items():
        cfg = dict(**BASE, **kw)
        eq, tl, ct = run_backtest(close, sma200, START, END, **cfg)
        s = summarize(eq, tl, ct)
        rows.append({
            "Variant": label,
            "Return (%)": round(s["abs_ret_pct"], 2),
            "MDD (%)": round(s["mdd_pct"], 2),
            "Sharpe": round(s["sharpe"], 3) if pd.notna(s["sharpe"]) else float("nan"),
            "vs Nifty500 (pp)": round(s["abs_ret_pct"] - n500, 2),
            "Orders": s["n_orders"],
        })

    print(pd.DataFrame(rows).set_index("Variant").to_string())
