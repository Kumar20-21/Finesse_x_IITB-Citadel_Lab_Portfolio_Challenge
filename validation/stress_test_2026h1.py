"""
The report's stated forward-looking stress test (1 January 2026 - 30 June 2026) is no longer
hypothetical: real price data through August 2026 is already in data/prices_close.pkl, so this
window can be evaluated for real. None of this session's experiments (liquidity filter, invvol
eps, cash_buffer, redeploy_leftover, n_holdings) used 2026 data for any decision, so this
remains a genuine, uncontaminated blind test.

Runs the submitted baseline plus every variant tested and rejected/pending in this session as
its own fresh-capital backtest over 2026 H1 alone, and reports each against the Nifty 500 /
Nifty 100 benchmarks over the same window.

Usage (run from the repo root):
    python validation/stress_test_2026h1.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
START, END = "2026-01-01", "2026-06-30"

VARIANTS = {
    "Baseline (submitted)": {},
    "Redeploy, cap enforced": dict(redeploy_leftover=True, redeploy_cap=0.15),
    "Redeploy, breach allowed": dict(redeploy_leftover=True),
    "Liquidity filter (10th pctile)": None,  # needs volume, handled separately
    "invvol_eps=10": dict(invvol_eps=10),
    "cash_buffer=0.01": dict(cash_buffer=0.01),
    "n_holdings=9": dict(n_holdings=9),
    "n_holdings=7": dict(n_holdings=7),
}


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    volume = pd.read_pickle("data/prices_volume.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    b = bench.loc[START:END]
    n500_ret = (b["^CRSLDX"].iloc[-1] / b["^CRSLDX"].iloc[0] - 1) * 100
    n100_ret = (b["^CNX100"].iloc[-1] / b["^CNX100"].iloc[0] - 1) * 100
    print(f"Nifty 500, {START} to {END}: {n500_ret:+.2f}%")
    print(f"Nifty 100, {START} to {END}: {n100_ret:+.2f}%")
    print()

    rows = []
    for label, kw in VARIANTS.items():
        cfg = dict(**BASE)
        if label == "Liquidity filter (10th pctile)":
            cfg.update(volume=volume, min_adv_pctile=0.10)
        else:
            cfg.update(kw)
        eq, tl, ct = run_backtest(close, sma200, START, END, **cfg)
        s = summarize(eq, tl, ct)
        rows.append({
            "Variant": label,
            "Return (%)": round(s["abs_ret_pct"], 2),
            "MDD (%)": round(s["mdd_pct"], 2),
            "Sharpe": round(s["sharpe"], 3) if pd.notna(s["sharpe"]) else float("nan"),
            "vs Nifty500 (pp)": round(s["abs_ret_pct"] - n500_ret, 2),
        })

    print(pd.DataFrame(rows).set_index("Variant").to_string())
