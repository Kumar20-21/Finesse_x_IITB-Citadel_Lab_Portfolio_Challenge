"""
Compares the submitted redeployment rule (leftover cash tops up already-funded names, cap not
re-enforced) against two alternatives: leaving the shortfall as idle cash, and re-enforcing the
15% cap during top-up. Reports Train (2021-2024), Full (2021-2025), 2025 holdout, and a
20-quarter rolling excess-return grid.

Usage:
    python validation/redeploy_comparison.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weight_cap=0.15, quality_weight=0.5)
INITIAL_CAPITAL = 1_00_00_000

VARIANTS = {
    "No redeploy (idle cash)": dict(redeploy_leftover=False),
    "Redeploy, cap enforced": dict(redeploy_leftover=True, redeploy_cap=0.15),
    "Redeploy, breach allowed (submitted)": dict(redeploy_leftover=True),
}


def quarterly_grid(close, sma200, bench, cfg):
    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    quarters = [(qs, min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp("2025-12-31"))) for qs in q_starts]
    excess = []
    for qs, qe in quarters:
        b = bench.loc[qs:qe, "^CRSLDX"]
        bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100
        eq, _, _ = run_backtest(close, sma200, qs, qe, **cfg)
        strat_ret = eq["PortfolioValue"].iloc[-1] / INITIAL_CAPITAL * 100 - 100
        excess.append(strat_ret - bench_ret)
    return pd.Series(excess)


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    for label, kw in VARIANTS.items():
        cfg = dict(**BASE, **kw)
        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        eq_25, tl_25, ct_25 = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_25 = summarize(eq_25, tl_25, ct_25)
        grid = quarterly_grid(close, sma200, bench, cfg)
        print(f"{label}")
        print(f"  Train CAGR: {s_tr['cagr_pct']:.2f}%   Full CAGR: {s_full['cagr_pct']:.2f}%   "
              f"Full Sharpe: {s_full['sharpe']:.3f}   2025 holdout: {s_25['abs_ret_pct']:.2f}%")
        print(f"  20-quarter hit-rate: {(grid > 0).mean() * 100:.1f}%   "
              f"worst quarter: {grid.min():+.2f}pp\n")
