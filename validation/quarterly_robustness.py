"""
Runs the submitted strategy as 20 independent, non-overlapping calendar-quarter backtests
(2021Q1-2025Q4) and reports the excess return over Nifty 500 in each one. Backs the report's
quarterly_excess.pdf chart and hit-rate claim.

Usage:
    python validation/quarterly_robustness.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest  # noqa: E402

CONFIG = dict(mom_weight=0.5, weight_cap=0.15, quality_weight=0.5, redeploy_leftover=True)
INITIAL_CAPITAL = 1_00_00_000

if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    quarters = [(qs, min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp("2025-12-31"))) for qs in q_starts]

    excess = []
    for qs, qe in quarters:
        b = bench.loc[qs:qe, "^CRSLDX"]
        bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100
        eq, _, _ = run_backtest(close, sma200, qs, qe, **CONFIG)
        strat_ret = eq["PortfolioValue"].iloc[-1] / INITIAL_CAPITAL * 100 - 100
        label = f"{qs.year}Q{(qs.month - 1) // 3 + 1}"
        excess.append(strat_ret - bench_ret)
        print(f"{label}: {strat_ret - bench_ret:+.2f}pp")

    excess = pd.Series(excess)
    print(f"\nHit rate: {(excess > 0).mean() * 100:.1f}%  (n={len(excess)})")
