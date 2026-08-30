"""
Compares a two-factor version of the strategy (momentum + low-volatility only) against the
submitted three-factor version (quality included) on the 2025 blind holdout.

Usage:
    python validation/momentum_crash_check.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weight_cap=0.15, redeploy_leftover=True)

if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    n500 = bench.loc["2025-01-01":"2025-12-31", "^CRSLDX"]
    n500_ret = (n500.iloc[-1] / n500.iloc[0] - 1) * 100
    print(f"Nifty 500, 2025: {n500_ret:+.2f}%\n")

    for label, qw in [("Two-factor (no quality)", 0.0), ("Three-factor (quality=0.5, submitted)", 0.5)]:
        cfg = dict(**BASE, quality_weight=qw)
        eq, tl, ct = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s = summarize(eq, tl, ct)
        print(f"{label}: {s['abs_ret_pct']:+.2f}%")
