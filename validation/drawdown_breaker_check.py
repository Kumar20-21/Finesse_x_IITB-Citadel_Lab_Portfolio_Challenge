"""
Tests a portfolio-level drawdown circuit breaker (sell 50% of every holding, once per quarter,
the first time portfolio value falls 8% below its post-rebalance peak) against the submitted
strategy (no breaker), on the full 2021-2025 backtest and a 20-quarter rolling excess-return
grid.

Usage:
    python validation/drawdown_breaker_check.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weight_cap=0.15, quality_weight=0.5, redeploy_leftover=True)
INITIAL_CAPITAL = 1_00_00_000

VARIANTS = {
    "No breaker (submitted)": {},
    "Breaker 8%/50%": dict(dd_breaker=0.08, dd_breaker_frac=0.5),
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
        eq, tl, ct = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s = summarize(eq, tl, ct)
        grid = quarterly_grid(close, sma200, bench, cfg)
        print(f"{label}: PnL={s['net_pnl']/1e7:.3f}cr CAGR={s['cagr_pct']:.2f}% "
              f"MDD={s['mdd_pct']:.2f}% Sharpe={s['sharpe']:.3f} "
              f"worst_quarter={grid.min():+.2f}pp hit_rate={(grid > 0).mean() * 100:.1f}%")
