"""
22-independent-quarterly-window robustness check. Reduces reliance on any single lucky/unlucky
period: each of the 20 calendar quarters from 2021 Q1 to 2025 Q4 is run as its own fresh-capital
backtest, so the resulting hit-rate and excess-return statistics are a much larger sample than
a single full-period or single held-out-year comparison. Also includes the "Final + breaker"
variant (a portfolio-level drawdown circuit breaker) that was tested and rejected: it improves
this quarterly diagnostic but reduces every headline metric in the full 2021-2025 backtest,
so it was left out of the final design (see the report's Limitations section).

Usage (run from the repo root):
    python validation/quarterly_robustness.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest  # noqa: E402

CANDIDATES = {
    "Equal-weight": dict(mom_weight=0.5, weighting="equal", reentry=False),
    "Momentum-tilted": dict(mom_weight=0.6, weighting="score", weight_cap=0.20, reentry=True),
    "Risk-weighted": dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True),
    "Final (submitted)": dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True,
                               quality_weight=0.5),
    "Final + breaker (rejected)": dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True,
                                        quality_weight=0.5, dd_breaker=0.08, dd_breaker_frac=0.5),
}
INITIAL_CAPITAL = 1_00_00_000


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    quarters = [(qs, min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp("2025-12-31"))) for qs in q_starts]

    rows = []
    for qs, qe in quarters:
        sim_days = close.index[(close.index >= qs) & (close.index <= qe)]
        if len(sim_days) < 5:
            continue
        b = bench.loc[sim_days[0]:sim_days[-1], "^CRSLDX"]
        row = {"Quarter": f"{qs.year}Q{(qs.month - 1) // 3 + 1}",
               "Benchmark(Nifty500)": (b.iloc[-1] / b.iloc[0] - 1) * 100}
        for name, cfg in CANDIDATES.items():
            eq, tl, ct = run_backtest(close, sma200, qs, qe, **cfg)
            row[name] = eq["PortfolioValue"].iloc[-1] / INITIAL_CAPITAL * 100 - 100
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Quarter")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(df)
    print()
    for name in CANDIDATES:
        excess = df[name] - df["Benchmark(Nifty500)"]
        print(f"{name:28s} hit-rate {100 * (excess > 0).mean():5.1f}%   "
              f"avg excess {excess.mean():+6.2f}pp   worst quarter {df[name].min():+6.2f}%")
