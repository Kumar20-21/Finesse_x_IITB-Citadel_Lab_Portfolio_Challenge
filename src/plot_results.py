"""
Generate every chart used in report/report.tex, so the figures are reproducible from the
committed data/results exactly like the numbers (see README).

Produces, all written to report/:
  - equity_curve.pdf      Portfolio value vs. Nifty 500 and Nifty 100, 2021-2025
  - drawdown.pdf           Drawdown from running peak, strategy vs. Nifty 500, 2021-2025
  - quarterly_excess.pdf   Strategy excess return over Nifty 500, per calendar quarter
                           (the 20-quarter robustness grid referenced in the report's
                           Limitations section)

Usage (run from the repo root):
    python src/plot_results.py
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest  # noqa: E402

INITIAL_CAPITAL = 1_00_00_000
FINAL_CONFIG = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)


def plot_equity_curve(eq, bench):
    strategy = eq["PortfolioValue"] / INITIAL_CAPITAL
    bench = bench.reindex(strategy.index, method="ffill")
    n500 = bench["^CRSLDX"] / bench["^CRSLDX"].iloc[0]
    n100 = bench["^CNX100"] / bench["^CNX100"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(strategy.index, strategy.values, label="Strategy", color="#1a3a6b", linewidth=1.4)
    ax.plot(n500.index, n500.values,
            label="Nifty 500", color="#c0522d", linewidth=1.1, linestyle="--")
    ax.plot(n100.index, n100.values,
            label="Nifty 100", color="#4c4c4c", linewidth=1.1, linestyle=":")
    ax.set_ylabel("Portfolio Value (Rs crore)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/equity_curve.pdf")
    plt.close(fig)


def plot_drawdown(eq, bench):
    strategy = eq["PortfolioValue"] / INITIAL_CAPITAL
    n500 = bench["^CRSLDX"].reindex(strategy.index, method="ffill")
    n500 = n500 / n500.iloc[0]

    strategy_dd = strategy / strategy.cummax() - 1
    n500_dd = n500 / n500.cummax() - 1

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                     color="#1a3a6b", alpha=0.35, label="Strategy")
    ax.plot(n500_dd.index, n500_dd.values * 100, color="#c0522d", linewidth=1.1,
            linestyle="--", label="Nifty 500")
    ax.set_ylabel("Drawdown from peak (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/drawdown.pdf")
    plt.close(fig)


def plot_quarterly_excess(close, sma200, bench):
    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    quarters = [(qs, min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp("2025-12-31"))) for qs in q_starts]

    labels, excess = [], []
    for qs, qe in quarters:
        sim_days = close.index[(close.index >= qs) & (close.index <= qe)]
        if len(sim_days) < 5:
            continue
        b = bench.loc[sim_days[0]:sim_days[-1], "^CRSLDX"]
        bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100
        eq, _, _ = run_backtest(close, sma200, qs, qe, **FINAL_CONFIG)
        strat_ret = eq["PortfolioValue"].iloc[-1] / INITIAL_CAPITAL * 100 - 100
        labels.append(f"{qs.year}Q{(qs.month - 1) // 3 + 1}")
        excess.append(strat_ret - bench_ret)

    colors = ["#1a3a6b" if v >= 0 else "#c0522d" for v in excess]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar(labels, excess, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Excess return vs. Nifty 500 (pp)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("report/quarterly_excess.pdf")
    plt.close(fig)


def main():
    eq = pd.read_pickle("results/equity_curve.pkl")
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    plot_equity_curve(eq, bench)
    plot_drawdown(eq, bench)
    plot_quarterly_excess(close, sma200, bench)
    print("Wrote report/equity_curve.pdf, report/drawdown.pdf, report/quarterly_excess.pdf")


if __name__ == "__main__":
    main()
