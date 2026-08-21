"""
Run the official submitted strategy over the full 1 Jan 2021 - 31 Dec 2025 backtest window
and save the results (equity curve, trade log, closed trades) to results/.

Strategy configuration (see report Section: Methodology for the full rationale):
  - Composite score per stock, recomputed every quarter:
        0.5 x [0.5 x Z(12-1 month momentum) + 0.5 x Z(low-volatility)] + 0.5 x Z(trend-quality R^2)
  - Top 10 stocks by composite score are selected each quarter.
  - Positions are sized inverse-volatility, capped at 15% and renormalised.
  - A 200-day moving average trend filter exits (and can re-enter) any position daily,
    independent of the quarterly rebalance schedule.
  - 0.1% transaction cost on every buy and every sell.

Usage:
    python src/run_backtest.py
"""
import pandas as pd

from backtest_engine import run_backtest, summarize, print_summary

INITIAL_CAPITAL = 1_00_00_000
START = "2021-01-01"
END = "2025-12-31"

STRATEGY_CONFIG = dict(
    mom_weight=0.5,
    weighting="invvol",
    weight_cap=0.15,
    reentry=True,
    quality_weight=0.5,
)


def main():
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    eq, trade_log, closed_trades = run_backtest(
        close, sma200, START, END,
        initial_capital=INITIAL_CAPITAL, **STRATEGY_CONFIG,
    )

    eq.to_pickle("results/equity_curve.pkl")
    trade_log.to_pickle("results/trade_log.pkl")
    closed_trades.to_pickle("results/closed_trades.pkl")

    print_summary(summarize(eq, trade_log, closed_trades, label="Official submitted strategy, 2021-2025"))


if __name__ == "__main__":
    main()
