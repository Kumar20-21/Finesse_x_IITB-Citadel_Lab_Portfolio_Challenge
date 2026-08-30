"""
Runs the submitted strategy over 2021-01-01 to 2025-12-31 and saves the equity curve, trade
log, and closed trades to results/.

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
    weight_cap=0.15,
    quality_weight=0.5,
    redeploy_leftover=True,
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

    print_summary(summarize(eq, trade_log, closed_trades, label="Submitted strategy, 2021-2025"))


if __name__ == "__main__":
    main()
