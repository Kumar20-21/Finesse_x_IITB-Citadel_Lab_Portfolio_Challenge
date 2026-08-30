"""
Computes required performance/risk/trade metrics and benchmark comparison from the results
of run_backtest.py, plus Alpha/Beta/Information Ratio/Sortino/Calmar.

Usage:
    python src/evaluate.py
"""
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 1_00_00_000


def load_results():
    eq = pd.read_pickle("results/equity_curve.pkl")
    trade_log = pd.read_pickle("results/trade_log.pkl")
    closed_trades = pd.read_pickle("results/closed_trades.pkl")
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    return eq, trade_log, closed_trades, bench


def required_metrics(eq, trade_log, closed_trades, bench):
    end_val = eq["PortfolioValue"].iloc[-1]
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    abs_return = end_val / INITIAL_CAPITAL - 1
    cagr = (end_val / INITIAL_CAPITAL) ** (1 / n_years) - 1
    daily_rets = eq["PortfolioValue"].pct_change().dropna()
    mdd = (eq["PortfolioValue"] / eq["PortfolioValue"].cummax() - 1).min()
    sharpe = cagr / (daily_rets.std() * np.sqrt(252))  # rf = 0%, per the guidelines

    win_rate = closed_trades["Profitable"].mean()
    avg_win = closed_trades.loc[closed_trades["Profitable"], "ReturnPct"].mean()
    avg_loss = closed_trades.loc[~closed_trades["Profitable"], "ReturnPct"].mean()
    gain_loss_ratio = avg_win / abs(avg_loss)

    trades_per_stock = trade_log.groupby("Ticker").size()
    turnover_x = trade_log["Notional"].sum() / INITIAL_CAPITAL

    b500 = bench.loc[eq.index[0]:eq.index[-1], "^CRSLDX"]
    b100 = bench.loc[eq.index[0]:eq.index[-1], "^CNX100"]
    bench500_cagr = (b500.iloc[-1] / b500.iloc[0]) ** (1 / n_years) - 1
    bench100_cagr = (b100.iloc[-1] / b100.iloc[0]) ** (1 / n_years) - 1

    return {
        "Final Portfolio Value (Rs)": end_val,
        "Total Net PnL (Rs)": end_val - INITIAL_CAPITAL,
        "Absolute Return (%)": abs_return * 100,
        "Annualised Return / CAGR (%)": cagr * 100,
        "Maximum Drawdown (%)": mdd * 100,
        "Sharpe Ratio (rf=0%)": sharpe,
        "Gain-to-Loss Ratio": gain_loss_ratio,
        "Accuracy (% profitable trades)": win_rate * 100,
        "Total Orders": len(trade_log),
        "Closed Round-Trip Trades": len(closed_trades),
        "Trades per Stock (avg)": trades_per_stock.mean(),
        "Turnover (x initial capital)": turnover_x,
        "Total Transaction Costs (Rs)": trade_log["Cost"].sum(),
        "Benchmark Nifty 500 CAGR (%)": bench500_cagr * 100,
        "Benchmark Nifty 100 CAGR (%)": bench100_cagr * 100,
        "Outperformance vs Nifty 500, CAGR (pp)": (cagr - bench500_cagr) * 100,
        "Outperformance vs Nifty 100, CAGR (pp)": (cagr - bench100_cagr) * 100,
    }, cagr, mdd, daily_rets


def additional_portfolio_metrics(eq, cagr, mdd, daily_rets, bench):
    """Alpha, Beta, Information Ratio, Sortino and Calmar vs Nifty 500 (rf = 0%)."""
    b500_rets = bench["^CRSLDX"].pct_change().dropna()
    aligned = pd.concat([daily_rets, b500_rets], axis=1, join="inner")
    aligned.columns = ["port", "bench"]

    beta = aligned["port"].cov(aligned["bench"]) / aligned["bench"].var()
    r_squared = aligned["port"].corr(aligned["bench"]) ** 2
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    b500 = bench.loc[eq.index[0]:eq.index[-1], "^CRSLDX"]
    bench_cagr = (b500.iloc[-1] / b500.iloc[0]) ** (1 / n_years) - 1
    alpha = cagr - beta * bench_cagr

    active_rets = aligned["port"] - aligned["bench"]
    tracking_error = active_rets.std() * np.sqrt(252)
    information_ratio = (cagr - bench_cagr) / tracking_error
    downside_dev = np.sqrt((daily_rets.clip(upper=0) ** 2).mean()) * np.sqrt(252)
    sortino = cagr / downside_dev
    calmar = cagr / abs(mdd)

    return {
        "Alpha vs Nifty 500, annualised (%)": alpha * 100,
        "Beta vs Nifty 500": beta,
        "R-Squared vs Nifty 500": r_squared,
        "Tracking Error, annualised (%)": tracking_error * 100,
        "Information Ratio vs Nifty 500": information_ratio,
        "Sortino Ratio (rf=0%)": sortino,
        "Calmar Ratio": calmar,
    }


if __name__ == "__main__":
    eq, trade_log, closed_trades, bench = load_results()
    metrics, cagr, mdd, daily_rets = required_metrics(eq, trade_log, closed_trades, bench)

    print("=== Required metrics (Round 2 guidelines, Section 7) ===")
    for k, v in metrics.items():
        print(f"{k:45s} {v:,.4f}" if isinstance(v, float) else f"{k:45s} {v}")

    print()
    print("=== Additional portfolio-manager metrics (for context) ===")
    for k, v in additional_portfolio_metrics(eq, cagr, mdd, daily_rets, bench).items():
        print(f"{k:45s} {v:,.4f}")
