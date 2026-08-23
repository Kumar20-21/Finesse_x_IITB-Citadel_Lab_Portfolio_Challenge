"""
Tests a deliberate cash sleeve (cash_buffer in run_backtest()) as the fix for the "9 of 10
funded" pattern traced in the report's Limitations section: with target weights summing to
exactly 100% of portfolio value, the 0.1% transaction cost on every buy is not netted out of
anyone's target, so fully funding all N_HOLDINGS positions structurally needs slightly more
than 100% of available cash -- whichever name is processed last in the buy loop is the one
left short, regardless of its own weight (confirmed: raising the invvol shift constant does not
fix this, see invvol_eps_search.py).

Scaling every target weight down by (1 - cash_buffer) reserves enough headroom to cover that
cost margin by design. This searches a grid of buffer sizes, reporting both the fraction of the
20 quarterly rebalances that fund all 10 names, and the full-period / 2025 holdout performance
impact of deliberately not investing that slice.

Usage (run from the repo root):
    python validation/cash_buffer_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
BUFFER_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05]


def count_full10(close, tl):
    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    rebal_dates = [close.index[close.index >= qs][0] for qs in q_starts]
    shares, counts = {}, []
    rows = tl.sort_values("Date").to_dict("records")
    idx = 0
    for rd in rebal_dates:
        while idx < len(rows) and rows[idx]["Date"] <= rd:
            r = rows[idx]
            t = r["Ticker"]
            shares[t] = shares.get(t, 0) + (r["Shares"] if r["Side"] == "BUY" else -r["Shares"])
            idx += 1
        counts.append(sum(1 for s in shares.values() if s > 0.5))
    return sum(1 for c in counts if c == 10), counts


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    rows = []
    for buf in BUFFER_GRID:
        cfg = dict(**BASE, cash_buffer=buf)
        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)
        eq_test, tl_test, ct_test = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_test = summarize(eq_test, tl_test, ct_test)
        n_full10, counts = count_full10(close, tl_full)

        rows.append({
            "cash_buffer": buf,
            "Quarters w/ full 10": f"{n_full10}/20",
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "2025 Return (%)": round(s_test["abs_ret_pct"], 2),
        })

    print(pd.DataFrame(rows).set_index("cash_buffer").to_string())
