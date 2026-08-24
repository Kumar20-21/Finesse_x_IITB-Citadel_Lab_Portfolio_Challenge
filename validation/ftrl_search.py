"""
Tests Follow-the-Regularized-Leader (FTRL) as an alternative to inverse-volatility weighting,
suggested for comparison. FTRL with an entropic regularizer has the well-known closed form of
the Hedge / multiplicative-weights algorithm: weight on a selected name is proportional to
exp(eta * cumulative composite score), where the cumulative score optionally decays each quarter
(decay=1.0 is textbook infinite-memory FTRL; decay<1 discounts older history). Selection (top-10
by composite score) is unchanged; only the weighting step is replaced.

First sweeps eta at decay=1.0 (coarse, then narrower), then checks decay sensitivity at the best
eta found. Selection uses only 2021-2024 training data and full-backtest metrics, deliberately
excluding the 2025 held-out test, matching quality_weight_search.py's discipline.

Usage (run from the repo root):
    python validation/ftrl_search.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="ftrl", weight_cap=0.15, reentry=True,
            quality_weight=0.5, redeploy_leftover=True)


def run_one(close, sma200, eta, decay):
    cfg = dict(**BASE, ftrl_eta=eta, ftrl_decay=decay)
    eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
    s_tr = summarize(eq_tr, tl_tr, ct_tr)
    eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
    s_full = summarize(eq_full, tl_full, ct_full)
    return s_tr, s_full


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    print("=== Eta sweep at decay=1.0 (textbook FTRL, infinite memory) ===")
    rows = []
    for eta in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        s_tr, s_full = run_one(close, sma200, eta, 1.0)
        rows.append({"eta": eta, "Train CAGR (%)": round(s_tr["cagr_pct"], 2),
                     "Full PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
                     "Full CAGR (%)": round(s_full["cagr_pct"], 2),
                     "Full Sharpe": round(s_full["sharpe"], 3),
                     "Full MDD (%)": round(s_full["mdd_pct"], 2)})
    df1 = pd.DataFrame(rows).set_index("eta")
    print(df1.to_string())
    best_eta = df1["Train CAGR (%)"].idxmax()
    print(f"\nBest eta by train CAGR: {best_eta}")

    print(f"\n=== Decay sweep at eta={best_eta} ===")
    rows = []
    for decay in [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]:
        s_tr, s_full = run_one(close, sma200, best_eta, decay)
        rows.append({"decay": decay, "Train CAGR (%)": round(s_tr["cagr_pct"], 2),
                     "Full PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
                     "Full CAGR (%)": round(s_full["cagr_pct"], 2),
                     "Full Sharpe": round(s_full["sharpe"], 3),
                     "Full MDD (%)": round(s_full["mdd_pct"], 2)})
    df2 = pd.DataFrame(rows).set_index("decay")
    print(df2.to_string())
