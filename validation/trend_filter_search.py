"""
Tests alternatives to the 200-day moving-average trend filter. Motivated by a concrete case: in
the submitted strategy, BSE.NS was selected into the top-10 four separate times but every closed
round-trip on it was a loss (-33.1%, -4.1%, -23.5%; combined -60.7%), while BSE.NS itself
returned +784% as a simple buy-and-hold over the same span -- the trend filter's exact,
zero-buffer MA-crossover exit repeatedly whipsawed out of a stock that was, in hindsight, one of
the best performers in the universe.

Tests three families of alternatives against the full 2021-2025 backtest, the 2021-2024 training
window, and the 2025 held-out test (never used for selection):
  - No trend filter at all (sma passed as all-NaN, so the exit/re-entry loop never fires)
  - Different MA lengths (100, 150, 200 [current], 250, 300 days)
  - A buffer zone on the 200-DMA (exit only if price is materially, not just marginally, below)

Usage (run from the repo root):
    python validation/trend_filter_search.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, quality_weight=0.5,
            redeploy_leftover=True)


def bse_check(tl):
    ct_bse_returns = []
    bse = tl[tl["Ticker"] == "BSE.NS"].sort_values("Date")
    shares = 0
    entry_notional = 0
    for _, r in bse.iterrows():
        if r["Side"] == "BUY":
            shares += r["Shares"]
            entry_notional += r["Notional"]
        else:
            if shares > 0:
                ret = r["Price"] / (entry_notional / shares) - 1
                ct_bse_returns.append(ret * 100)
            shares = 0
            entry_notional = 0
    return ct_bse_returns


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()
    sma_none = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    variants = {
        "No trend filter": dict(sma=sma_none, reentry=False),
        "MA-100": dict(sma=close.rolling(100, min_periods=100).mean(), reentry=True),
        "MA-150": dict(sma=close.rolling(150, min_periods=150).mean(), reentry=True),
        "MA-200 (submitted)": dict(sma=sma200, reentry=True),
        "MA-250": dict(sma=close.rolling(250, min_periods=250).mean(), reentry=True),
        "MA-300": dict(sma=close.rolling(300, min_periods=300).mean(), reentry=True),
        "MA-200, 5% buffer": dict(sma=sma200, reentry=True, trend_buffer=0.05),
        "MA-200, 8% buffer": dict(sma=sma200, reentry=True, trend_buffer=0.08),
        "MA-200, 12% buffer": dict(sma=sma200, reentry=True, trend_buffer=0.12),
    }

    rows = []
    for label, v in variants.items():
        sma = v.pop("sma")
        cfg = dict(**BASE, **v)

        eq_tr, tl_tr, ct_tr = run_backtest(close, sma, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)

        eq_full, tl_full, ct_full = run_backtest(close, sma, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full)

        eq_test, tl_test, ct_test = run_backtest(close, sma, "2025-01-01", "2025-12-31", **cfg)
        s_test = summarize(eq_test, tl_test, ct_test)

        bse_rets = bse_check(tl_full)
        bse_combined = sum(bse_rets) if bse_rets else float("nan")

        rows.append({
            "Variant": label,
            "Train21-24 CAGR (%)": round(s_tr["cagr_pct"], 2),
            "Full21-25 PnL (Rs cr)": round(s_full["net_pnl"] / 1e7, 3),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "2025 Return (%)": round(s_test["abs_ret_pct"], 2),
            "BSE combined RT return (%)": round(bse_combined, 1) if bse_rets else "n/a",
            "BSE num round-trips": len(bse_rets),
        })

    print(pd.DataFrame(rows).set_index("Variant").to_string())
