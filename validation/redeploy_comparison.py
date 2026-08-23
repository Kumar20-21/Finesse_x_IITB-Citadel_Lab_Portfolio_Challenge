"""
Full side-by-side of the submitted baseline vs. the two redeploy_leftover variants
(cap-enforced top-up vs. cap-breach-allowed top-up): full 2021-2025 backtest, the 2021-2024
training window, both genuine holdouts (2025, 2026 H1), AND the 20-independent-quarter rolling
grid (same method as quarterly_robustness.py) so a single lucky/unlucky path isn't mistaken for
a real edge.

Usage (run from the repo root):
    python validation/redeploy_comparison.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize, print_summary  # noqa: E402

BASE = dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True, quality_weight=0.5)
INITIAL_CAPITAL = 1_00_00_000

VARIANTS = {
    "Baseline (submitted)": {},
    "Redeploy, cap enforced": dict(redeploy_leftover=True, redeploy_cap=0.15),
    "Redeploy, breach allowed": dict(redeploy_leftover=True),
}


def quarterly_grid(close, sma200, bench, cfg):
    q_starts = pd.date_range("2021-01-01", "2025-10-01", freq="QS")
    quarters = [(qs, min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp("2025-12-31"))) for qs in q_starts]
    excess = []
    for qs, qe in quarters:
        sim_days = close.index[(close.index >= qs) & (close.index <= qe)]
        if len(sim_days) < 5:
            continue
        b = bench.loc[sim_days[0]:sim_days[-1], "^CRSLDX"]
        bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100
        eq, _, _ = run_backtest(close, sma200, qs, qe, **cfg)
        strat_ret = eq["PortfolioValue"].iloc[-1] / INITIAL_CAPITAL * 100 - 100
        excess.append(strat_ret - bench_ret)
    excess = pd.Series(excess)
    return {
        "hit_rate_pct": (excess > 0).mean() * 100,
        "avg_excess_pp": excess.mean(),
        "worst_quarter_excess_pp": excess.min(),
        "best_quarter_excess_pp": excess.max(),
    }


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    summary_rows = []
    for label, kw in VARIANTS.items():
        cfg = dict(**BASE, **kw)

        eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        s_tr = summarize(eq_tr, tl_tr, ct_tr)

        eq_full, tl_full, ct_full = run_backtest(close, sma200, "2021-01-01", "2025-12-31", **cfg)
        s_full = summarize(eq_full, tl_full, ct_full, label=label)
        print_summary(s_full)

        eq_25, tl_25, ct_25 = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        s_25 = summarize(eq_25, tl_25, ct_25)

        eq_h1, tl_h1, ct_h1 = run_backtest(close, sma200, "2026-01-01", "2026-06-30", **cfg)
        s_h1 = summarize(eq_h1, tl_h1, ct_h1)

        grid = quarterly_grid(close, sma200, bench, cfg)

        print(f"   Train21-24 CAGR: {s_tr['cagr_pct']:.2f}%   2025 holdout: {s_25['abs_ret_pct']:.2f}%   "
              f"2026H1 holdout: {s_h1['abs_ret_pct']:.2f}%")
        print(f"   20-quarter grid -> hit-rate: {grid['hit_rate_pct']:.1f}%   "
              f"avg excess: {grid['avg_excess_pp']:+.2f}pp   "
              f"worst quarter: {grid['worst_quarter_excess_pp']:+.2f}pp   "
              f"best quarter: {grid['best_quarter_excess_pp']:+.2f}pp")
        print()

        summary_rows.append({
            "Variant": label,
            "Train21-24 CAGR (%)": round(s_tr["cagr_pct"], 2),
            "Full21-25 CAGR (%)": round(s_full["cagr_pct"], 2),
            "Full21-25 Sharpe": round(s_full["sharpe"], 3),
            "Full21-25 MDD (%)": round(s_full["mdd_pct"], 2),
            "Orders": s_full["n_orders"],
            "2025 holdout (%)": round(s_25["abs_ret_pct"], 2),
            "2026H1 holdout (%)": round(s_h1["abs_ret_pct"], 2),
            "Q-hit-rate (%)": round(grid["hit_rate_pct"], 1),
            "Q-avg-excess (pp)": round(grid["avg_excess_pp"], 2),
            "Q-worst (pp)": round(grid["worst_quarter_excess_pp"], 2),
        })

    print("=== Summary ===")
    print(pd.DataFrame(summary_rows).set_index("Variant").T.to_string())
