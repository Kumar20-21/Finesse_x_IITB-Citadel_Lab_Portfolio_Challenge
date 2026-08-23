"""
Train/test validation: lock every candidate design's rules using only 2021-2024 data, then
evaluate each one, unmodified, on 2025 alone as a genuine held-out test. This is the discipline
referenced in the report's Methodology and Limitations sections: design choices were not
selected by looking at how they performed on the very data meant to test generalisation.

Usage (run from the repo root):
    python validation/train_test_validation.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize, print_summary  # noqa: E402

CANDIDATES = {
    "Equal-weight": dict(mom_weight=0.5, weighting="equal", reentry=False),
    "Momentum-tilted": dict(mom_weight=0.6, weighting="score", weight_cap=0.20, reentry=True),
    "Risk-weighted": dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True),
    "Final (submitted)": dict(mom_weight=0.5, weighting="invvol", weight_cap=0.15, reentry=True,
                               quality_weight=0.5, redeploy_leftover=True),
}


if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    bench = pd.read_pickle("data/benchmark.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    print("=" * 90)
    print("TRAIN: 2021-01-01 to 2024-12-31 (in-sample, design comparison only)")
    print("=" * 90)
    for name, cfg in CANDIDATES.items():
        eq, tl, ct = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **cfg)
        print_summary(summarize(eq, tl, ct, label=name))

    print("=" * 90)
    print("TEST: blind evaluation on 2025 alone, rules unchanged from training")
    print("=" * 90)
    for name, cfg in CANDIDATES.items():
        eq, tl, ct = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **cfg)
        print_summary(summarize(eq, tl, ct, label=name))

    b500 = bench.loc["2025-01-01":"2025-12-31", "^CRSLDX"]
    b100 = bench.loc["2025-01-01":"2025-12-31", "^CNX100"]
    print(f"Nifty 500, 2025: {(b500.iloc[-1] / b500.iloc[0] - 1) * 100:+.2f}%")
    print(f"Nifty 100, 2025: {(b100.iloc[-1] / b100.iloc[0] - 1) * 100:+.2f}%")
