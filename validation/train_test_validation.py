"""
Reports the submitted strategy's performance on the 2021-2024 training window and the
2025 blind holdout, with no re-tuning between the two.

Usage:
    python validation/train_test_validation.py
"""
import sys
import pandas as pd

sys.path.insert(0, "src")
from backtest_engine import run_backtest, summarize, print_summary  # noqa: E402

CONFIG = dict(mom_weight=0.5, weight_cap=0.15, quality_weight=0.5, redeploy_leftover=True)

if __name__ == "__main__":
    close = pd.read_pickle("data/prices_close.pkl").sort_index()
    sma200 = close.rolling(200, min_periods=200).mean()

    eq_tr, tl_tr, ct_tr = run_backtest(close, sma200, "2021-01-01", "2024-12-31", **CONFIG)
    print_summary(summarize(eq_tr, tl_tr, ct_tr, label="Train, 2021-2024"))

    eq_25, tl_25, ct_25 = run_backtest(close, sma200, "2025-01-01", "2025-12-31", **CONFIG)
    print_summary(summarize(eq_25, tl_25, ct_25, label="Blind holdout, 2025"))
