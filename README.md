# Finesse x IITB-Citadel Lab Portfolio Challenge — Round 2

**Team JP's Portfolio** — Keshav Kumar, Jyoti Pandey

A systematic, rules-based 10-stock equity portfolio strategy over the Nifty 100 / Midcap 100 /
Smallcap 100 universe, backtested 1 Jan 2021 – 31 Dec 2025. See `report/report.pdf` for the full
writeup (problem, methodology, results, benchmark comparison, limitations). This README covers
how to reproduce the code.

## Strategy summary

Every quarter, a composite score combining three price-based factors — 12-1 month momentum,
low-volatility, and a trend-quality factor (R² of the price trend) — is computed for all ~300
eligible stocks. The 10 highest-scoring stocks are selected and sized inverse-volatility, capped
at 15% per name for this initial selection pass; any leftover cash is redeployed into the names
that were funded, which can push a name above 15%. A stock already trading below its own 200-day
average on the rebalance date itself is not bought that quarter. A 0.1% transaction cost applies
to every buy and sell. Full rationale is in `report/report.pdf`.

## Repository structure

```
src/
  build_universe.py      Build the 300-stock universe from the raw NSE index lists
  download_data.py       Download daily prices (yfinance) for the universe + benchmarks
  backtest_engine.py     Core engine: scoring, selection, weighting, rebalancing
  run_backtest.py        Run the submitted strategy, 2021-2025
  evaluate.py            Compute all required metrics + benchmark comparison
  plot_results.py        Generate every report figure (equity curve, drawdown, quarterly excess)

validation/
  train_test_validation.py    Train 2021-2024, blind test on 2025
  quarterly_robustness.py     20-quarter rolling-window robustness check
  quality_weight_search.py    Parameter search for the quality-factor weight
  redeploy_comparison.py      Compares the redeployment rule against two alternatives
  momentum_crash_check.py     Two-factor vs. three-factor blind 2025 comparison
  drawdown_breaker_check.py   Tests a drawdown circuit breaker (rejected)

data/
  raw_index_lists/       Raw NSE constituent CSVs for the three indices
  universe.csv           Built by build_universe.py
  prices_close.pkl, prices_volume.pkl, benchmark.pkl    Built by download_data.py

results/
  equity_curve.pkl, trade_log.pkl, closed_trades.pkl    Built by run_backtest.py

report/
  report.tex, report.pdf
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

The `data/` and `results/` folders already contain the exact snapshot used for the submission,
so `evaluate.py` can be run immediately without re-downloading anything. To regenerate everything
from scratch:

```bash
cd portfolio-strategy   # run all commands from the repo root
python3 src/build_universe.py     # -> data/universe.csv
python3 src/download_data.py      # -> data/prices_close.pkl, prices_volume.pkl, benchmark.pkl
python3 src/run_backtest.py       # -> results/equity_curve.pkl, trade_log.pkl, closed_trades.pkl
python3 src/evaluate.py           # prints all required metrics + benchmark comparison
python3 src/plot_results.py       # -> report/equity_curve.pdf, drawdown.pdf, quarterly_excess.pdf
```

Note: re-running `download_data.py` pulls current data from Yahoo Finance, which can differ
slightly from the original snapshot (adjusted-close series are retroactively updated for
corporate actions after the original download). The committed `data/*.pkl` files are the exact
data the submitted results were produced from.

### Validation / robustness checks

```bash
python3 validation/train_test_validation.py
python3 validation/quarterly_robustness.py
python3 validation/quality_weight_search.py
python3 validation/redeploy_comparison.py
python3 validation/momentum_crash_check.py
python3 validation/drawdown_breaker_check.py
```

These reproduce the evidence cited in the report's Methodology and Limitations sections: the
train/2025-holdout split, the 20-quarter robustness grid, the grid search confirming the
quality-factor weight sits in a stable region, the comparison behind the redeployment rule, the
momentum-crash evidence for including the quality factor, and the rejected drawdown breaker.

## Data sources

- **Universe constituents**: NSE index constituent lists (Nifty 100, Nifty Midcap 100, Nifty
  Smallcap 100), snapshotted in `data/raw_index_lists/`.
- **Prices and benchmarks**: Yahoo Finance via the `yfinance` package (NSE-adjusted daily
  closes, dividend/split adjusted; benchmark tickers `^CRSLDX` for Nifty 500 and `^CNX100` for
  Nifty 100).

## Key results (2021-2025 backtest)

| Metric | Value |
|---|---|
| Total Net PnL | Rs 6.37 crore |
| Annualised Return (CAGR) | 49.15% |
| Maximum Drawdown | -30.07% |
| Sharpe Ratio (rf=0%) | 2.35 |
| Gain-to-Loss Ratio | 1.70 |
| Accuracy | 61.62% |
| Benchmark (Nifty 500) CAGR | 15.36% |

Full metric list, methodology, and discussion of limitations are in `report/report.pdf`.
