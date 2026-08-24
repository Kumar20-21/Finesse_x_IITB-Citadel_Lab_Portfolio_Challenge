# Finesse x Citadel Portfolio Challenge — Round 2

**Team JP's Portfolio** — Keshav Kumar, Jyoti Pandey

A systematic, rules-based 10-stock equity portfolio strategy over the Nifty 100 / Midcap 100 /
Smallcap 100 universe, backtested 1 Jan 2021 – 31 Dec 2025. See `report/report.pdf` for the full
5-6 page writeup (problem, methodology, results, benchmark comparison, limitations). This README
covers only how to reproduce the code.

## Strategy summary

Every quarter, a composite score combining three price-based factors — 12-1 month momentum,
low-volatility, and a trend-quality factor (R² of the price trend) — is computed for all ~300
eligible stocks. The 10 highest-scoring stocks are selected and sized inverse-volatility, capped
at 15% per name for this initial selection pass; any leftover cash (typically because the
lowest-priority name is structurally underfunded once transaction costs are added) is redeployed
into the names that were funded, which can push a name above 15%. Between rebalances, a 200-day
moving average trend filter can exit (and re-enter) any individual position on any trading day. A
0.1% transaction cost applies to every buy and sell. Full rationale for each design choice,
including three tested-and-rejected alternatives to the redeployment rule, is in
`report/report.pdf`.

## Repository structure

```
src/
  build_universe.py      Build the 300-stock universe from the raw NSE index lists
  download_data.py       Download daily prices (yfinance) for the universe + benchmarks
  backtest_engine.py     Core engine: scoring, selection, weighting, rebalancing, trend filter
  run_backtest.py        Run the official submitted strategy, 2021-2025
  evaluate.py            Compute all required metrics + benchmark comparison
  plot_results.py        Generate every report figure (equity curve, drawdown, quarterly excess)

validation/
  train_test_validation.py    Lock rules on 2021-2024, test blind on 2025 (no re-tuning)
  quarterly_robustness.py     20-quarter rolling-window robustness check
  quality_weight_search.py    Parameter search for the quality-factor weight

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

The `data/` and `results/` folders already contain the exact snapshot used for the submission
(prices as pulled during development, and the resulting backtest output), so `evaluate.py` can
be run immediately without re-downloading anything. To regenerate everything from scratch:

```bash
cd portfolio-strategy   # run all commands from the repo root
python3 src/build_universe.py     # -> data/universe.csv
python3 src/download_data.py      # -> data/prices_close.pkl, prices_volume.pkl, benchmark.pkl
python3 src/run_backtest.py       # -> results/equity_curve.pkl, trade_log.pkl, closed_trades.pkl
python3 src/evaluate.py           # prints all required metrics + benchmark comparison
python3 src/plot_results.py       # -> report/equity_curve.pdf, drawdown.pdf, quarterly_excess.pdf
```

Note: re-running `download_data.py` pulls current data from Yahoo Finance, which can differ
slightly from the original snapshot (adjusted-close series are retroactively updated for any
corporate actions that occur after the original download). The committed `data/*.pkl` files are
the exact data the submitted results were produced from.

### Validation / robustness checks

```bash
python3 validation/train_test_validation.py
python3 validation/quarterly_robustness.py
python3 validation/quality_weight_search.py
python3 validation/liquidity_filter_search.py
python3 validation/invvol_eps_search.py
python3 validation/cash_buffer_search.py
python3 validation/n_holdings_search.py
python3 validation/weight_cap_search.py
python3 validation/redeploy_comparison.py
```

These reproduce the evidence discussed in the report's Methodology and Limitations sections:
the train/2021-24-test/2025 split, the 20-quarter robustness grid (including why a tested
drawdown-breaker variant was rejected), the parameter search confirming the quality-factor
weight is a stable choice rather than an overfit one, and several tested-and-rejected
alternatives to the submitted design (a liquidity/ADV screen, a larger inverse-vol shift
constant, a deliberate cash buffer, fewer than 10 holdings, and a looser concentration cap) --
each script's docstring explains what was tested and why it didn't make the cut. The one
alternative that DID make the cut, redeploying leftover cash into already-selected names, is
compared against the two alternatives it beat in `redeploy_comparison.py`.

## Data sources

- **Universe constituents**: NSE index constituent lists (Nifty 100, Nifty Midcap 100, Nifty
  Smallcap 100), snapshotted in `data/raw_index_lists/`.
- **Prices and benchmarks**: Yahoo Finance via the `yfinance` package (NSE-adjusted daily
  closes, dividend/split adjusted; benchmark tickers `^CRSLDX` for Nifty 500 and `^CNX100` for
  Nifty 100).

## Key results (2021-2025 backtest)

| Metric | Value |
|---|---|
| Total Net PnL | Rs 7.13 crore |
| Annualised Return (CAGR) | 52.10% |
| Maximum Drawdown | -25.85% |
| Sharpe Ratio (rf=0%) | 2.71 |
| Gain-to-Loss Ratio | 2.26 |
| Accuracy | 54.10% |
| Benchmark (Nifty 500) CAGR | 15.36% |

Full metric list, methodology, and discussion of limitations are in `report/report.pdf`.
