"""
Download all raw price data needed for the backtest: daily OHLCV for every stock in the
universe, plus the two benchmark indices (Nifty 500 and Nifty 100).

Data is pulled from 2019-06-01 onward (well before the 2021-01-01 backtest start) so the
momentum (12-1 month) and trend (200-day) lookback windows have enough history on day one.

Usage:
    python src/download_data.py
"""
import time

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/universe.csv"
START = "2019-06-01"
BATCH_SIZE = 25

BENCHMARKS = {
    "^CRSLDX": "Nifty 500",
    "^CNX100": "Nifty 100",
}


def download_universe_prices(tickers):
    all_close, all_volume, failed = {}, {}, []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        for attempt in range(3):
            try:
                data = yf.download(batch, start=START, group_by="ticker",
                                    auto_adjust=True, threads=True, progress=False)
                break
            except Exception as e:
                print(f"retry batch {i}: {e}")
                time.sleep(3)
        else:
            failed.extend(batch)
            continue

        for t in batch:
            try:
                sub = data if len(batch) == 1 else data[t]
                close = sub["Close"].dropna()
                if close.empty:
                    failed.append(t)
                    continue
                all_close[t] = close
                all_volume[t] = sub["Volume"].dropna()
            except Exception:
                failed.append(t)
        print(f"batch {i}-{i + BATCH_SIZE} done, failed so far: {len(failed)}")
        time.sleep(1)

    return pd.DataFrame(all_close), pd.DataFrame(all_volume), failed


def download_benchmarks():
    series = {}
    for ticker in BENCHMARKS:
        df = yf.download(ticker, start=START, auto_adjust=True, progress=False)
        series[ticker] = df["Close"].iloc[:, 0] if hasattr(df["Close"], "columns") else df["Close"]
    return pd.DataFrame(series)


if __name__ == "__main__":
    uni = pd.read_csv(UNIVERSE_PATH)
    tickers = uni["YF_Ticker"].tolist()

    close_df, vol_df, failed = download_universe_prices(tickers)
    close_df.to_pickle("data/prices_close.pkl")
    vol_df.to_pickle("data/prices_volume.pkl")
    print(f"Universe prices: {close_df.shape[1]}/{len(tickers)} tickers succeeded")
    if failed:
        print("Failed tickers:", failed)

    bench_df = download_benchmarks()
    bench_df.to_pickle("data/benchmark.pkl")
    print(f"Benchmark data: {list(BENCHMARKS.values())} saved, "
          f"{bench_df.index.min().date()} to {bench_df.index.max().date()}")
