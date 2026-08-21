"""
Build the eligible stock universe: Nifty 100 + Nifty Midcap 100 + Nifty Smallcap 100.

Reads the three raw NSE index constituent lists in data/raw_index_lists/ and writes a single
deduplicated universe.csv with a Yahoo Finance ticker column for the data-download step.

Usage:
    python src/build_universe.py
"""
import pandas as pd

RAW_DIR = "data/raw_index_lists"
OUT_PATH = "data/universe.csv"

FILES = {
    "Nifty100": f"{RAW_DIR}/nifty100.csv",
    "NiftyMidcap100": f"{RAW_DIR}/niftymidcap100.csv",
    "NiftySmallcap100": f"{RAW_DIR}/niftysmallcap100.csv",
}


def build_universe():
    frames = []
    for index_name, path in FILES.items():
        df = pd.read_csv(path)
        df["Index"] = index_name
        frames.append(df)

    uni = pd.concat(frames, ignore_index=True)
    uni["Symbol"] = uni["Symbol"].str.strip()
    uni["YF_Ticker"] = uni["Symbol"] + ".NS"

    # A handful of symbols can appear in more than one index list; keep one row per symbol
    # and record every index it belongs to instead of duplicating it.
    memberships = uni.groupby("Symbol")["Index"].apply(lambda s: ",".join(sorted(set(s)))).rename("Indices")
    uni = uni.drop_duplicates(subset="Symbol").drop(columns="Index").merge(memberships, on="Symbol")
    uni = uni[["Symbol", "YF_Ticker", "Company Name", "Industry", "Indices"]]
    return uni


if __name__ == "__main__":
    uni = build_universe()
    uni.to_csv(OUT_PATH, index=False)
    print(f"Universe: {uni.shape[0]} stocks written to {OUT_PATH}")
    print(uni["Indices"].value_counts())
