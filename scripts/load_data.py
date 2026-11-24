"""Read the five ARFF files into one CSV and report basic data quality.

Usage: python scripts/load_data.py data/raw data/companies.csv

Each file holds financial ratios from one year and a label saying whether the company
went bankrupt some years later: 1year.arff -> 5 years ahead, ..., 5year.arff -> 1 year ahead.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = {"1year": 5, "2year": 4, "3year": 3, "4year": 2, "5year": 1}


def read_arff(path):
    names, rows, in_data = [], [], False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@attribute"):
                names.append(line.split()[1])
            elif low.startswith("@data"):
                in_data = True
            elif in_data:
                rows.append(line.split(","))
    df = pd.DataFrame(rows, columns=names).replace("?", np.nan)
    feats = [c for c in names if c != "class"]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df["class"] = df["class"].astype(int)
    return df.rename(columns={c: "X" + c[4:] for c in feats}).rename(columns={"class": "bankrupt"})


def main(raw_dir="data/raw", out="data/companies.csv"):
    frames = []
    for stem, h in HORIZON.items():
        d = read_arff(Path(raw_dir) / f"{stem}.arff")
        d.insert(0, "years_ahead", h)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if c.startswith("X")]

    print("rows by horizon:\n", df.groupby("years_ahead").bankrupt.agg(["size", "sum", "mean"]))
    dup = df.duplicated(subset=["years_ahead"] + feats + ["bankrupt"])
    conflict = df.duplicated(subset=["years_ahead"] + feats, keep=False) & ~df.duplicated(
        subset=["years_ahead"] + feats + ["bankrupt"], keep=False)
    print(f"exact duplicate rows within a horizon: {dup.sum()}")
    print(f"same figures but different label: {conflict.sum()}")
    miss = df[feats].isna().mean().sort_values(ascending=False)
    print("most missing columns:\n", miss.head(5).round(3))
    print(f"rows with at least one missing value: {df[feats].isna().any(axis=1).mean():.1%}")
    inf = np.isinf(df[feats]).sum().sum()
    print(f"infinite values: {inf}")

    df = df[~dup].reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
