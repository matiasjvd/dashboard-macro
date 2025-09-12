#!/usr/bin/env python3
import argparse
import os
import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

# This script copies a subset of data from a source SQLite DB (macro_data table)
# into this repo, in one of three formats: sqlite (default), parquet, csv.
# Manual execution only; suitable for Streamlit Cloud workflows.


def read_subset_from_sqlite(src_db: Path, countries: List[str] | None, metrics: List[str] | None,
                            start: str | None, end: str | None) -> pd.DataFrame:
    conn = sqlite3.connect(str(src_db))
    try:
        clauses = []
        params: list = []
        if countries:
            qmarks = ",".join(["?"] * len(countries))
            clauses.append(f"country IN ({qmarks})")
            params.extend(countries)
        if metrics:
            qmarks = ",".join(["?"] * len(metrics))
            clauses.append(f"metric IN ({qmarks})")
            params.extend(metrics)
        if start:
            clauses.append("DATE >= ?")
            params.append(start)
        if end:
            clauses.append("DATE <= ?")
            params.append(end)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        q = f"SELECT country, DATE, metric, value FROM macro_data{where} ORDER BY country, DATE"
        df = pd.read_sql(q, conn, params=params)
        # Normalize types
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["DATE", "value"]).drop_duplicates()
        return df
    finally:
        conn.close()


def write_sqlite(df: pd.DataFrame, out_db: Path):
    out_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_db))
    try:
        # Create table if needed
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_data (
              country TEXT,
              DATE TEXT,
              metric TEXT,
              value REAL
            )
            """
        )
        conn.commit()
        # Replace strategy: remove overlapping rows then append
        if not df.empty:
            countries = df["country"].dropna().unique().tolist()
            metrics = df["metric"].dropna().unique().tolist()
            if countries and metrics:
                # Delete overlapping scope
                c_q = ",".join(["?"] * len(countries))
                m_q = ",".join(["?"] * len(metrics))
                conn.execute(f"DELETE FROM macro_data WHERE country IN ({c_q}) AND metric IN ({m_q})", countries + metrics)
                conn.commit()
        df2 = df.copy()
        df2["DATE"] = df2["DATE"].dt.strftime("%Y-%m-%d")
        df2.to_sql("macro_data", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()


def write_parquet(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def write_csv(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser(description="Extract subset of macro_data into this repo")
    p.add_argument("--source-db", required=True, help="Path to source SQLite DB (macro_data table)")
    p.add_argument("--countries", default="", help="Comma-separated country names")
    p.add_argument("--metrics", default="", help="Comma-separated metric names")
    p.add_argument("--start", default="", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="", help="End date YYYY-MM-DD")
    p.add_argument("--format", choices=["sqlite", "parquet", "csv"], default="sqlite")

    args = p.parse_args()

    src_db = Path(args.source_db).expanduser().resolve()
    if not src_db.exists():
        raise SystemExit(f"Source DB not found: {src_db}")

    countries = [c.strip() for c in args.countries.split(",") if c.strip()] or None
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    start = args.start or None
    end = args.end or None

    df = read_subset_from_sqlite(src_db, countries, metrics, start, end)

    if args.format == "sqlite":
        out_db = Path("data/macro_data.db").resolve()
        write_sqlite(df, out_db)
        print(f"Wrote {len(df)} rows to {out_db}")
    elif args.format == "parquet":
        out_path = Path("data/macro_data.parquet").resolve()
        write_parquet(df, out_path)
        print(f"Wrote {len(df)} rows to {out_path}")
    else:
        out_path = Path("data/macro_data.csv").resolve()
        write_csv(df, out_path)
        print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()