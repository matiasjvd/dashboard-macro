#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

"""
Manual refresh: replaces local data/macro_data.db (or .parquet) with the file
from the source path. Safe for Streamlit Cloud workflow since it's an explicit action.
"""

def main():
    p = argparse.ArgumentParser(description="Replace local data file with source file")
    p.add_argument("--source", required=True, help="Path to source file (.db or .parquet)")
    p.add_argument("--target", choices=["sqlite", "parquet"], default="sqlite", help="Target kind in ./data")
    args = p.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Source file not found: {src}")

    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "sqlite":
        dst = data_dir / "macro_data.db"
    else:
        dst = data_dir / "macro_data.parquet"

    # Remove old file if exists
    if dst.exists():
        dst.unlink()

    shutil.copy2(src, dst)
    print(f"Replaced {dst.name} with {src}")


if __name__ == "__main__":
    main()

3