"""
evaluate_model.py — Offline model evaluation against the results/ CSVs.

Reads pre-computed metric CSV files from results/us/ and results/india/
and prints a summary table.  Does not require live data or model inference.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --market US
    python scripts/evaluate_model.py --market IN
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def print_summary(market: str) -> None:
    subdir = "us" if market == "US" else "india"
    summary_path = RESULTS_DIR / subdir / "metrics_overall_summary.csv"
    rows = load_csv(summary_path)
    if not rows:
        return

    print(f"\n{'=' * 60}")
    print(f"  {market} Model — Overall Summary")
    print(f"{'=' * 60}")
    headers = list(rows[0].keys())
    print(f"  {' | '.join(headers)}")
    print(f"  {'-' * (len(headers) * 12)}")
    for row in rows:
        print("  " + " | ".join(row.get(h, "–") for h in headers))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation from results/ CSVs")
    parser.add_argument("--market", choices=["US", "IN", "both"], default="both")
    args = parser.parse_args()

    markets = ["US", "IN"] if args.market == "both" else [args.market]
    for m in markets:
        print_summary(m)
    print()


if __name__ == "__main__":
    main()
