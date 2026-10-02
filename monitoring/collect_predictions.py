"""
collect_predictions.py — Backfill actual prices into stored predictions.

For each prediction row where actual_value is NULL, this script checks if
the prediction date has passed, fetches the real closing price from yfinance,
and updates error / absolute_error fields.

Run this daily after market close to maintain a complete evaluation dataset.

Usage:
    python monitoring/collect_predictions.py
    python monitoring/collect_predictions.py --market US
    python monitoring/collect_predictions.py --dry-run

Status:
    PREPARED FOR FUTURE IMPLEMENTATION — the schema, logic, and yfinance
    integration are complete. Schedule this script daily after market close
    using the monitoring.yml workflow or a cron job.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "predictions.db"


def fetch_actual_price(symbol: str, date_str: str) -> float | None:
    """Fetch the closing price for `symbol` on `date_str` (YYYY-MM-DD)."""
    try:
        from datetime import timedelta

        import yfinance as yf
        date = datetime.strptime(date_str, "%Y-%m-%d")
        end  = date + timedelta(days=2)
        hist = yf.Ticker(symbol).history(start=date.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception as e:
        print(f"  ⚠ Could not fetch actual for {symbol} on {date_str}: {e}", file=sys.stderr)
        return None


def backfill(market: str | None = None, dry_run: bool = False) -> dict:
    if not DB_PATH.exists():
        print("No predictions database found.", file=sys.stderr)
        return {"updated": 0, "skipped": 0}

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    query = "SELECT * FROM predictions WHERE actual_value IS NULL"
    if market:
        query += f" AND market = '{market.upper()}'"

    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} prediction(s) awaiting actual values.")

    now = datetime.now(timezone.utc)
    updated = skipped = 0

    for row in rows:
        pred_ts     = datetime.fromisoformat(row["timestamp"])
        pred_date   = pred_ts.strftime("%Y-%m-%d")
        horizon     = row["horizon"]
        symbol      = row["symbol"]
        pred_id     = row["prediction_id"]
        pred_value  = row["predicted_value"]

        # Only backfill if the target date is in the past
        days_elapsed = (now - pred_ts).days
        if days_elapsed < horizon:
            skipped += 1
            continue

        actual = fetch_actual_price(symbol, pred_date)
        if actual is None:
            skipped += 1
            continue

        error    = round(actual - pred_value, 4)
        abs_err  = round(abs(error), 4)

        if dry_run:
            print(f"  [dry-run] {symbol} {pred_date}: predicted={pred_value:.2f} actual={actual:.2f} err={error:.4f}")
        else:
            conn.execute(
                "UPDATE predictions SET actual_value=?, error=?, absolute_error=? WHERE prediction_id=?",
                (actual, error, abs_err, pred_id),
            )
            updated += 1
            print(f"  ✓ {symbol} {pred_date}: predicted={pred_value:.2f} actual={actual:.2f} err={error:.4f}")

    if not dry_run:
        conn.commit()
    conn.close()

    return {"updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill actual prices into prediction database")
    parser.add_argument("--market",  type=str, default=None, help="Filter by market: US or IN")
    parser.add_argument("--dry-run", action="store_true",    help="Print changes without writing to DB")
    args = parser.parse_args()

    result = backfill(market=args.market, dry_run=args.dry_run)
    print(f"\nDone. Updated: {result['updated']}, Skipped: {result['skipped']}")


if __name__ == "__main__":
    main()
