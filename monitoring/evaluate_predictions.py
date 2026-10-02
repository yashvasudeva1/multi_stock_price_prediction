"""
evaluate_predictions.py — Offline model performance evaluation.

Reads stored predictions from monitoring/predictions.db and computes
performance metrics for rows that have actual_value populated.

Usage:
    python monitoring/evaluate_predictions.py
    python monitoring/evaluate_predictions.py --report monitoring/performance_report.json
    python monitoring/evaluate_predictions.py --window 7   # 7-day rolling
    python monitoring/evaluate_predictions.py --window 30  # 30-day rolling

Status:
    PREPARED FOR FUTURE IMPLEMENTATION — requires actual_value to be backfilled.
    The schema and metrics calculations are implemented; the backfill pipeline
    that joins predictions with real prices has not yet been built.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "predictions.db"


def load_predictions(window_days: int | None = None) -> list[dict]:
    """Load predictions from SQLite where actual_value is not NULL."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. No predictions to evaluate.", file=sys.stderr)
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    if window_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        rows = conn.execute(
            "SELECT * FROM predictions WHERE actual_value IS NOT NULL AND timestamp >= ?",
            (cutoff,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE actual_value IS NOT NULL"
        ).fetchall()

    conn.close()
    return [dict(r) for r in rows]


def compute_metrics(rows: list[dict]) -> dict:
    """Compute MAE, RMSE, MAPE, and directional accuracy from a list of prediction rows."""
    if not rows:
        return {"n": 0, "message": "No predictions with actual values found"}

    actual    = np.array([r["actual_value"]    for r in rows], dtype=float)
    predicted = np.array([r["predicted_value"] for r in rows], dtype=float)

    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100)

    if len(actual) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100)
    else:
        dir_acc = None

    return {
        "n":        len(rows),
        "mae":      round(mae, 4),
        "rmse":     round(rmse, 4),
        "mape":     round(mape, 4),
        "dir_acc":  round(dir_acc, 2) if dir_acc is not None else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PRISM model predictions")
    parser.add_argument("--window", type=int, default=None, help="Rolling window in days (e.g. 7, 30)")
    parser.add_argument("--report", type=str, default=None, help="Path to write JSON report")
    args = parser.parse_args()

    rows    = load_predictions(window_days=args.window)
    metrics = compute_metrics(rows)

    if args.window:
        metrics["window_days"] = args.window

    print(json.dumps(metrics, indent=2))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, indent=2))
        print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
