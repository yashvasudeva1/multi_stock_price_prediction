"""
drift_report.py — Lightweight prediction distribution drift monitor.

Compares the distribution of recent predictions against a reference period
without adding heavy external dependencies.

Metrics:
    - Population Stability Index (PSI)
    - KS statistic (scipy if available, else manual)
    - Mean shift
    - Std shift
    - Prediction quantiles

Usage:
    python monitoring/drift_report.py
    python monitoring/drift_report.py --reference-days 30 --recent-days 7
    python monitoring/drift_report.py --report monitoring/drift_report.json

Status:
    PREPARED FOR FUTURE IMPLEMENTATION — requires sufficient prediction volume
    in monitoring/predictions.db before drift metrics are meaningful.
    The PSI and KS implementations are ready to use once data is available.
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

# PSI thresholds (industry standard)
PSI_STABLE     = 0.10   # < 0.10 → no significant shift
PSI_MONITORING = 0.20   # 0.10–0.20 → monitor closely
# > 0.20 → significant shift, investigate


def _load_predictions_window(days: int, market: str | None = None) -> np.ndarray:
    if not DB_PATH.exists():
        return np.array([])
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn   = sqlite3.connect(str(DB_PATH))
    query  = "SELECT predicted_value FROM predictions WHERE timestamp >= ?"
    params = [cutoff]
    if market:
        query += " AND market = ?"
        params.append(market.strip().upper())
    query += " ORDER BY timestamp"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return np.array([r[0] for r in rows], dtype=float)


def _psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index."""
    min_val = min(reference.min(), production.min())
    max_val = max(reference.max(), production.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts  = np.histogram(reference,  bins=bin_edges)[0]
    prod_counts = np.histogram(production, bins=bin_edges)[0]

    # Avoid zeros
    ref_pct  = np.maximum(ref_counts  / len(reference),  1e-6)
    prod_pct = np.maximum(prod_counts / len(production), 1e-6)

    psi = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))
    return round(psi, 4)


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """KS statistic (two-sample, manual implementation)."""
    combined = np.sort(np.concatenate([a, b]))
    cdf_a    = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cdf_b    = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    return round(float(np.max(np.abs(cdf_a - cdf_b))), 4)


def _quantiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {}
    return {
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def generate_report(
    reference_days: int = 30,
    recent_days: int = 7,
    market: str | None = None,
) -> dict:
    reference = _load_predictions_window(reference_days, market=market)
    recent    = _load_predictions_window(recent_days, market=market)

    report: dict = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "market":         market or "ALL",
        "reference_days": reference_days,
        "recent_days":    recent_days,
        "reference_n":    len(reference),
        "recent_n":       len(recent),
    }

    if len(reference) < 10 or len(recent) < 5:
        report["status"]  = "insufficient_data"
        report["message"] = (
            f"Need ≥10 reference predictions (got {len(reference)}) and "
            f"≥5 recent predictions (got {len(recent)}) for drift analysis."
        )
        return report

    psi = _psi(reference, recent)
    ks  = _ks_statistic(reference, recent)

    if psi < PSI_STABLE:
        drift_status = "stable"
    elif psi < PSI_MONITORING:
        drift_status = "monitor"
    else:
        drift_status = "drift_detected"

    report.update({
        "status":       drift_status,
        "psi":          psi,
        "ks_statistic": ks,
        "reference_stats": {
            "mean":      round(float(reference.mean()), 4),
            "std":       round(float(reference.std()),  4),
            "quantiles": _quantiles(reference),
        },
        "recent_stats": {
            "mean":      round(float(recent.mean()), 4),
            "std":       round(float(recent.std()),  4),
            "quantiles": _quantiles(recent),
        },
        "mean_shift": round(float(recent.mean() - reference.mean()), 4),
        "std_shift":  round(float(recent.std()  - reference.std()),  4),
    })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM prediction drift report")
    parser.add_argument("--market",         type=str, default=None, help="Filter by market (US or IN)")
    parser.add_argument("--reference-days", type=int, default=30)
    parser.add_argument("--recent-days",    type=int, default=7)
    parser.add_argument("--report",         type=str, default=None)
    args = parser.parse_args()

    report = generate_report(args.reference_days, args.recent_days, market=args.market)
    print(json.dumps(report, indent=2))

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to {path}")

    # Non-zero exit on drift so CI can optionally alert
    if report.get("status") == "drift_detected":
        print("\n⚠  Drift detected — review predictions.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
