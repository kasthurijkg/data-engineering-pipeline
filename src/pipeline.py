"""Data quality, anomaly detection, and explainable root-cause candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED = ["order_id", "customer_id", "order_amount", "order_status", "event_timestamp"]
ALLOWED_STATUSES = {"PLACED", "SHIPPED", "DELIVERED", "CANCELLED"}


def generate_demo_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic order events with reproducible data-quality defects."""
    if n < 30:
        raise ValueError("n must be at least 30")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "order_id": [f"ORD-{i:05d}" for i in range(n)],
        "customer_id": [f"CUST-{v:04d}" for v in rng.integers(1, 120, n)],
        "order_amount": rng.lognormal(mean=4.5, sigma=0.45, size=n).round(2),
        "order_status": rng.choice(["PLACED", "SHIPPED", "DELIVERED"], n),
        "event_timestamp": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
        "source_system": rng.choice(["IMS", "IFS"], n, p=[0.7, 0.3]),
    })
    frame.loc[4, "customer_id"] = None
    frame.loc[7, "order_amount"] = -99.0
    frame.loc[11, "order_status"] = "UNKNOWN"
    frame.loc[14, "order_id"] = frame.loc[13, "order_id"]
    frame.loc[18, "event_timestamp"] = "bad-timestamp"
    frame.loc[24, "order_amount"] = 99999.0
    return frame


def validate_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return row-level rule failures and an aggregate metrics report."""
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    if df.empty:
        raise ValueError("Input data has no rows")

    checks = pd.DataFrame(index=df.index)
    for col in REQUIRED:
        checks[f"null_{col}"] = df[col].isna() | df[col].astype("string").str.strip().eq("").fillna(False)

    amounts = pd.to_numeric(df["order_amount"], errors="coerce")
    checks["invalid_amount"] = amounts.isna() | (amounts <= 0)
    checks["invalid_status"] = ~df["order_status"].isin(ALLOWED_STATUSES)
    checks["invalid_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True).isna()
    checks["duplicate_order_id"] = df["order_id"].duplicated(keep=False) & df["order_id"].notna()
    checks = checks.fillna(False).astype(bool)
    checks["failed_checks"] = checks.sum(axis=1)
    checks["quality_pass"] = checks["failed_checks"].eq(0)
    result = df.copy()
    for col in checks:
        result[col] = checks[col].to_numpy()
    metrics = {
        "total_rows": int(len(df)),
        "passing_rows": int(checks["quality_pass"].sum()),
        "failing_rows": int((~checks["quality_pass"]).sum()),
        "quality_score_pct": round(100 * float(checks["quality_pass"].mean()), 2),
        "rule_failures": {col: int(checks[col].sum()) for col in checks if col not in ("quality_pass", "failed_checks")},
    }
    return result, metrics


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """Use Isolation Forest on valid amounts plus a frequency feature."""
    if not (0 < contamination <= 0.5):
        raise ValueError("contamination must be between 0 and 0.5")
    out = df.copy()
    amounts = pd.to_numeric(out["order_amount"], errors="coerce")
    out["anomaly_flag"] = False
    out["anomaly_score"] = np.nan
    valid = amounts.notna() & (amounts > 0)
    if int(valid.sum()) < 10:
        return out
    frequency = out["customer_id"].fillna("<missing>").astype(str).map(
        out["customer_id"].fillna("<missing>").astype(str).value_counts()
    )
    features = pd.DataFrame({
        "log_amount": np.log1p(amounts.loc[valid]),
        "customer_frequency": frequency.loc[valid].astype(float),
    }, index=out.index[valid])
    model = IsolationForest(random_state=42, contamination=contamination, n_estimators=150)
    predictions = model.fit_predict(features)
    out.loc[valid, "anomaly_flag"] = predictions == -1
    out.loc[valid, "anomaly_score"] = model.decision_function(features)
    return out


def root_cause_candidates(df: pd.DataFrame) -> list[dict]:
    """Rank potential causes using failing rules and correlated source systems.

    These are explainable hypotheses, not proven causal diagnoses.
    """
    failures = df[df["failed_checks"] > 0]
    if failures.empty:
        return []
    explanations = []
    checks = [c for c in df.columns if c.startswith("null_") or c in
              ("invalid_amount", "invalid_status", "invalid_timestamp", "duplicate_order_id")]
    for rule in checks:
        affected = df[df[rule]]
        if affected.empty:
            continue
        candidate = {
            "rule": rule,
            "affected_rows": int(len(affected)),
            "possible_cause": {
                "invalid_amount": "Amount transformation, currency mapping, or invalid source values",
                "invalid_status": "Unmapped source status or schema/value drift",
                "invalid_timestamp": "Incorrect timestamp format or ingestion parsing",
                "duplicate_order_id": "Replay, non-idempotent ingestion, or incorrect business key",
            }.get(rule, "Missing upstream field, join miss, or schema mismatch"),
        }
        if "source_system" in df.columns:
            counts = affected["source_system"].fillna("<missing>").value_counts()
            candidate["most_affected_source"] = str(counts.index[0])
            candidate["source_failures"] = int(counts.iloc[0])
        explanations.append(candidate)
    return sorted(explanations, key=lambda item: (-item["affected_rows"], item["rule"]))


def run_pipeline(input_path: Path | None, output_dir: Path, rows: int = 300) -> dict:
    data = pd.read_csv(input_path) if input_path else generate_demo_data(rows)
    checked, quality = validate_quality(data)
    detected = detect_anomalies(checked)
    candidates = root_cause_candidates(detected)
    report = {
        "quality": quality,
        "ml_anomalies": int(detected["anomaly_flag"].sum()),
        "root_cause_candidates": candidates,
        "note": "Root causes are ranked hypotheses; Isolation Forest anomalies require review.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    detected.to_csv(output_dir / "scored_orders.csv", index=False)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise data quality monitoring MVP")
    parser.add_argument("--input", type=Path, default=None, help="CSV input; omit for synthetic data")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--rows", type=int, default=300)
    args = parser.parse_args()
    report = run_pipeline(args.input, args.output, args.rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
