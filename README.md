# AI-Driven Data Quality Monitoring, Anomaly Detection, and Root-Cause Analysis

A reproducible Python MVP for validating enterprise order data, detecting unusual numeric behavior, and explaining likely upstream failure points.

## Architecture

CSV / synthetic IMS-IFS orders → schema checks → row-level data-quality rules → Isolation Forest anomaly detection → ranked root-cause hypotheses → CSV and JSON reports.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m src.pipeline --output outputs
pytest -q
```

For a real CSV matching the expected columns:

```bash
python -m src.pipeline --input data/orders.csv --output outputs
```

Required columns: `order_id`, `customer_id`, `order_amount`, `order_status`, `event_timestamp`. Optional: `source_system`.

### Output

- `outputs/scored_orders.csv`: original rows, rule flags, pass/fail, anomaly score, anomaly flag
- `outputs/report.json`: quality score, rule failure counts, anomaly count, root-cause hypotheses

Outputs are gitignored. Demo data is synthetic, not customer data.

## Scope & limitations

- Data quality: missing values, bad amounts, unsupported statuses, invalid timestamps, duplicate keys.
- ML: unsupervised Isolation Forest on log order amount and customer frequency. A flagged record is unusual, **not necessarily wrong**.
- Root-cause analysis: a transparent, rule-driven *hypothesis generator*; it does not establish causality.
- This is an initial local MVP, not an independently validated production monitoring platform. Tune rules, contamination, thresholds and privacy controls for each real dataset.

## Next milestones

1. Configurable YAML quality rules and a metrics history store.
2. Source-to-target reconciliation and schema drift tracking for BigQuery/Iceberg.
3. Statistical drift detection, temporal patterns and alert thresholds.
4. Root-cause evidence across DAG lineage (Airflow/Composer), plus review feedback.
5. Streamlit dashboard, scheduled workflows, unit/integration tests and deployment.
