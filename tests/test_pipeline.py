import pandas as pd
import pytest

from src.pipeline import detect_anomalies, generate_demo_data, root_cause_candidates, validate_quality


def test_demo_is_reproducible():
    pd.testing.assert_frame_equal(generate_demo_data(), generate_demo_data())


def test_quality_rules_find_injected_defects():
    scored, metrics = validate_quality(generate_demo_data())
    assert metrics["failing_rows"] >= 5
    assert metrics["rule_failures"]["duplicate_order_id"] == 2
    assert metrics["rule_failures"]["invalid_timestamp"] >= 1
    assert scored["quality_pass"].dtype == bool


def test_missing_required_column_rejected():
    with pytest.raises(ValueError, match="Required columns missing"):
        validate_quality(pd.DataFrame({"order_id": ["a"]}))


def test_anomaly_detection_returns_scores():
    quality, _ = validate_quality(generate_demo_data())
    result = detect_anomalies(quality)
    assert result["anomaly_flag"].any()
    assert result["anomaly_score"].notna().any()


def test_root_cause_is_rule_grounded():
    quality, _ = validate_quality(generate_demo_data())
    candidates = root_cause_candidates(quality)
    assert any(c["rule"] == "invalid_status" for c in candidates)
    assert all(c["affected_rows"] > 0 for c in candidates)
