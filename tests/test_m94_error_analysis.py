from __future__ import annotations

import csv
from pathlib import Path
import pytest

from business_entity_resolution.evaluation.error_analysis import (
    classify_false_negative,
    analyze_strong_evidence_scores,
    extract_fns_from_predictions,
)


def test_classify_false_negative_rules():
    # Blocking miss
    assert classify_false_negative(0.0, 0, 0, 0.0, 0.0, 0.0, 0, is_blocked=False) == "1_BLOCKING_MISS"

    # Country conflict
    assert classify_false_negative(0.9, 1, 1, 1.0, 1.0, 1.0, 1) == "6_COUNTRY_CONFLICT"

    # Strong evidence below threshold
    assert classify_false_negative(0.85, 0, 1, 0.95, 0.9, 0.5, 1) == "2_STRONG_EVIDENCE_BELOW_THRESHOLD"

    # Address conflict
    assert classify_false_negative(0.70, 0, 0, 0.95, 0.95, 0.0, 1) == "5_ADDRESS_CONFLICT"

    # Weak evidence
    assert classify_false_negative(0.10, 0, 0, 0.40, 0.20, 0.10, 1) == "3_WEAK_EVIDENCE"


def test_analyze_strong_evidence_scores_simulation(tmp_path: Path):
    dummy_pred = tmp_path / "dummy_pred.tsv"
    rows = [
        {"source1_id": "S1-1", "target_source": "S2", "target_id": "S2-1", "calibrated_score": "0.99", "label": "1", "strong_evidence_count": "1"},
        {"source1_id": "S1-2", "target_source": "S2", "target_id": "S2-2", "calibrated_score": "0.85", "label": "1", "strong_evidence_count": "1"},
        {"source1_id": "S1-3", "target_source": "S2", "target_id": "S2-3", "calibrated_score": "0.75", "label": "1", "strong_evidence_count": "1"},
        {"source1_id": "S1-4", "target_source": "S2", "target_id": "S2-4", "calibrated_score": "0.10", "label": "0", "strong_evidence_count": "1"},
        {"source1_id": "S1-5", "target_source": "S2", "target_id": "S2-5", "calibrated_score": "0.40", "label": "0", "strong_evidence_count": "1"},
    ]
    with dummy_pred.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    res = analyze_strong_evidence_scores(dummy_pred, threshold=0.875)
    assert res["true_positives"]["count"] == 3
    assert res["false_positives"]["count"] == 2
    assert res["false_positives"]["max"] == 0.40

    # Test simulation at 0.70 threshold:
    # Recovers S1-2 (0.85) and S1-3 (0.75), intro FP = 0 (max FP is 0.40) -> safe!
    sim_70 = next(s for s in res["rescue_simulations"] if s["rescue_threshold"] == 0.70)
    assert sim_70["additional_tp_recovered"] == 2
    assert sim_70["introduced_fp"] == 0
    assert sim_70["safe"] is True
