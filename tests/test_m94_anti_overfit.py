from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.validate_m94_anti_overfit import validate_anti_overfit


def test_anti_overfit_passes_on_b5_evaluation(tmp_path: Path):
    source_eval = Path("artifacts/diagnostics/m92d_b5_evaluation.json")
    if not source_eval.exists():
        pytest.skip("m92d_b5_evaluation.json does not exist yet")

    # Copy to tmp_path to test without side-effects
    temp_json = tmp_path / "test_eval.json"
    temp_json.write_text(source_eval.read_text(encoding="utf-8"), encoding="utf-8")

    res = validate_anti_overfit(temp_json, tolerance_pp=0.50, update_source=True)

    assert res["overall_status"] == "PASS"
    assert res["generalization_gap_pp"] <= 0.50
    assert len(res["checks"]) == 5
    assert all(c["pass"] for c in res["checks"])

    # Verify the JSON file was updated
    updated = json.loads(temp_json.read_text(encoding="utf-8"))
    assert updated["anti_overfit_status"]["status"] == "PASS"


def test_anti_overfit_fails_when_gap_exceeds_tolerance(tmp_path: Path):
    dummy_json = tmp_path / "dummy_eval.json"
    dummy_data = {
        "oof": {
            "selected_threshold": 0.875,
            "selected_metrics": {"macro_f05": 0.98},
        },
        "holdout": {
            "selected_threshold_from_oof": 0.875,
            "metrics": {"macro_f05": 0.96},  # Gap = 2.0 pp
        },
        "data_contract": {
            "test_data_used": False,
            "test_labels_used": False,
        },
        "partitioning": {
            "no_s1_cross_partition": True,
            "holdout_fraction": 0.15,
        },
        "config": {
            "model_type": "logistic",
            "calibration_method": "platt",
        },
        "anti_overfit_status": {"status": "INVESTIGATE"},
    }
    dummy_json.write_text(json.dumps(dummy_data), encoding="utf-8")

    res = validate_anti_overfit(dummy_json, tolerance_pp=0.50, update_source=True)
    assert res["overall_status"] == "INVESTIGATE"
    assert res["generalization_gap_pp"] == pytest.approx(2.0, 0.01)

    # Status should not be changed to PASS
    updated = json.loads(dummy_json.read_text(encoding="utf-8"))
    assert updated["anti_overfit_status"]["status"] == "INVESTIGATE"
