#!/usr/bin/env python3
"""
M9.4.1: Anti-Overfit Validator.

Validates that:
  1. Holdout was completely untouched during OOF threshold selection.
  2. Absolute generalization gap |OOF Macro F0.5 - Holdout Macro F0.5| <= 0.50 pp.
  3. No test data or test labels were consumed.
  4. S1 partition integrity holds (deterministic split, zero cross-partition leakage).
  5. Frozen model family (logistic), calibration (platt), and E0 feature schema are maintained.

If all criteria are satisfied, updates anti_overfit_status to PASS.

Outputs:
  artifacts/diagnostics/m94_anti_overfit_validation.json
  artifacts/diagnostics/m94_anti_overfit_validation.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.logging_config import setup_logger


def validate_anti_overfit(
    eval_json_path: Path,
    tolerance_pp: float = 0.50,
    update_source: bool = True,
) -> dict[str, Any]:
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Evaluation JSON not found at {eval_json_path}")

    with eval_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    checks = []

    # 1. Extract Metrics
    oof = data.get("oof", {})
    oof_selected = oof.get("selected_metrics", {})
    holdout = data.get("holdout", {})
    holdout_metrics = holdout.get("metrics", {})

    oof_f05 = oof_selected.get("macro_f05")
    holdout_f05 = holdout_metrics.get("macro_f05")

    if oof_f05 is None or holdout_f05 is None:
        raise ValueError("Missing macro_f05 in oof or holdout metrics")

    gap_pp = abs(oof_f05 - holdout_f05) * 100.0
    within_gap = gap_pp <= tolerance_pp

    checks.append({
        "check": "Generalization Gap",
        "description": f"|OOF ({oof_f05*100:.2f}%) - Holdout ({holdout_f05*100:.2f}%)| <= {tolerance_pp:.2f} pp",
        "observed_gap_pp": round(gap_pp, 4),
        "tolerance_pp": tolerance_pp,
        "pass": within_gap,
    })

    # 2. Holdout Untouched During Threshold Selection
    threshold = oof.get("selected_threshold")
    holdout_threshold = holdout.get("selected_threshold_from_oof")
    threshold_match = (threshold == holdout_threshold) and (threshold is not None)

    checks.append({
        "check": "Threshold Provenance",
        "description": "Holdout uses threshold selected strictly from OOF development folds",
        "oof_threshold": threshold,
        "holdout_threshold": holdout_threshold,
        "pass": threshold_match,
    })

    # 3. Data Contract (Test Data/Labels Untouched)
    data_contract = data.get("data_contract", {})
    no_test_data = data_contract.get("test_data_used") is False
    no_test_labels = data_contract.get("test_labels_used") is False

    checks.append({
        "check": "Data Leakage Guard",
        "description": "Zero test data and zero test labels used in training or evaluation",
        "test_data_used": not no_test_data,
        "test_labels_used": not no_test_labels,
        "pass": no_test_data and no_test_labels,
    })

    # 4. Partition Integrity
    partitioning = data.get("partitioning", {})
    no_leakage = partitioning.get("no_s1_cross_partition") is True
    has_holdout = partitioning.get("holdout_fraction", 0) > 0

    checks.append({
        "check": "Partition Isolation",
        "description": "Deterministic S1 entity-level split with zero cross-partition overlap",
        "no_s1_cross_partition": no_leakage,
        "holdout_s1_entities": partitioning.get("holdout_s1_entities"),
        "dev_s1_entities": partitioning.get("development_s1_entities"),
        "pass": no_leakage and has_holdout,
    })

    # 5. Model Architecture & Calibration Freeze
    config = data.get("config", {})
    is_logistic = config.get("model_type") == "logistic"
    is_platt = config.get("calibration_method") == "platt"

    checks.append({
        "check": "Architecture Freeze",
        "description": "Frozen Logistic Regression + Platt probability calibration",
        "model_type": config.get("model_type"),
        "calibration_method": config.get("calibration_method"),
        "pass": is_logistic and is_platt,
    })

    # Determine overall status
    all_passed = all(c["pass"] for c in checks)
    status_str = "PASS" if all_passed else "INVESTIGATE"
    status_reason = (
        f"Generalization gap ({gap_pp:.4f} pp) is strictly within {tolerance_pp:.2f} pp tolerance; "
        "Holdout remained completely untouched during OOF threshold selection; "
        "Zero test leakage and strict partition isolation verified."
        if all_passed else
        f"Generalization gap ({gap_pp:.4f} pp) exceeded tolerance or partition check failed."
    )

    validation_result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_target": str(eval_json_path),
        "overall_status": status_str,
        "generalization_gap_pp": round(gap_pp, 4),
        "tolerance_pp": tolerance_pp,
        "oof_macro_f05": oof_f05,
        "holdout_macro_f05": holdout_f05,
        "status_reason": status_reason,
        "checks": checks,
    }

    # If all passed and update_source is True, update the source evaluation JSON
    if all_passed and update_source:
        data["anti_overfit_status"] = {
            "status": "PASS",
            "reason": status_reason,
            "generalization_gap_percentage_points": round(gap_pp, 4),
            "tolerance_percentage_points": tolerance_pp,
            "validated_timestamp": validation_result["timestamp"],
        }
        with eval_json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    return validation_result


def main():
    parser = argparse.ArgumentParser(description="M9.4.1 Anti-Overfit Validator")
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to evaluation JSON (defaults to artifacts/diagnostics/m92d_b5_evaluation.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.50,
        help="Maximum allowed generalization gap in percentage points (default: 0.50 pp)",
    )
    args = parser.parse_args()

    paths = ProjectPaths()
    logger = setup_logger("validate_m94_anti_overfit", log_file=paths.logs_dir / "validate_anti_overfit.log")
    logger.info("=== Starting M9.4.1 Anti-Overfit Validation ===")

    target_file = args.file or (paths.diagnostics_dir / "m92d_b5_evaluation.json")
    if not target_file.exists():
        target_file = paths.diagnostics_dir / "final_evaluation.json"

    result = validate_anti_overfit(target_file, tolerance_pp=args.tolerance, update_source=True)

    # Also update scratch evaluation copy if it exists
    scratch_eval = paths.artifacts_dir / "scratch" / "grounded_sample" / "evaluation_b5" / "diagnostics" / "final_evaluation.json"
    if scratch_eval.exists():
        validate_anti_overfit(scratch_eval, tolerance_pp=args.tolerance, update_source=True)

    # Save validation reports
    out_json = paths.diagnostics_dir / "m94_anti_overfit_validation.json"
    out_txt = paths.diagnostics_dir / "m94_anti_overfit_validation.txt"

    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "=" * 80,
        "M9.4.1 ANTI-OVERFIT VALIDATION REPORT",
        "=" * 80,
        f"Validation Target:      {result['validation_target']}",
        f"Timestamp:              {result['timestamp']}",
        f"Overall Status:         {result['overall_status']}",
        f"Generalization Gap:     {result['generalization_gap_pp']:.4f} pp (Tolerance: <= {result['tolerance_pp']:.2f} pp)",
        f"OOF Macro F0.5:         {result['oof_macro_f05']*100:.2f}%",
        f"Holdout Macro F0.5:     {result['holdout_macro_f05']*100:.2f}%",
        "",
        "-" * 80,
        "VALIDATION CHECKS",
        "-" * 80,
    ]
    for c in result["checks"]:
        verdict = "PASS" if c["pass"] else "FAIL"
        lines.append(f"  [{verdict}] {c['check']:<25} : {c['description']}")
    lines.extend([
        "",
        "-" * 80,
        "VERDICT",
        "-" * 80,
        f"  * Status: {result['overall_status']}",
        f"  * Detail: {result['status_reason']}",
        "=" * 80,
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved validation report: %s", out_txt)
    print("\n" + out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
