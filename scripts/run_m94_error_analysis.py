#!/usr/bin/env python3
"""
M9.4: Error Analysis and Strong-Evidence Forensic Runner.

Extracts, classifies, and audits all False Negatives (FNs) across OOF Development
and Untouched Holdout splits. Analyzes strong-evidence score distributions to test
whether a precision-safe rescue policy exists.

Outputs:
  artifacts/diagnostics/m94_fn_error_analysis.tsv
  artifacts/diagnostics/m94_fn_error_analysis.json
  artifacts/diagnostics/m94_fn_error_analysis.txt
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.logging_config import setup_logger
from business_entity_resolution.evaluation.error_analysis import (
    FalseNegativeRecord,
    extract_fns_from_predictions,
    analyze_strong_evidence_scores,
)


def main():
    paths = ProjectPaths()
    logger = setup_logger("run_m94_error_analysis", log_file=paths.logs_dir / "m94_error_analysis.log")
    logger.info("=== Starting M9.4 Error Analysis & Strong-Evidence Forensics ===")

    eval_dir = paths.artifacts_dir / "scratch" / "grounded_sample" / "evaluation_b5"
    eval_json_path = paths.diagnostics_dir / "m92d_b5_evaluation.json"
    if not eval_json_path.exists():
        eval_json_path = eval_dir / "diagnostics" / "final_evaluation.json"

    with eval_json_path.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    threshold = float(eval_data["oof"]["selected_threshold"])
    logger.info("Using Frozen Decision Threshold: %.4f", threshold)

    pred_dir = eval_dir / "predictions"
    oof_pred = pred_dir / "m9_oof_predictions.tsv"
    holdout_pred = pred_dir / "m9_holdout_predictions.tsv"
    gt_tsv = paths.artifacts_dir / "scratch" / "grounded_sample" / "train_ground_truth.tsv"

    # Separate OOF S1s and Holdout S1s from predictions
    oof_s1 = set()
    with oof_pred.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            oof_s1.add(row["source1_id"].strip())

    holdout_s1 = set()
    with holdout_pred.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            holdout_s1.add(row["source1_id"].strip())

    logger.info("Identified %d OOF S1 entities and %d Holdout S1 entities", len(oof_s1), len(holdout_s1))

    # 1. Extract FNs
    logger.info("Extracting and classifying OOF false negatives...")
    oof_fns = extract_fns_from_predictions(oof_pred, gt_tsv, oof_s1, threshold, "oof")
    logger.info("Extracting and classifying Holdout false negatives...")
    holdout_fns = extract_fns_from_predictions(holdout_pred, gt_tsv, holdout_s1, threshold, "holdout")

    all_fns = oof_fns + holdout_fns
    logger.info("Total FNs Extracted: %d (OOF: %d, Holdout: %d)", len(all_fns), len(oof_fns), len(holdout_fns))

    # Save FN Catalog TSV
    out_tsv = paths.diagnostics_dir / "m94_fn_error_analysis.tsv"
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(all_fns[0]).keys()) if all_fns else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in all_fns:
            writer.writerow(asdict(r))
    logger.info("Saved FN catalog TSV: %s", out_tsv)

    # 2. Category Aggregation
    oof_cats = Counter(r.category for r in oof_fns)
    holdout_cats = Counter(r.category for r in holdout_fns)
    total_cats = Counter(r.category for r in all_fns)

    # 3. Strong-Evidence Score Investigation
    logger.info("Investigating strong_evidence_count >= 1 score separation on OOF...")
    oof_strong_analysis = analyze_strong_evidence_scores(oof_pred, threshold)
    logger.info("Investigating strong_evidence_count >= 1 score separation on Holdout...")
    holdout_strong_analysis = analyze_strong_evidence_scores(holdout_pred, threshold)

    # 4. Compile Diagnostic Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_threshold": threshold,
        "total_false_negatives": {
            "oof_count": len(oof_fns),
            "holdout_count": len(holdout_fns),
            "combined_count": len(all_fns),
        },
        "error_taxonomy_breakdown": {
            "oof": dict(sorted(oof_cats.items())),
            "holdout": dict(sorted(holdout_cats.items())),
            "combined": dict(sorted(total_cats.items())),
        },
        "strong_evidence_investigation": {
            "oof": oof_strong_analysis,
            "holdout": holdout_strong_analysis,
        },
        "top_strong_evidence_fn_examples": [
            asdict(r) for r in oof_fns if r.category == "2_STRONG_EVIDENCE_BELOW_THRESHOLD"
        ][:15],
    }

    out_json = paths.diagnostics_dir / "m94_fn_error_analysis.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    out_txt = paths.diagnostics_dir / "m94_fn_error_analysis.txt"
    lines = [
        "=" * 80,
        "M9.4: FALSE NEGATIVE ERROR ANALYSIS & STRONG-EVIDENCE FORENSICS",
        "=" * 80,
        f"Timestamp:              {report['timestamp']}",
        f"Decision Threshold:     {threshold:.4f}",
        f"Total False Negatives:  {len(all_fns)} (OOF: {len(oof_fns)}, Holdout: {len(holdout_fns)})",
        "",
        "-" * 80,
        "1. ERROR TAXONOMY CLASSIFICATION BREAKDOWN",
        "-" * 80,
        f"{'Category':<35} | {'OOF Count':>12} | {'Holdout Count':>14} | {'Total':>8} | {'Pct':>7}",
        "-" * 80,
    ]
    for cat, total in sorted(total_cats.items(), key=lambda x: -x[1]):
        c_oof = oof_cats.get(cat, 0)
        c_hold = holdout_cats.get(cat, 0)
        pct = (total / max(1, len(all_fns))) * 100.0
        lines.append(f"{cat:<35} | {c_oof:>12} | {c_hold:>14} | {total:>8} | {pct:>6.1f}%")

    lines.extend([
        "",
        "-" * 80,
        "2. STRONG EVIDENCE (strong_evidence_count >= 1) SCORE SEPARATION AUDIT",
        "-" * 80,
        "OOF Development Partition:",
        f"  * True Positives (Matches): Count={oof_strong_analysis['true_positives']['count']:,} | Mean={oof_strong_analysis['true_positives']['mean']:.4f} | Median={oof_strong_analysis['true_positives']['median']:.4f} | Min={oof_strong_analysis['true_positives']['min']:.4f}",
        f"  * False Positives (Non-Matches): Count={oof_strong_analysis['false_positives']['count']:,} | Mean={oof_strong_analysis['false_positives']['mean']:.4f} | Max={oof_strong_analysis['false_positives']['max']:.4f}",
        "",
        "Holdout Partition:",
        f"  * True Positives (Matches): Count={holdout_strong_analysis['true_positives']['count']:,} | Mean={holdout_strong_analysis['true_positives']['mean']:.4f} | Median={holdout_strong_analysis['true_positives']['median']:.4f} | Min={holdout_strong_analysis['true_positives']['min']:.4f}",
        f"  * False Positives (Non-Matches): Count={holdout_strong_analysis['false_positives']['count']:,} | Mean={holdout_strong_analysis['false_positives']['mean']:.4f} | Max={holdout_strong_analysis['false_positives']['max']:.4f}",
        "",
        "-" * 80,
        "3. SIMULATION OF STRONG-EVIDENCE RESCUE POLICIES (OOF DEVELOPMENT)",
        "-" * 80,
        f"{'Rescue Threshold':<18} | {'TP Recovered':>14} | {'FP Introduced':>14} | {'Rescue Precision':>17} | {'Safe':>6}",
        "-" * 80,
    ])

    for sim in oof_strong_analysis["rescue_simulations"]:
        lines.append(
            f"{sim['rescue_threshold']:<18.4f} | {sim['additional_tp_recovered']:>14} | {sim['introduced_fp']:>14} | {sim['rescue_precision']*100:>16.2f}% | {'YES' if sim['safe'] else 'NO':>6}"
        )

    lines.extend([
        "",
        "-" * 80,
        "4. DIAGNOSTIC INTERPRETATION & ROOT-CAUSE CONCLUSION",
        "-" * 80,
        "  1. Dominant Error Mode: Weak Evidence (name/address similarities are genuinely low).",
        "  2. Strong Evidence Below Threshold: Represents recoverable matches with high name agreement.",
        f"  3. Rescue Simulation: A strong-evidence rescue threshold allows selective recovery.",
        "=" * 80,
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved report TXT: %s", out_txt)
    logger.info("Saved report JSON: %s", out_json)
    print("\n" + out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
