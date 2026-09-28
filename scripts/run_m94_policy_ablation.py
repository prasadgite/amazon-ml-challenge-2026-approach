#!/usr/bin/env python3
"""
M9.4.4: Controlled Decision Policy Ablation.

Evaluates decision policies on frozen OOF predictions and untouched Holdout predictions:
  P0: Control — Frozen Global Threshold (tau = 0.8750)
  P1-A: Strong Evidence Rescue (tau_rescue = 0.70, strong_evidence_count >= 1, country_conflict == 0)
  P1-B: Strong Evidence Rescue (tau_rescue = 0.60, strong_evidence_count >= 1, country_conflict == 0)
  P2: Strong Evidence with Margin Requirement (tau_rescue = 0.65, margin >= 0.10 over 2nd candidate)
  P3: S1-level Competitive Argmax Rescue (top-1 target per S1 if score >= 0.65, country_conflict == 0)

Outputs:
  artifacts/diagnostics/m94_policy_ablation.json
  artifacts/diagnostics/m94_policy_ablation.txt
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.logging_config import setup_logger


def compute_s1_metrics(
    decisions: set[tuple[str, str]],
    ground_truth: set[tuple[str, str]],
    all_s1_ids: set[str],
) -> dict[str, float]:
    """Compute exact macro F0.5, macro precision/recall, pair precision/recall, and singleton stats."""
    # Build truth map and pred map
    truth_map: dict[str, set[str]] = defaultdict(set)
    for s1, t in ground_truth:
        if s1 in all_s1_ids:
            truth_map[s1].add(t)

    pred_map: dict[str, set[str]] = defaultdict(set)
    for s1, t in decisions:
        if s1 in all_s1_ids:
            pred_map[s1].add(t)

    pair_tp = len(decisions & ground_truth)
    pair_fp = len(decisions - ground_truth)
    pair_fn = len(ground_truth - decisions)

    pair_prec = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) > 0 else 1.0
    pair_rec = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) > 0 else 0.0

    s1_precisions = []
    s1_recalls = []
    singleton_fps = 0
    total_singletons = 0

    for s1 in sorted(all_s1_ids):
        t_set = truth_map.get(s1, set())
        p_set = pred_map.get(s1, set())

        is_singleton = len(t_set) == 0
        if is_singleton:
            total_singletons += 1
            if len(p_set) > 0:
                singleton_fps += 1

        if not t_set and not p_set:
            s1_precisions.append(1.0)
            s1_recalls.append(1.0)
        elif not t_set and p_set:
            s1_precisions.append(0.0)
            s1_recalls.append(1.0)
        elif t_set and not p_set:
            s1_precisions.append(0.0)
            s1_recalls.append(0.0)
        else:
            tp = len(t_set & p_set)
            prec = tp / len(p_set)
            rec = tp / len(t_set)
            s1_precisions.append(prec)
            s1_recalls.append(rec)

    macro_prec = sum(s1_precisions) / len(s1_precisions) if s1_precisions else 0.0
    macro_rec = sum(s1_recalls) / len(s1_recalls) if s1_recalls else 0.0

    beta_sq = 0.25  # F0.5
    denom = (beta_sq * macro_prec) + macro_rec
    macro_f05 = ((1.0 + beta_sq) * macro_prec * macro_rec / denom) if denom > 0 else 0.0

    return {
        "macro_f05": round(macro_f05, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "pair_precision": round(pair_prec, 6),
        "pair_recall": round(pair_rec, 6),
        "pair_tp": pair_tp,
        "pair_fp": pair_fp,
        "pair_fn": pair_fn,
        "singleton_fps": singleton_fps,
        "total_singletons": total_singletons,
    }


def evaluate_policy_on_predictions(
    predictions_path: Path,
    ground_truth: set[tuple[str, str]],
    all_s1_ids: set[str],
    policy_fn: Callable[[dict[str, Any]], bool],
) -> dict[str, float]:
    """Stream predictions, apply policy_fn, and compute S1-level metrics."""
    decisions: set[tuple[str, str]] = set()

    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_id"].strip()
            t_id = row["target_id"].strip()
            if s1 not in all_s1_ids:
                continue

            parsed_row = {
                "source1_id": s1,
                "target_id": t_id,
                "calibrated_score": float(row.get("calibrated_score", 0.0)),
                "raw_score": float(row.get("raw_score", 0.0)),
                "strong_evidence_count": int(float(row.get("strong_evidence_count", 0))),
                "country_conflict": int(float(row.get("country_conflict", 0))),
                "name_jaro_winkler": float(row.get("name_jaro_winkler", 0.0)),
                "name_token_jaccard": float(row.get("name_token_jaccard", 0.0)),
                "address_token_jaccard": float(row.get("address_token_jaccard", 0.0)),
            }

            if policy_fn(parsed_row):
                decisions.add((s1, t_id))

    return compute_s1_metrics(decisions, ground_truth, all_s1_ids)


def evaluate_s1_competitive_policy(
    predictions_path: Path,
    ground_truth: set[tuple[str, str]],
    all_s1_ids: set[str],
    base_threshold: float = 0.8750,
    rescue_threshold: float = 0.65,
) -> dict[str, float]:
    """P3: S1-level competitive rescue policy."""
    s1_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_id"].strip()
            if s1 not in all_s1_ids:
                continue
            s1_candidates[s1].append({
                "target_id": row["target_id"].strip(),
                "score": float(row.get("calibrated_score", 0.0)),
                "country_conflict": int(float(row.get("country_conflict", 0))),
                "strong_evidence": int(float(row.get("strong_evidence_count", 0))),
            })

    decisions: set[tuple[str, str]] = set()

    for s1, cands in s1_candidates.items():
        # Sort candidates descending by score
        sorted_cands = sorted(cands, key=lambda x: -x["score"])
        accepted_for_s1 = []

        for c in sorted_cands:
            if c["country_conflict"] == 1:
                continue
            if c["score"] >= base_threshold:
                accepted_for_s1.append(c["target_id"])

        # If none met base threshold, check if top-1 qualifies for strong-evidence rescue
        if not accepted_for_s1 and sorted_cands:
            top = sorted_cands[0]
            if top["country_conflict"] == 0 and top["strong_evidence"] >= 1 and top["score"] >= rescue_threshold:
                # Require margin over 2nd candidate if 2nd candidate exists
                if len(sorted_cands) == 1 or (top["score"] - sorted_cands[1]["score"] >= 0.10):
                    accepted_for_s1.append(top["target_id"])

        for t in accepted_for_s1:
            decisions.add((s1, t))

    return compute_s1_metrics(decisions, ground_truth, all_s1_ids)


def main():
    paths = ProjectPaths()
    logger = setup_logger("run_m94_policy_ablation", log_file=paths.logs_dir / "m94_policy_ablation.log")
    logger.info("=== Starting M9.4 Decision Policy Ablation ===")

    eval_dir = paths.artifacts_dir / "scratch" / "grounded_sample" / "evaluation_b5"
    pred_dir = eval_dir / "predictions"
    oof_pred = pred_dir / "m9_oof_predictions.tsv"
    holdout_pred = pred_dir / "m9_holdout_predictions.tsv"
    gt_tsv = paths.artifacts_dir / "scratch" / "grounded_sample" / "train_ground_truth.tsv"

    # Load Ground Truth pairs
    gt_pairs: set[tuple[str, str]] = set()
    with gt_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("source1_entity_id") or row.get("source1_id") or "").strip()
            for t in (row.get("matched_entity_ids") or "").replace(";", ",").replace("|", ",").split(","):
                t = t.strip()
                if t:
                    gt_pairs.add((s1, t))

    # Load S1 partition sets
    oof_s1 = set()
    with oof_pred.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            oof_s1.add(row["source1_id"].strip())

    holdout_s1 = set()
    with holdout_pred.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            holdout_s1.add(row["source1_id"].strip())

    logger.info("OOF S1: %d | Holdout S1: %d", len(oof_s1), len(holdout_s1))

    # Base Threshold
    TAU = 0.8750

    # Policy definitions
    policies = {
        "P0 (Control: Global Threshold 0.8750)": lambda r: (
            r["calibrated_score"] >= TAU and r["country_conflict"] == 0
        ),
        "P1-A (Strong Evidence Rescue @ 0.70)": lambda r: (
            r["country_conflict"] == 0 and (
                r["calibrated_score"] >= TAU or
                (r["strong_evidence_count"] >= 1 and r["calibrated_score"] >= 0.70)
            )
        ),
        "P1-B (Strong Evidence Rescue @ 0.60)": lambda r: (
            r["country_conflict"] == 0 and (
                r["calibrated_score"] >= TAU or
                (r["strong_evidence_count"] >= 1 and r["calibrated_score"] >= 0.60)
            )
        ),
        "P2 (Strong Evidence @ 0.65 + Address Consistency)": lambda r: (
            r["country_conflict"] == 0 and (
                r["calibrated_score"] >= TAU or
                (r["strong_evidence_count"] >= 1 and r["calibrated_score"] >= 0.65 and r["address_token_jaccard"] > 0.0)
            )
        ),
    }

    results = []

    for name, p_fn in policies.items():
        logger.info("Evaluating %s...", name)
        oof_m = evaluate_policy_on_predictions(oof_pred, gt_pairs, oof_s1, p_fn)
        hold_m = evaluate_policy_on_predictions(holdout_pred, gt_pairs, holdout_s1, p_fn)
        gap = round(abs(oof_m["macro_f05"] - hold_m["macro_f05"]) * 100.0, 4)

        results.append({
            "policy": name,
            "oof": oof_m,
            "holdout": hold_m,
            "generalization_gap_pp": gap,
        })

    # Evaluate P3: S1-level competitive argmax rescue
    logger.info("Evaluating P3 (S1 Competitive Top-1 Rescue @ 0.65)...")
    oof_p3 = evaluate_s1_competitive_policy(oof_pred, gt_pairs, oof_s1, base_threshold=TAU, rescue_threshold=0.65)
    hold_p3 = evaluate_s1_competitive_policy(holdout_pred, gt_pairs, holdout_s1, base_threshold=TAU, rescue_threshold=0.65)
    results.append({
        "policy": "P3 (S1 Competitive Top-1 Rescue @ 0.65)",
        "oof": oof_p3,
        "holdout": hold_p3,
        "generalization_gap_pp": round(abs(oof_p3["macro_f05"] - hold_p3["macro_f05"]) * 100.0, 4),
    })

    # Reference values (P0)
    p0_holdout_f05 = results[0]["holdout"]["macro_f05"]

    # Compile report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_threshold": TAU,
        "reference_holdout_macro_f05": p0_holdout_f05,
        "policies": results,
    }

    out_json = paths.diagnostics_dir / "m94_policy_ablation.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    out_txt = paths.diagnostics_dir / "m94_policy_ablation.txt"
    lines = [
        "=" * 105,
        "M9.4: DECISION POLICY ABLATION REPORT",
        "=" * 105,
        f"Timestamp:              {report['timestamp']}",
        f"Base Global Threshold:  {TAU:.4f}",
        f"P0 Reference Holdout:   {p0_holdout_f05*100:.2f}% Macro F0.5",
        "",
        "-" * 105,
        f"{'Policy':<42} | {'OOF F0.5':>9} | {'Holdout F0.5':>12} | {'Delta vs P0':>11} | {'Holdout Prec':>12} | {'Sing FP':>7} | {'Gap':>8}",
        "-" * 105,
    ]

    for r in results:
        pol = r["policy"]
        oof_f = r["oof"]["macro_f05"] * 100.0
        h_f = r["holdout"]["macro_f05"] * 100.0
        delta = (r["holdout"]["macro_f05"] - p0_holdout_f05) * 100.0
        h_prec = r["holdout"]["macro_precision"] * 100.0
        s_fp = r["holdout"]["singleton_fps"]
        gap = r["generalization_gap_pp"]

        lines.append(
            f"{pol:<42} | {oof_f:>8.2f}% | {h_f:>11.2f}% | {delta:>+8.2f} pp | {h_prec:>11.2f}% | {s_fp:>7} | {gap:>7.2f} pp"
        )

    lines.extend([
        "-" * 105,
        "",
        "-" * 105,
        "PROMOTION GATE EVALUATION",
        "-" * 105,
    ])

    best_promoted = None
    best_gain = 0.0

    for r in results[1:]:  # Compare alternatives against P0
        gain = r["holdout"]["macro_f05"] - p0_holdout_f05
        s_fp = r["holdout"]["singleton_fps"]
        prec_drop = p0_holdout_f05 - r["holdout"]["macro_precision"]
        gap = r["generalization_gap_pp"]

        passes = (gain > 0.0) and (s_fp == 0) and (gap <= 0.50)
        lines.append(f"  * {r['policy']}:")
        lines.append(f"      - Holdout Gain vs P0: {gain*100:+.2f} pp (Required: > 0.0 pp)")
        lines.append(f"      - Singleton False Positives: {s_fp} (Required: 0)")
        lines.append(f"      - Generalization Gap: {gap:.2f} pp (Required: <= 0.50 pp)")
        lines.append(f"      - Verdict: {'PROMOTED' if passes else 'REJECTED'}")

        if passes and gain > best_gain:
            best_gain = gain
            best_promoted = r

    lines.extend([
        "",
        "=" * 105,
        f"FINAL DECISION: {'PROMOTE ' + best_promoted['policy'] if best_promoted else 'RETAIN P0 (FROZEN CONTROL AT TAU = ' + str(TAU) + ')'}",
        "=" * 105,
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved report TXT: %s", out_txt)
    logger.info("Saved report JSON: %s", out_json)
    print("\n" + out_txt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
