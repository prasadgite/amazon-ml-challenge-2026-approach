"""
M9.4: Error Analysis and Strong-Evidence Forensic Diagnostic.

Analyzes false negatives (FNs) across OOF development and Holdout partitions:
  - Classifies FNs into standardized error buckets (Blocking Miss, Strong Evidence Below Threshold,
    Weak Evidence, Address Conflict, Country Conflict, Competition, Other).
  - Investigates the strong_evidence_count >= 1 score distribution for both positives and negatives.
  - Simulates the precision / recall trade-offs for candidate rescue policies.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FalseNegativeRecord:
    source1_id: str
    target_id: str
    target_source: str
    partition: str
    calibrated_score: float
    raw_score: float
    name_jaro_winkler: float
    name_token_jaccard: float
    address_token_jaccard: float
    blocking_hit_count: int
    strong_evidence_count: int
    country_conflict: int
    alias_exact: int
    domain_exact: int
    category: str


def classify_false_negative(
    calibrated_score: float,
    country_conflict: int,
    strong_evidence_count: int,
    name_jaro_winkler: float,
    name_token_jaccard: float,
    address_token_jaccard: float,
    blocking_hit_count: int,
    is_blocked: bool = True,
) -> str:
    """Classify an FN into a mutually exclusive root-cause category."""
    if not is_blocked or blocking_hit_count == 0:
        return "1_BLOCKING_MISS"

    if country_conflict == 1:
        return "6_COUNTRY_CONFLICT"

    # Strong evidence that scored below decision threshold
    if strong_evidence_count >= 1 or name_token_jaccard >= 0.85 or (name_jaro_winkler >= 0.95 and address_token_jaccard >= 0.20):
        # Check if address strongly conflicts despite high name similarity
        if address_token_jaccard == 0.0 and name_token_jaccard >= 0.90:
            return "5_ADDRESS_CONFLICT"
        return "2_STRONG_EVIDENCE_BELOW_THRESHOLD"

    if address_token_jaccard == 0.0 and name_jaro_winkler >= 0.80:
        return "5_ADDRESS_CONFLICT"

    # Weak evidence
    if strong_evidence_count == 0 and name_jaro_winkler < 0.80 and name_token_jaccard < 0.50:
        return "3_WEAK_EVIDENCE"

    return "7_OTHER"


def extract_fns_from_predictions(
    predictions_path: Path,
    ground_truth_path: Path,
    evaluated_s1_ids: set[str],
    threshold: float,
    partition: str,
) -> list[FalseNegativeRecord]:
    """Extract and classify all FNs for a given partition."""
    # 1. Load ground truth for evaluated S1 entities
    gt_pairs: set[tuple[str, str]] = set()
    with ground_truth_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = (row.get("source1_entity_id") or row.get("source1_id") or "").strip()
            if s1 not in evaluated_s1_ids:
                continue
            raw_targets = (row.get("matched_entity_ids") or "").strip()
            for t in raw_targets.replace(";", ",").replace("|", ",").split(","):
                t = t.strip()
                if t:
                    gt_pairs.add((s1, t))

    # 2. Stream scored predictions
    scored_gt: set[tuple[str, str]] = set()
    accepted_gt: set[tuple[str, str]] = set()
    fn_records: list[FalseNegativeRecord] = []

    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1 = row["source1_id"].strip()
            t_id = row["target_id"].strip()
            t_src = row.get("target_source", "S2" if t_id.startswith("S2") else "S3").strip()
            pair = (s1, t_id)

            label = int(row.get("label", 0))
            score = float(row.get("calibrated_score", 0.0))
            raw_s = float(row.get("raw_score", 0.0))

            if pair in gt_pairs or label == 1:
                scored_gt.add(pair)
                if score >= threshold:
                    accepted_gt.add(pair)
                else:
                    # Scored FN
                    c_conf = int(float(row.get("country_conflict", 0)))
                    sec = int(float(row.get("strong_evidence_count", 0)))
                    njw = float(row.get("name_jaro_winkler", 0.0))
                    ntj = float(row.get("name_token_jaccard", 0.0))
                    atj = float(row.get("address_token_jaccard", 0.0))
                    bhc = int(float(row.get("blocking_hit_count", 1)))
                    ae = int(float(row.get("alias_exact", 0)))
                    de = int(float(row.get("domain_exact", 0)))

                    category = classify_false_negative(
                        calibrated_score=score,
                        country_conflict=c_conf,
                        strong_evidence_count=sec,
                        name_jaro_winkler=njw,
                        name_token_jaccard=ntj,
                        address_token_jaccard=atj,
                        blocking_hit_count=bhc,
                        is_blocked=True,
                    )

                    fn_records.append(FalseNegativeRecord(
                        source1_id=s1,
                        target_id=t_id,
                        target_source=t_src,
                        partition=partition,
                        calibrated_score=round(score, 6),
                        raw_score=round(raw_s, 6),
                        name_jaro_winkler=round(njw, 4),
                        name_token_jaccard=round(ntj, 4),
                        address_token_jaccard=round(atj, 4),
                        blocking_hit_count=bhc,
                        strong_evidence_count=sec,
                        country_conflict=c_conf,
                        alias_exact=ae,
                        domain_exact=de,
                        category=category,
                    ))

    # 3. Identify Unblocked Ground Truth (Blocking Misses)
    unblocked = gt_pairs - scored_gt
    for s1, t_id in unblocked:
        t_src = "S2" if t_id.startswith("S2") else "S3"
        fn_records.append(FalseNegativeRecord(
            source1_id=s1,
            target_id=t_id,
            target_source=t_src,
            partition=partition,
            calibrated_score=0.0,
            raw_score=0.0,
            name_jaro_winkler=0.0,
            name_token_jaccard=0.0,
            address_token_jaccard=0.0,
            blocking_hit_count=0,
            strong_evidence_count=0,
            country_conflict=0,
            alias_exact=0,
            domain_exact=0,
            category="1_BLOCKING_MISS",
        ))

    return fn_records


def analyze_strong_evidence_scores(
    predictions_path: Path,
    threshold: float,
) -> dict[str, Any]:
    """Quantify score distribution and potential rescue safety for strong_evidence_count >= 1."""
    pos_scores = []
    neg_scores = []

    with predictions_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sec = int(float(row.get("strong_evidence_count", 0)))
            if sec < 1:
                continue

            score = float(row.get("calibrated_score", 0.0))
            label = int(row.get("label", 0))

            if label == 1:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

    def dist_summary(scores: list[float]) -> dict[str, float]:
        if not scores:
            return {"count": 0, "mean": 0.0, "min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.array(scores, dtype=np.float64)
        return {
            "count": len(scores),
            "mean": round(float(np.mean(arr)), 6),
            "min": round(float(np.min(arr)), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "median": round(float(np.percentile(arr, 50)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    # Simulation sweep for rescue threshold
    simulations = []
    rescue_thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.825, 0.85, round(threshold, 4)]
    for r_thresh in sorted(set(rescue_thresholds)):
        # True pairs recovered that currently score below threshold
        rescued_tp = sum(1 for s in pos_scores if r_thresh <= s < threshold)
        # False positives introduced that currently score below threshold
        introduced_fp = sum(1 for s in neg_scores if s >= r_thresh)
        # Precision among rescued candidates
        total_rescued = rescued_tp + sum(1 for s in neg_scores if r_thresh <= s < threshold)
        rescue_prec = (rescued_tp / total_rescued) if total_rescued > 0 else 1.0

        simulations.append({
            "rescue_threshold": r_thresh,
            "additional_tp_recovered": rescued_tp,
            "introduced_fp": sum(1 for s in neg_scores if r_thresh <= s < threshold),
            "rescue_precision": round(rescue_prec, 4),
            "safe": bool(introduced_fp == 0),
        })

    return {
        "strong_evidence_rows": len(pos_scores) + len(neg_scores),
        "true_positives": dist_summary(pos_scores),
        "false_positives": dist_summary(neg_scores),
        "rescue_simulations": simulations,
    }
