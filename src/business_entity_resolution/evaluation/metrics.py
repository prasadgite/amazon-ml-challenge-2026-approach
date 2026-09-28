"""
Official S1-Level Macro F0.5 and Pairwise Evaluation Metrics for Industrial Entity Resolution.
Strictly implements the Amazon ML 2026 competition evaluation metric, including singleton accounting.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class S1Score:
    """Individual entity resolution score for a single Source 1 record."""
    source1_id: str
    true_targets: tuple[str, ...]
    predicted_targets: tuple[str, ...]
    precision: float
    recall: float
    f05: float
    is_singleton: bool
    is_correct_singleton: bool


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregated evaluation metrics across an entire evaluation cohort."""
    total_s1: int
    total_singletons: int
    correct_singletons: int
    singleton_false_positives: int
    s1_with_truth: int
    s1_with_predictions: int

    # S1-Level Macro Metrics (Official Competition Metric)
    macro_f05: float
    macro_precision: float
    macro_recall: float

    # Pairwise Decision Metrics (Diagnostic)
    pair_tp: int
    pair_fp: int
    pair_fn: int
    pair_precision: float
    pair_recall: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_s1_f05(
    predicted: set[str],
    truth: set[str],
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """Calculate precision, recall, and F-beta score for a single S1 entity.

    Official Competition Rules:
      1. If truth is empty (singleton):
         - Empty prediction -> Precision=1.0, Recall=1.0, F-score=1.0
         - Non-empty prediction -> Precision=0.0, Recall=1.0, F-score=0.0
      2. If truth is non-empty:
         - Empty prediction -> Precision=0.0, Recall=0.0, F-score=0.0
         - Non-empty prediction:
           Precision = |P ∩ T| / |P|
           Recall = |P ∩ T| / |T|
           F-beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    """
    beta_sq = beta * beta

    if not truth:
        if not predicted:
            return 1.0, 1.0, 1.0
        return 0.0, 1.0, 0.0

    if not predicted:
        return 0.0, 0.0, 0.0

    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)

    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return precision, recall, 0.0

    f_beta = (1.0 + beta_sq) * (precision * recall) / denom
    return precision, recall, f_beta


def compute_cohort_metrics(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
    all_s1_ids: set[str] | list[str],
    beta: float = 0.5,
) -> EvaluationMetrics:
    """Compute official S1-level macro metrics and pairwise diagnostic metrics across an S1 cohort."""
    s1_scores: list[S1Score] = []
    total_singletons = 0
    correct_singletons = 0
    singleton_fps = 0
    s1_with_truth = 0
    s1_with_pred = 0

    pair_tp = 0
    pair_fp = 0
    pair_fn = 0

    for s1 in sorted(all_s1_ids):
        pred_set = predictions.get(s1, set())
        true_set = ground_truth.get(s1, set())

        is_singleton = len(true_set) == 0
        if is_singleton:
            total_singletons += 1
            if not pred_set:
                correct_singletons += 1
            else:
                singleton_fps += 1
        else:
            s1_with_truth += 1

        if pred_set:
            s1_with_pred += 1

        p, r, f = calculate_s1_f05(pred_set, true_set, beta=beta)
        s1_scores.append(S1Score(
            source1_id=s1,
            true_targets=tuple(sorted(true_set)),
            predicted_targets=tuple(sorted(pred_set)),
            precision=p,
            recall=r,
            f05=f,
            is_singleton=is_singleton,
            is_correct_singleton=is_singleton and not pred_set,
        ))

        # Pairwise tallies
        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        pair_tp += tp
        pair_fp += fp
        pair_fn += fn

    total_s1 = len(s1_scores)
    macro_f05 = statistics.mean(s.f05 for s in s1_scores) if total_s1 else 0.0
    macro_precision = statistics.mean(s.precision for s in s1_scores) if total_s1 else 0.0
    macro_recall = statistics.mean(s.recall for s in s1_scores) if total_s1 else 0.0

    pair_prec = (pair_tp / (pair_tp + pair_fp)) if (pair_tp + pair_fp) else 1.0
    pair_rec = (pair_tp / (pair_tp + pair_fn)) if (pair_tp + pair_fn) else 0.0

    return EvaluationMetrics(
        total_s1=total_s1,
        total_singletons=total_singletons,
        correct_singletons=correct_singletons,
        singleton_false_positives=singleton_fps,
        s1_with_truth=s1_with_truth,
        s1_with_predictions=s1_with_pred,
        macro_f05=macro_f05,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        pair_tp=pair_tp,
        pair_fp=pair_fp,
        pair_fn=pair_fn,
        pair_precision=pair_prec,
        pair_recall=pair_rec,
    )
