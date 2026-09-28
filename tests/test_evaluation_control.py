"""
Comprehensive unit tests for Industrial Entity Resolution Layer 0: Data & Evaluation Control.
"""

import pytest
from pathlib import Path
from business_entity_resolution.evaluation import (
    EvaluationPopulation,
    PopulationMismatchError,
    DatasetSplits,
    PartitionConfig,
    calculate_s1_f05,
    compute_cohort_metrics,
    audit_evaluation_leakage,
    ExperimentContract,
)


def test_population_singleton_accounting(tmp_path: Path):
    gt_file = tmp_path / "gt.tsv"
    gt_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1_1\tS2_1,S3_1\n"
        "S1_2\t\n"  # True singleton
        "S1_3\tS2_3\n"
        "S1_4\t\n",  # True singleton
        encoding="utf-8",
    )

    pop = EvaluationPopulation(
        population_id="test_pop",
        s1_ids={"S1_1", "S1_2", "S1_3", "S1_4"},
        truth=EvaluationPopulation.load_ground_truth(gt_file),
    )

    assert pop.total_s1 == 4
    assert pop.total_singletons == 2
    assert pop.singletons == {"S1_2", "S1_4"}
    assert pop.total_ground_truth_pairs == 3


def test_population_feature_alignment_invariant(tmp_path: Path):
    gt_file = tmp_path / "gt.tsv"
    gt_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1_1\tS2_1\n"
        "S1_2\tS2_2\n",
        encoding="utf-8",
    )
    pop = EvaluationPopulation(
        population_id="test_pop",
        s1_ids={"S1_1", "S1_2"},
        truth=EvaluationPopulation.load_ground_truth(gt_file),
    )

    # Valid: features cover all S1s
    pop.validate_feature_population({"S1_1", "S1_2"})

    # Invalid: non-singleton S1_2 is missing from features -> must raise PopulationMismatchError
    with pytest.raises(PopulationMismatchError, match="Feature population does not match"):
        pop.validate_feature_population({"S1_1"})


def test_dataset_splits_zero_leakage():
    s1_ids = {f"S1_{i}" for i in range(1000)}
    splits = DatasetSplits.create(s1_ids, PartitionConfig(seed=42, holdout_fraction=0.15, n_folds=5))

    # Invariants
    assert len(splits.holdout_s1) > 0
    assert len(splits.development_s1) > 0
    assert splits.holdout_s1.isdisjoint(splits.development_s1)
    assert (splits.holdout_s1 | splits.development_s1) == s1_ids

    # 5 folds disjointness in dev
    all_fold_s1 = set()
    for fold_idx in range(5):
        fold_s1 = splits.val_s1_for_fold(fold_idx)
        train_s1 = splits.train_s1_for_fold(fold_idx)
        assert fold_s1.isdisjoint(train_s1)
        assert fold_s1.isdisjoint(splits.holdout_s1)
        all_fold_s1.update(fold_s1)
    assert all_fold_s1 == splits.development_s1


def test_official_s1_f05_singleton_rules():
    # 1. Correct singleton: empty truth, empty prediction -> F0.5 = 1.0
    p, r, f = calculate_s1_f05(predicted=set(), truth=set(), beta=0.5)
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # 2. Corrupted singleton: empty truth, non-empty prediction -> F0.5 = 0.0
    p, r, f = calculate_s1_f05(predicted={"S2_99"}, truth=set(), beta=0.5)
    assert (p, r, f) == (0.0, 1.0, 0.0)

    # 3. Non-singleton: true match missed (empty prediction) -> F0.5 = 0.0
    p, r, f = calculate_s1_f05(predicted=set(), truth={"S2_1"}, beta=0.5)
    assert (p, r, f) == (0.0, 0.0, 0.0)

    # 4. Perfect match:
    p, r, f = calculate_s1_f05(predicted={"S2_1", "S3_1"}, truth={"S2_1", "S3_1"}, beta=0.5)
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # 5. Precision-heavy beta=0.5 behavior:
    # 1 TP, 1 FP on truth size 1 -> P = 0.5, R = 1.0
    # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
    p, r, f = calculate_s1_f05(predicted={"S2_1", "S2_WRONG"}, truth={"S2_1"}, beta=0.5)
    assert p == 0.5
    assert r == 1.0
    assert abs(f - 0.5555555) < 1e-5


def test_cohort_metrics_aggregation():
    truth = {
        "S1_1": {"S2_1"},
        "S1_2": {"S2_2", "S3_2"},
        "S1_3": set(),  # singleton
    }
    predictions = {
        "S1_1": {"S2_1"},      # perfect (F0.5=1.0)
        "S1_2": {"S2_2"},      # 1 TP, 1 FN (P=1.0, R=0.5 -> F0.5=0.8333)
        "S1_3": set(),         # perfect singleton (F0.5=1.0)
    }
    metrics = compute_cohort_metrics(predictions, truth, {"S1_1", "S1_2", "S1_3"})
    assert metrics.total_s1 == 3
    assert metrics.total_singletons == 1
    assert metrics.correct_singletons == 1
    assert metrics.singleton_false_positives == 0
    assert metrics.macro_precision == 1.0
    assert metrics.macro_f05 > 0.90


def test_leakage_audit_catches_violations():
    # Detects partition overlap
    report = audit_evaluation_leakage(
        train_s1={"S1_1", "S1_2"},
        holdout_s1={"S1_2", "S1_3"},  # S1_2 leaks into both!
    )
    assert not report.passed
    assert report.partition_overlap_count == 1
    assert any("S1 partition overlap" in v for v in report.violations)

    # Detects threshold tuning on holdout
    report2 = audit_evaluation_leakage(
        train_s1={"S1_1"},
        holdout_s1={"S1_2"},
        threshold_tuned_on={"S1_2"},
    )
    assert not report2.passed
    assert any("Threshold was tuned on holdout" in v for v in report2.violations)


def test_experiment_contract_hashing():
    c1 = ExperimentContract.create("EXP_1", "POP_100", {"threshold": 0.85, "model": "logistic"})
    c2 = ExperimentContract.create("EXP_2", "POP_100", {"threshold": 0.85, "model": "logistic"})
    c3 = ExperimentContract.create("EXP_3", "POP_100", {"threshold": 0.90, "model": "logistic"})

    assert c1.config_hash == c2.config_hash
    assert c1.config_hash != c3.config_hash
