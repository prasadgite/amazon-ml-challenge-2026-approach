import csv
from pathlib import Path

from business_entity_resolution.evaluation.final_evaluator import FinalEvaluationConfig, FinalEvaluator
from business_entity_resolution.matching.model import FEATURE_COLUMNS, MatchModel
from business_entity_resolution.decision.calibration import DecisionEngine


def _write_fixture(tmp_path: Path):
    feature_path = tmp_path / "train_pair_features.tsv"
    gt_path = tmp_path / "train_ground_truth.tsv"
    s1_path = tmp_path / "train_source1.tsv"

    with s1_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["entity_id", "name"], delimiter="\t")
        w.writeheader()
        for i in range(120):
            w.writerow({"entity_id": f"S1_{i}", "name": f"Entity {i}"})

    with gt_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t")
        w.writeheader()
        for i in range(120):
            # Every fourth S1 is a singleton. Others have one true S2 target.
            matches = "" if i % 4 == 0 else f"S2_{i}"
            w.writerow({"source1_entity_id": f"S1_{i}", "matched_entity_ids": matches})

    columns = ["source1_id", "target_source", "target_id", *FEATURE_COLUMNS, "label"]
    with feature_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        w.writeheader()
        for i in range(120):
            # True pair for non-singletons.
            if i % 4 != 0:
                row = {c: "0" for c in columns}
                row.update(source1_id=f"S1_{i}", target_source="S2", target_id=f"S2_{i}", label="1")
                row.update(name_jaro_winkler="0.99", name_token_jaccard="1.0", address_token_jaccard="1.0",
                           country_match="1", name_exact="1", address_exact="1", strong_evidence_count="4",
                           combined_evidence_count="6", blocking_hit_count="2")
                w.writerow(row)

            # Hard negative candidate. For singleton S1s this is the only candidate.
            row = {c: "0" for c in columns}
            row.update(source1_id=f"S1_{i}", target_source="S2", target_id=f"S2_BAD_{i}", label="0")
            row.update(name_jaro_winkler="0.40", name_token_jaccard="0.10", address_token_jaccard="0.10",
                       country_match="0", strong_evidence_count="0", combined_evidence_count="0",
                       blocking_hit_count="1")
            w.writerow(row)

    return feature_path, gt_path, s1_path


def test_partition_is_entity_level_and_deterministic(tmp_path):
    evaluator = FinalEvaluator(FinalEvaluationConfig(n_folds=5, holdout_fraction=0.15))
    assignments = {}
    for i in range(200):
        assignments[f"S1_{i}"] = evaluator._role(f"S1_{i}")
    assert assignments == {
        f"S1_{i}": evaluator._role(f"S1_{i}") for i in range(200)
    }
    assert all(role in {"development", "holdout"} for role, _ in assignments.values())
    dev_folds = {fold for role, fold in assignments.values() if role == "development"}
    assert dev_folds == {0, 1, 2, 3, 4}


def test_official_singleton_macro_f05(tmp_path):
    _, gt_path, s1_path = _write_fixture(tmp_path)
    prediction_path = tmp_path / "pred.tsv"
    with prediction_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source1_id", "target_source", "target_id", "calibrated_score"], delimiter="\t")
        w.writeheader()
        # Predict nothing for every singleton and the true target for every non-singleton.
        for i in range(120):
            if i % 4 != 0:
                w.writerow({"source1_id": f"S1_{i}", "target_source": "S2", "target_id": f"S2_{i}", "calibrated_score": "0.99"})
    evaluator = FinalEvaluator(FinalEvaluationConfig())
    metrics = evaluator._s1_metrics(prediction_path, gt_path, s1_path, 0.5)
    assert metrics["macro_f05"] == 1.0
    assert metrics["singleton_false_positives"] == 0


def test_m9_end_to_end_is_test_blind(tmp_path):
    feature_path, gt_path, s1_path = _write_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts"
    evaluator = FinalEvaluator(FinalEvaluationConfig(
        n_folds=3,
        holdout_fraction=0.20,
        calibration_fraction=0.25,
        threshold_grid_size=31,
        max_positive_rows=500,
        max_negative_rows=1000,
        max_calibration_rows=500,
        top_errors=10,
    ))
    report = evaluator.run(feature_path, gt_path, s1_path, artifact_dir)

    assert report["data_contract"]["test_data_used"] is False
    assert report["data_contract"]["test_labels_used"] is False
    assert report["partitioning"]["no_s1_cross_partition"] is True
    assert report["oof"]["selected_metrics"]["s1_entities"] > 0
    assert report["holdout"]["metrics"]["s1_entities"] > 0
    assert Path(report["artifacts"]["oof_predictions"]).exists()
    assert Path(report["artifacts"]["holdout_predictions"]).exists()
    assert Path(report["artifacts"]["frozen_model"]).exists()
    assert Path(report["artifacts"]["frozen_policy"]).exists()

    model = MatchModel.load(Path(report["artifacts"]["frozen_model"]))
    policy = DecisionEngine.load(Path(report["artifacts"]["frozen_policy"]), model)
    assert policy.threshold is not None
