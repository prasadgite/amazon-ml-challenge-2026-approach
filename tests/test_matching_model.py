import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.models.matching_model import (
    MatchModel,
    MatchModelConfig,
    ModelTrainer,
    ModelPredictor,
    is_validation_entity,
    compute_negative_difficulty,
)
from business_entity_resolution.features.pairwise import FeatureSchema


def test_leakage_safe_entity_splitting():
    # 1. Determinism: identical IDs must yield identical partitions
    assert is_validation_entity("S1_100", 0.20) == is_validation_entity("S1_100", 0.20)
    assert is_validation_entity("S1_200", 0.20) == is_validation_entity("S1_200", 0.20)

    # 2. Distribution: across 1000 IDs, fraction should be approximately 0.20
    ids = [f"ENTITY_{i}" for i in range(1000)]
    val_count = sum(is_validation_entity(i, 0.20) for i in ids)
    assert 150 <= val_count <= 250


def test_negative_difficulty_scoring():
    # Near-miss negative (same name, high address similarity)
    hard_neg = {
        "name_jaro_winkler": 0.95,
        "address_edit_similarity": 0.90,
        "name_token_jaccard": 0.80,
        "address_token_jaccard": 0.85,
        "combined_evidence_count": 5,
        "strong_evidence_count": 2,
        "blocking_hit_count": 4,
    }
    # Trivial negative (low similarity)
    easy_neg = {
        "name_jaro_winkler": 0.10,
        "address_edit_similarity": 0.05,
        "name_token_jaccard": 0.0,
        "address_token_jaccard": 0.0,
        "combined_evidence_count": 0,
        "strong_evidence_count": 0,
        "blocking_hit_count": 1,
    }

    hard_diff = compute_negative_difficulty(hard_neg)
    easy_diff = compute_negative_difficulty(easy_neg)
    assert hard_diff > easy_diff
    assert hard_diff > 10.0
    assert easy_diff < 1.0


def test_match_model_fit_predict_and_coefficients(tmp_path):
    feature_names = ["feat1", "feat2", "feat3"]
    model = MatchModel(feature_names=feature_names, model_type="logistic", random_state=42)

    X = np.array([
        [1.0, 1.0, 1.0],
        [0.9, 0.8, 1.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
    ], dtype=np.float32)
    y = np.array([1, 1, 0, 0], dtype=np.int32)

    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 4
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] > probs[2]

    coefs = model.get_coefficients()
    assert len(coefs) == 3
    assert set(c["feature"] for c in coefs) == set(feature_names)
    assert all("coefficient" in c and "odds_multiplier" in c for c in coefs)


def test_model_serialization_and_deserialization(tmp_path):
    feature_names = ["f1", "f2"]
    model = MatchModel(feature_names=feature_names, model_type="logistic", random_state=42)
    X = np.array([[1.0, 0.5], [0.0, 0.2]], dtype=np.float32)
    y = np.array([1, 0], dtype=np.int32)
    model.fit(X, y)

    model_file = tmp_path / "model.pkl"
    model.save(model_file)
    assert model_file.exists()

    loaded = MatchModel.load(model_file)
    probs_orig = model.predict_proba(X)
    probs_loaded = loaded.predict_proba(X)
    np.testing.assert_allclose(probs_orig, probs_loaded)


def _generate_synthetic_features_tsv(path: Path, num_s1: int = 40):
    columns = FeatureSchema.columns(include_label=True)
    feature_cols = [c for c in columns if c not in ("source1_id", "target_source", "target_id", "label")]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)

        for i in range(num_s1):
            s1_id = f"S1_{i:03d}"
            # Positive candidate
            pos_row = [s1_id, "S2", f"S2_{i:03d}"]
            for col in feature_cols:
                if "exact" in col:
                    pos_row.append(1)
                elif "jaccard" in col or "similarity" in col:
                    pos_row.append(0.95)
                elif "count" in col:
                    pos_row.append(4)
                elif "conflict" in col:
                    pos_row.append(0)
                else:
                    pos_row.append(1)
            pos_row.append(1)  # label = 1
            writer.writerow(pos_row)

            # Negative candidate 1 (near-miss hard negative)
            neg1_row = [s1_id, "S3", f"S3_{i:03d}_hard"]
            for col in feature_cols:
                if "exact" in col:
                    neg1_row.append(0)
                elif "jaccard" in col or "similarity" in col:
                    neg1_row.append(0.70)
                elif "count" in col:
                    neg1_row.append(2)
                elif "conflict" in col:
                    neg1_row.append(0)
                else:
                    neg1_row.append(0)
            neg1_row.append(0)  # label = 0
            writer.writerow(neg1_row)

            # Negative candidate 2 (easy negative)
            neg2_row = [s1_id, "S3", f"S3_{i:03d}_easy"]
            for col in feature_cols:
                if "conflict" in col:
                    neg2_row.append(1)
                elif "similarity" in col or "jaccard" in col:
                    neg2_row.append(0.10)
                else:
                    neg2_row.append(0)
            neg2_row.append(0)  # label = 0
            writer.writerow(neg2_row)


def test_model_trainer_end_to_end(tmp_path):
    tsv_path = tmp_path / "train_pair_features.tsv"
    _generate_synthetic_features_tsv(tsv_path, num_s1=50)

    model_path = tmp_path / "models" / "match_model.pkl"
    diag_dir = tmp_path / "diagnostics"

    cfg = MatchModelConfig(
        validation_fraction=0.20,
        hard_negative_ratio=2,
        random_state=42,
    )
    trainer = ModelTrainer(cfg)
    summary = trainer.train_and_evaluate(tsv_path, model_path, diag_dir)

    assert model_path.exists()
    assert (diag_dir / "match_model.json").exists()
    assert (diag_dir / "match_model_coefficients.json").exists()
    assert (diag_dir / "match_model_thresholds.json").exists()

    assert summary["training_samples"]["positive_pairs"] > 0
    assert summary["training_samples"]["hard_negative_pairs"] > 0
    assert summary["validation_performance"]["pr_auc"] > 0.80

    # Inspect thresholds file
    with open(diag_dir / "match_model_thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    assert len(thresholds) == len(cfg.threshold_list)
    assert any(t["threshold"] == 0.50 for t in thresholds)
    assert any(t["threshold"] == 0.90 for t in thresholds)


def test_model_predictor_scoring(tmp_path):
    # Train a model first
    train_tsv = tmp_path / "train_features.tsv"
    _generate_synthetic_features_tsv(train_tsv, num_s1=30)
    model_path = tmp_path / "model.pkl"
    diag_dir = tmp_path / "diag"
    ModelTrainer().train_and_evaluate(train_tsv, model_path, diag_dir)

    # Now score a test features TSV
    test_tsv = tmp_path / "test_features.tsv"
    _generate_synthetic_features_tsv(test_tsv, num_s1=10)
    scores_tsv = tmp_path / "test_pair_scores.tsv"

    predictor = ModelPredictor(model_path)
    count = predictor.predict(test_tsv, scores_tsv, batch_size=5)
    assert count == 30  # 10 S1 * 3 candidates each
    assert scores_tsv.exists()

    with open(scores_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames == ["source1_id", "target_source", "target_id", "score", "label"]
        rows = list(reader)
        assert len(rows) == 30
        for r in rows:
            score = float(r["score"])
            assert 0.0 <= score <= 1.0
