from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from business_entity_resolution.evaluation.final_evaluator import (
    FinalEvaluationConfig,
    FinalEvaluator,
)


def _rows():
    rows = []

    # Positive examples.
    for i in range(20):
        rows.append(
            {
                "source1_id": f"S1-P{i}",
                "target_source": "S2",
                "target_id": f"S2-P{i}",
                "label": "1",
                "name_jaro_winkler": "0.95",
                "name_edit_similarity": "0.92",
                "name_token_jaccard": "0.90",
                "address_jaro_winkler": "0.85",
                "address_token_jaccard": "0.80",
                "country_match": "1",
                "domain_exact": "1",
                "alias_exact": "0",
            }
        )

    # Negative examples.
    for i in range(20):
        rows.append(
            {
                "source1_id": f"S1-N{i}",
                "target_source": "S2",
                "target_id": f"S2-N{i}",
                "label": "0",
                "name_jaro_winkler": "0.20",
                "name_edit_similarity": "0.15",
                "name_token_jaccard": "0.10",
                "address_jaro_winkler": "0.20",
                "address_token_jaccard": "0.10",
                "country_match": "0",
                "domain_exact": "0",
                "alias_exact": "0",
            }
        )

    return rows


def test_hgb_configuration_is_available():
    config = FinalEvaluationConfig(
        model_type="hgb",
        hgb_max_iter=25,
        hgb_learning_rate=0.05,
        hgb_max_leaf_nodes=15,
        hgb_min_samples_leaf=5,
        hgb_l2_regularization=1.0,
    )

    assert config.model_type == "hgb"
    assert config.hgb_max_iter == 25
    assert config.hgb_learning_rate == 0.05
    assert config.hgb_max_leaf_nodes == 15
    assert config.hgb_min_samples_leaf == 5
    assert config.hgb_l2_regularization == 1.0


def test_hgb_fits_small_training_population():
    pytest.importorskip("sklearn")

    config = FinalEvaluationConfig(
        model_type="hgb",
        hgb_max_iter=25,
        hgb_learning_rate=0.05,
        hgb_max_leaf_nodes=15,
        hgb_min_samples_leaf=5,
        hgb_l2_regularization=1.0,
    )

    evaluator = FinalEvaluator(config)

    rows = _rows()
    model = evaluator._fit_pipeline(rows)

    probabilities = model.predict_proba(
        evaluator._matrix(rows)
    )[:, 1]

    assert len(probabilities) == len(rows)
    assert all(0.0 <= float(p) <= 1.0 for p in probabilities)


def test_invalid_model_type_is_rejected():
    config = FinalEvaluationConfig(
        model_type="invalid-model",
    )

    evaluator = FinalEvaluator(config)

    with pytest.raises(ValueError, match="model_type"):
        evaluator._fit_pipeline(_rows())
