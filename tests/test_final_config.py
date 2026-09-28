from __future__ import annotations

import json
from pathlib import Path


CONFIG_PATH = Path("artifacts/final/final_config.json")


def load_config() -> dict:
    assert CONFIG_PATH.exists(), (
        f"Missing final configuration: {CONFIG_PATH}"
    )

    return json.loads(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_final_config_is_frozen():
    config = load_config()

    assert config["status"] == "FROZEN"


def test_final_pipeline_components():
    config = load_config()

    assert config["pipeline"] == {
        "blocking": "B5",
        "features": "E0",
        "model": "logistic_regression",
        "calibration": "platt",
        "decision_policy": "P1-A",
        "consolidation": "M8",
    }


def test_b5_configuration_is_frozen():
    config = load_config()
    blocking = config["blocking"]

    assert blocking["name"] == "B5"
    assert blocking["enable_alias_blocking"] is True
    assert blocking["enable_transliteration"] is True
    assert blocking["enable_country_fallback"] is True
    assert blocking["enable_smart_capping"] is True
    assert blocking["enable_secondary_identity"] is False


def test_e0_configuration_is_frozen():
    config = load_config()

    assert config["features"]["name"] == "E0"
    assert config["features"]["feature_count"] == 38


def test_p1a_configuration_is_frozen():
    config = load_config()
    policy = config["decision_policy"]

    assert policy["name"] == "P1-A"
    assert policy["primary_threshold"] == 0.875
    assert policy["strong_evidence_threshold"] == 0.7
    assert policy["minimum_strong_evidence_count"] == 1


def test_test_data_is_not_used_for_tuning():
    config = load_config()
    integrity = config["data_integrity"]

    assert integrity["test_data_used_for_tuning"] is False
    assert integrity["test_labels_used_for_tuning"] is False
