from __future__ import annotations

import json
from pathlib import Path


MANIFEST_PATH = Path("artifacts/final/model_manifest.json")


def load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"Missing model manifest: {MANIFEST_PATH}"
    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


def test_manifest_is_frozen():
    manifest = load_manifest()

    assert manifest["status"] == "FROZEN"


def test_model_manifest_is_logistic_platt():
    manifest = load_manifest()

    assert manifest["model"]["family"] == "logistic_regression"
    assert manifest["calibration"]["method"] == "platt"


def test_manifest_uses_e0():
    manifest = load_manifest()

    assert manifest["features"]["name"] == "E0"
    assert manifest["features"]["feature_count"] == 38


def test_manifest_uses_p1a():
    manifest = load_manifest()
    policy = manifest["decision_policy"]

    assert policy["name"] == "P1-A"
    assert policy["primary_threshold"] == 0.875
    assert policy["strong_evidence_threshold"] == 0.7
    assert policy["minimum_strong_evidence_count"] == 1


def test_manifest_does_not_use_test_data():
    manifest = load_manifest()

    assert manifest["training"]["test_data_used"] is False
    assert manifest["training"]["test_labels_used"] is False


def test_model_artifact_exists():
    manifest = load_manifest()

    artifact = Path(
        "artifacts/models",
        manifest["model"]["artifact"],
    )

    assert artifact.exists(), (
        f"Missing model artifact: {artifact}"
    )


def test_calibrator_artifact_exists():
    manifest = load_manifest()

    artifact = Path(
        "artifacts/models",
        manifest["calibration"]["artifact"],
    )

    assert artifact.exists(), (
        f"Missing calibrator artifact: {artifact}"
    )


def test_final_config_and_manifest_are_consistent():
    config = json.loads(
        Path(
            "artifacts/final/final_config.json"
        ).read_text(encoding="utf-8")
    )

    manifest = json.loads(
        Path(
            "artifacts/final/model_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert (
        config["decision_policy"]
        == manifest["decision_policy"]
    )

    assert (
        config["features"]["name"]
        == manifest["features"]["name"]
    )

    assert (
        config["features"]["feature_count"]
        == manifest["features"]["feature_count"]
    )

    assert (
        config["model"]["type"]
        == manifest["model"]["family"]
    )

    assert (
        config["calibration"]["method"]
        == manifest["calibration"]["method"]
    )
