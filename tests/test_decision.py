import csv
from pathlib import Path

from business_entity_resolution.matching.model import FEATURE_COLUMNS, MatchModel, MatchModelConfig
from business_entity_resolution.decision.calibration import DecisionConfig, DecisionEngine


def _write_fixture(path: Path):
    cols = ["source1_id", "target_source", "target_id", *FEATURE_COLUMNS, "label"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for i in range(240):
            positive = i % 3 == 0
            row = {c: "0" for c in cols}
            row.update(source1_id=f"s1-{i}", target_source="S2", target_id=f"t-{i}", label=str(int(positive)))
            row["name_jaro_winkler"] = "0.99" if positive else "0.30"
            row["name_token_jaccard"] = "1.0" if positive else "0.1"
            row["address_token_jaccard"] = "1.0" if positive else "0.1"
            row["country_match"] = "1" if positive else "0"
            row["name_exact"] = "1" if positive else "0"
            w.writerow(row)


def test_m7_calibration_and_threshold(tmp_path):
    path = tmp_path / "features.tsv"
    model_path = tmp_path / "model.pkl"
    decision_path = tmp_path / "decision.pkl"
    _write_fixture(path)

    model = MatchModel(MatchModelConfig(validation_fraction=0.25, max_validation_rows=1000))
    model.fit(path)
    model.save(model_path)

    engine = DecisionEngine(model, DecisionConfig(target_precision=0.99, calibration_fraction=0.5))
    report = engine.fit(path)
    assert engine.threshold is not None
    assert report["evaluation"]["threshold_selection"]["status"] in {
        "target_precision_reached", "target_precision_not_reached"
    }
    engine.save(decision_path)
    assert decision_path.exists()


def test_m7_country_conflict_veto(tmp_path):
    path = tmp_path / "features.tsv"
    _write_fixture(path)
    model = MatchModel(MatchModelConfig(validation_fraction=0.25, max_validation_rows=1000))
    model.fit(path)
    engine = DecisionEngine(model, DecisionConfig(target_precision=0.90, calibration_fraction=0.5))
    engine.fit(path)
    row = {c: "0" for c in FEATURE_COLUMNS}
    row["country_conflict"] = "1"
    decision, score, reason = engine.decision(row, 0.999)
    assert decision == "NON_MATCH"
    assert reason == "country_conflict_veto"
    assert 0 <= score <= 1
