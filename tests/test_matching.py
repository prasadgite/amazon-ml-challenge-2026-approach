import csv
from pathlib import Path

from business_entity_resolution.matching.model import FEATURE_COLUMNS, MatchModel, MatchModelConfig


def _write_fixture(path: Path):
    cols = ["source1_id", "target_source", "target_id", *FEATURE_COLUMNS, "label"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for i in range(80):
            positive = i % 2 == 0
            row = {c: "0" for c in cols}
            row.update(source1_id=f"s1-{i}", target_source="S2", target_id=f"t-{i}", label=str(int(positive)))
            row["name_jaro_winkler"] = "0.98" if positive else "0.10"
            row["name_token_jaccard"] = "1.0" if positive else "0.0"
            row["address_token_jaccard"] = "1.0" if positive else "0.0"
            row["country_match"] = "1" if positive else "0"
            row["name_exact"] = "1" if positive else "0"
            w.writerow(row)


def test_match_model_train_and_save(tmp_path):
    path = tmp_path / "features.tsv"
    model_path = tmp_path / "model.pkl"
    _write_fixture(path)
    model = MatchModel(MatchModelConfig(validation_fraction=0.25, max_validation_rows=1000))
    summary = model.fit(path)
    assert summary["train_positive"] > 0
    assert summary["train_negative"] > 0
    assert "validation" in summary
    model.save(model_path)
    assert model_path.exists()
    loaded = MatchModel.load(model_path)
    assert loaded.feature_columns == list(FEATURE_COLUMNS)


def test_threshold_evaluation(tmp_path):
    path = tmp_path / "features.tsv"
    _write_fixture(path)
    model = MatchModel(MatchModelConfig(validation_fraction=0.25, max_validation_rows=1000))
    model.fit(path)
    results = model.evaluate_thresholds(path, [0.5, 0.9])
    assert len(results) == 2
    assert all("precision" in r and "recall" in r for r in results)
