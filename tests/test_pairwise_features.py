import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
from business_entity_resolution.features.pairwise import PairFeatureEngine, FeatureSchema


def _write(path: Path, header, rows):
    path.write_text("\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


def test_pairwise_features_include_strong_identity_signals(tmp_path):
    header = ["entity_id", "business_name", "business_address", "country"]
    s1 = tmp_path / "train_source1.tsv"
    s2 = tmp_path / "train_source2.tsv"
    s3 = tmp_path / "train_source3.tsv"
    gt = tmp_path / "train_ground_truth.tsv"
    _write(s1, header, [("S1_1", "Acme Corporation", "12 Main Street 411001", "India")])
    _write(s2, header, [("S2_1", "ACME Corp", "12 Main St 411001", "IN")])
    _write(s3, header, [("S3_1", "Other Company", "99 Other Road 411002", "IN")])
    _write(gt, ["source1_entity_id", "matched_entity_ids"], [("S1_1", "S2_1")])

    blocking_db = tmp_path / "blocking.sqlite"
    blocker = BlockingEngine(blocking_db, BlockingConfig(max_bucket_size=50))
    blocker.initialize("train")
    blocker.build_target_index({"S2": s2, "S3": s3})
    blocker.generate_candidates(s1, "train")

    feature_db = tmp_path / "features.sqlite"
    engine = PairFeatureEngine(feature_db)
    engine.initialize()
    engine.build_record_store({"S1": s1, "S2": s2, "S3": s3})
    assert engine.load_ground_truth(gt) == 1
    output = tmp_path / "pair_features.tsv"
    stats = engine.generate_features(blocking_db, output, include_labels=True)

    assert stats["pairs"] >= 1
    assert stats["positive"] == 1
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    positive = next(r for r in rows if r["label"] == "1")
    assert positive["target_id"] == "S2_1"
    assert positive["country_match"] == "1"
    assert positive["address_exact"] == "1"
    assert float(positive["name_jaro_winkler"]) > 0.7
    assert int(positive["blocking_hit_count"]) >= 1


def test_feature_schema_is_stable(tmp_path):
    columns = FeatureSchema.columns(True)
    assert columns[0:3] == ["source1_id", "target_source", "target_id"]
    assert columns[-1] == "label"
    assert len(columns) == len(set(columns))


if __name__ == "__main__":
    import tempfile
    print("Running pairwise features unit tests...")
    with tempfile.TemporaryDirectory() as td:
        t_path = Path(td)
        test_pairwise_features_include_strong_identity_signals(t_path)
        test_feature_schema_is_stable(t_path)
    print("ALL PAIRWISE FEATURE TESTS PASSED!")
