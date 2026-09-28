from pathlib import Path
import csv
import json

from business_entity_resolution.consolidation import ConsolidationConfig, EntityConsolidator, UnionFind


def _write(path: Path, rows):
    fields = ["source1_id", "target_source", "target_id", "calibrated_score", "decision", "decision_reason", "country_conflict"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)


def test_union_find_transitive():
    uf = UnionFind(); uf.union("a", "b"); uf.union("b", "c")
    assert uf.find("a") == uf.find("c")


def test_consolidates_transitive_matches(tmp_path):
    decisions = tmp_path / "decisions.tsv"
    _write(decisions, [
        {"source1_id":"1","target_source":"S2","target_id":"S2_10","calibrated_score":"0.99","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S2","target_id":"S2_10","calibrated_score":"0.98","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S3","target_id":"S3_20","calibrated_score":"0.97","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
    ])
    c = EntityConsolidator()
    state = c.build_clusters(decisions)
    assert len(state["clusters"]) == 1
    cluster = next(iter(state["clusters"].values()))
    assert cluster["s1_count"] == 2
    assert cluster["target_count"] == 2


def test_country_conflict_is_rejected(tmp_path):
    decisions = tmp_path / "decisions.tsv"
    _write(decisions, [{
        "source1_id":"1","target_source":"S2","target_id":"S2_10","calibrated_score":"0.999",
        "decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"1"
    }])
    state = EntityConsolidator(ConsolidationConfig(reject_country_conflict=True)).build_clusters(decisions)
    assert len(state["edges"]) == 0
    assert state["rejected"]["country_conflict"] == 1


def test_cluster_id_is_deterministic(tmp_path):
    decisions = tmp_path / "decisions.tsv"
    _write(decisions, [{
        "source1_id":"1","target_source":"S2","target_id":"S2_10","calibrated_score":"0.999",
        "decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"
    }])
    c = EntityConsolidator()
    a = c.build_clusters(decisions)
    b = c.build_clusters(decisions)
    assert set(a["clusters"]) == set(b["clusters"])


def test_transitive_false_positive_is_measured(tmp_path):
    decisions = tmp_path / "decisions.tsv"
    _write(decisions, [
        {"source1_id":"1","target_source":"S2","target_id":"S2_10","calibrated_score":"0.99","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S2","target_id":"S2_10","calibrated_score":"0.98","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S3","target_id":"S3_20","calibrated_score":"0.97","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
    ])
    gt = tmp_path / "gt.tsv"
    with gt.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t")
        w.writeheader(); w.writerow({"source1_entity_id":"1", "matched_entity_ids":"S2_10"}); w.writerow({"source1_entity_id":"2", "matched_entity_ids":"S2_10,S3_20"})
    state = EntityConsolidator().build_clusters(decisions)
    report = __import__("business_entity_resolution.consolidation", fromlist=["ConsolidationEvaluator"]).ConsolidationEvaluator().evaluate(state, gt)
    assert report["cluster_closure_metrics"]["precision"] == 0.75
    assert report["transitive_inference"]["inferred_pairs"] == 1
    assert report["transitive_inference"]["incorrect_inferred_pairs"] == 1


def test_explicit_negative_blocks_transitive_merge(tmp_path):
    decisions = tmp_path / "decisions.tsv"
    _write(decisions, [
        {"source1_id":"1","target_source":"S2","target_id":"S2_10","calibrated_score":"0.99","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S2","target_id":"S2_10","calibrated_score":"0.98","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"2","target_source":"S3","target_id":"S3_20","calibrated_score":"0.97","decision":"MATCH","decision_reason":"threshold_pass","country_conflict":"0"},
        {"source1_id":"1","target_source":"S3","target_id":"S3_20","calibrated_score":"0.995","decision":"NON_MATCH","decision_reason":"below_threshold","country_conflict":"0"},
    ])
    state = EntityConsolidator().build_clusters(decisions)
    assert state["rejected_merge_conflicts"] == 1
    assert len(state["clusters"]) == 2
