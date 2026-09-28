"""
Unit tests for Milestone 4 - Candidate Blocking & Recall Evaluation.
9 automated tests covering multi-pass blocking, SQLite indexing, bucket capping,
candidate deduplication, strategy provenance, zero-match handling, and recall evaluation.
"""

import sys
from pathlib import Path

# Add src to pythonpath before imports
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.io.record import EntityRecord  # noqa: E402
from business_entity_resolution.normalization.normalizer import TextNormalizer  # noqa: E402
from business_entity_resolution.blocking.keys import (  # noqa: E402
    generate_multi_pass_keys,
    ALL_BLOCKING_STRATEGIES,
)
from business_entity_resolution.blocking.sqlite_index import SQLiteBlockingIndex  # noqa: E402
from business_entity_resolution.paths import ProjectPaths  # noqa: E402
from business_entity_resolution.config import PipelineConfig  # noqa: E402
from business_entity_resolution.blocking.evaluator import BlockingEvaluator  # noqa: E402


def test_multi_pass_blocking_keys():
    """Test 1: Verify all 10 multi-pass blocking strategies are generated."""
    normalizer = TextNormalizer()
    rec = EntityRecord(
        entity_id="S1-505333132",
        business_name="Clemons Silver Eastern Inc aka Halodelta",
        business_address="1619 Julia Park Drive, Spring, TX 77386",
        country="US",
    )
    norm = normalizer.normalize(rec)
    keys = generate_multi_pass_keys(norm)

    for strat in ALL_BLOCKING_STRATEGIES:
        assert strat in keys, f"Missing strategy: {strat}"
    assert len(keys["exact_name"]) >= 1
    assert len(keys["exact_addr"]) >= 1
    assert len(keys["exact_name_addr"]) >= 1
    assert len(keys["name_prefix"]) >= 1
    assert len(keys["street_prefix"]) >= 1
    assert len(keys["name_4grams"]) >= 1
    assert len(keys["addr_4grams"]) >= 1
    assert len(keys["hn_postal"]) >= 1


def test_exact_matches_strategy():
    """Test 2: Verify exact name and address keys format and country isolation."""
    normalizer = TextNormalizer()
    rec = EntityRecord(
        entity_id="S2-001",
        business_name="Alpha Omega Enterprises",
        business_address="100 Main Street, Dallas, TX 75201",
        country="US",
    )
    norm = normalizer.normalize(rec)
    keys = generate_multi_pass_keys(norm)

    assert any(k.startswith("us_en_") for k in keys["exact_name"])
    assert any("alpha omega" in k for k in keys["exact_name"])
    assert any(k.startswith("us_ea_") for k in keys["exact_addr"])
    assert any(k.startswith("us_ena_") for k in keys["exact_name_addr"])


def test_ngram_strategies():
    """Test 3: Verify name and address 4-grams extraction."""
    normalizer = TextNormalizer()
    rec = EntityRecord(
        entity_id="S3-101",
        business_name="Pharmacie Lafayette",
        business_address="12 Boulevard Haussmann, Paris 75009",
        country="France",
    )
    norm = normalizer.normalize(rec)
    keys = generate_multi_pass_keys(norm)

    assert len(keys["name_4grams"]) > 0
    assert any("phar" in k for k in keys["name_4grams"])
    assert len(keys["addr_4grams"]) > 0
    assert any("haus" in k for k in keys["addr_4grams"])


def test_hn_postal_and_domain_alias():
    """Test 4: Verify house number + postal, domain, and DBA/alias blocking."""
    normalizer = TextNormalizer()
    rec = EntityRecord(
        entity_id="S1-999",
        business_name="Acme Corp aka Acme Logistics (www.acme-corp.com)",
        business_address="59/101 MG Road, Bangalore 560001",
        country="India",
    )
    norm = normalizer.normalize(rec)
    keys = generate_multi_pass_keys(norm)

    assert any("59/101" in k and "560001" in k for k in keys["hn_postal"])
    assert any("acme" in k for k in keys["domain"])
    assert any("acme logistics" in k for k in keys["dba_alias"])


def test_sqlite_index_creation_and_insertion():
    """Test 5: Verify SQLiteBlockingIndex initialization, insertion, and stats."""
    index = SQLiteBlockingIndex(":memory:")
    normalizer = TextNormalizer()

    targets = [
        EntityRecord("S2-001", "Bistro Paris", "10 Rue de la Paix", "France"),
        EntityRecord("S3-002", "Bistro Paris", "10 Rue de la Paix", "France"),
        EntityRecord("S2-003", "Tech Hub", "500 Silicon Way", "US"),
    ]
    for t in targets:
        index.add_target(normalizer.normalize(t))

    stats = index.get_bucket_statistics()
    assert stats["total_unique_keys"] > 0
    assert stats["total_postings"] > 0
    assert stats["total_indexed_entities"] == 3
    assert stats["max"] >= 2
    index.close()


def test_bucket_capping():
    """Test 6: Verify bucket-size capping to prevent candidate explosion."""
    index = SQLiteBlockingIndex(":memory:")
    normalizer = TextNormalizer()

    # Add 10 targets that share the exact same name
    for i in range(10):
        rec = EntityRecord(f"S2-{i:03d}", "Universal Store", f"{i} Main St", "US")
        index.add_target(normalizer.normalize(rec))

    query_rec = normalizer.normalize(EntityRecord("S1-001", "Universal Store", "99 Broadway", "US"))

    # When cap is 5, the high-frequency "Universal Store" exact name bucket (size 10) must be skipped
    cands_capped = index.query_candidates(query_rec, max_bucket_size=5)
    # When cap is 15, the bucket is included
    cands_uncapped = index.query_candidates(query_rec, max_bucket_size=15)

    assert len(cands_capped) < len(cands_uncapped)
    stats = index.get_bucket_statistics(max_bucket_size=5)
    assert stats["capped_keys"] > 0
    index.close()


def test_candidate_deduplication_and_provenance():
    """Test 7: Verify candidate deduplication and tracking of hits + strategies."""
    index = SQLiteBlockingIndex(":memory:")
    normalizer = TextNormalizer()

    target = EntityRecord("S2-777", "Apex Logistics Ltd", "44 Industrial Parkway", "US")
    index.add_target(normalizer.normalize(target))

    query = normalizer.normalize(EntityRecord("S1-888", "Apex Logistics Ltd", "44 Industrial Parkway", "US"))
    cands = index.query_candidates(query, max_bucket_size=100)

    assert "S2-777" in cands
    cand_info = cands["S2-777"]
    assert cand_info["hits"] >= 3  # Matches exact name, exact addr, prefix, ngrams, etc.
    assert "exact_name" in cand_info["strategies"]
    assert "exact_addr" in cand_info["strategies"]
    index.close()


def test_zero_match_s1_handling():
    """Test 8: Verify S1 entity with 0 candidate matches is handled cleanly."""
    index = SQLiteBlockingIndex(":memory:")
    normalizer = TextNormalizer()

    target = EntityRecord("S2-001", "Bistro Paris", "10 Rue de la Paix", "France")
    index.add_target(normalizer.normalize(target))

    # Completely different entity and country
    query = normalizer.normalize(EntityRecord("S1-999", "Zenith Electronics", "100 High Tech Blvd", "US"))
    cands = index.query_candidates(query, max_bucket_size=100)

    assert len(cands) == 0
    assert isinstance(cands, dict)
    index.close()


def test_recall_and_reduction_ratio_evaluation():
    """Test 9: Verify recall and candidate reduction ratio calculation."""
    paths = ProjectPaths()
    config = PipelineConfig(sample_size=5)
    evaluator = BlockingEvaluator(paths, config)

    res = evaluator.run_evaluation(sample_s1=5, background_distractor_count=50, max_bucket_size=500)
    assert res["total_s1_evaluated"] == 5
    assert res["recall_pct"] >= 95.0
    assert res["reduction_ratio"] > 0.50
    assert "train_blocking_recall.json" in str(paths.diagnostics_dir / "train_blocking_recall.json")
    assert (paths.diagnostics_dir / "train_blocking_recall.json").exists()


def _write_tsv(path: Path, header, rows):
    path.write_text("\t".join(header) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n", encoding="utf-8")


def test_blocking_recalls_exact_positive_pairs(tmp_path):
    from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
    from business_entity_resolution.blocking.evaluator import BlockingRecallEvaluator
    import sqlite3

    s1 = tmp_path / "train_source1.tsv"
    s2 = tmp_path / "train_source2.tsv"
    s3 = tmp_path / "train_source3.tsv"
    gt = tmp_path / "train_ground_truth.tsv"
    _write_tsv(s1, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "Acme Corporation", "12 Main Street 411001", "India"),
        ("S1_2", "No Match LLC", "99 Unknown Road 411999", "India"),
    ])
    _write_tsv(s2, ["entity_id", "business_name", "business_address", "country"], [
        ("S2_1", "ACME Corp", "12 Main St 411001", "IN"),
        ("S2_2", "Other Company", "77 Other Rd 411002", "IN"),
    ])
    _write_tsv(s3, ["entity_id", "business_name", "business_address", "country"], [
        ("S3_1", "Acme Corporation", "12 Main Street 411001", "India"),
    ])
    _write_tsv(gt, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1,S3_1"),
        ("S1_2", ""),
    ])

    db = tmp_path / "blocking.sqlite"
    engine = BlockingEngine(db, BlockingConfig(max_bucket_size=50))
    engine.initialize("train")
    engine.build_target_index({"S2": s2, "S3": s3})
    stats = engine.generate_candidates(s1, "train")
    assert stats.candidate_pairs >= 2

    report = BlockingRecallEvaluator(db).evaluate(gt)
    assert report.total_ground_truth_pairs == 2
    assert report.covered_ground_truth_pairs == 2
    assert report.recall == 1.0
    assert report.zero_match_s1_rows == 1


def test_bucket_cap_is_enforced(tmp_path):
    from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
    import sqlite3

    s1 = tmp_path / "s1.tsv"
    s2 = tmp_path / "s2.tsv"
    _write_tsv(s1, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "Same Name", "1 A Street", "India"),
    ])
    rows = [(f"S2_{i}", "Same Name", f"{i} A Street", "India") for i in range(1, 6)]
    _write_tsv(s2, ["entity_id", "business_name", "business_address", "country"], rows)

    db = tmp_path / "blocking.sqlite"
    engine = BlockingEngine(db, BlockingConfig(max_bucket_size=2))
    engine.initialize("train")
    engine.build_target_index({"S2": s2})
    stats = engine.generate_candidates(s1, "train")
    assert stats.capped_bucket_keys > 0

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] <= 1


if __name__ == "__main__":
    print("Running Milestone 4 blocking unit tests...")
    test_multi_pass_blocking_keys()
    print("  [1/9] test_multi_pass_blocking_keys passed.")
    test_exact_matches_strategy()
    print("  [2/9] test_exact_matches_strategy passed.")
    test_ngram_strategies()
    print("  [3/9] test_ngram_strategies passed.")
    test_hn_postal_and_domain_alias()
    print("  [4/9] test_hn_postal_and_domain_alias passed.")
    test_sqlite_index_creation_and_insertion()
    print("  [5/9] test_sqlite_index_creation_and_insertion passed.")
    test_bucket_capping()
    print("  [6/9] test_bucket_capping passed.")
    test_candidate_deduplication_and_provenance()
    print("  [7/9] test_candidate_deduplication_and_provenance passed.")
    test_zero_match_s1_handling()
    print("  [8/9] test_zero_match_s1_handling passed.")
    test_recall_and_reduction_ratio_evaluation()
    print("  [9/9] test_recall_and_reduction_ratio_evaluation passed.")
    print("ALL 9 MILESTONE 4 BLOCKING TESTS PASSED!")

