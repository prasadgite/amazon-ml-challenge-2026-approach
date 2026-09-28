"""
Unit and integration tests for Milestone 9.2:
1. Alias / DBA blocking
2. Cross-script transliteration (Devanagari, Bengali, Punjabi to Latin)
3. Scoped country fallback
4. Smart capping with specific overflow sub-keys
5. Strategy-level recall breakdown
6. FinalEvaluator evaluation cohort resolution (preventing synthetic singletons)
"""

import csv
import sqlite3
from pathlib import Path
import pytest

from business_entity_resolution.normalization.normalizer import TextNormalizer, NormalizedRecord, _transliterate_text
from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
from business_entity_resolution.blocking.evaluator import BlockingRecallEvaluator
from business_entity_resolution.evaluation.final_evaluator import FinalEvaluationConfig, FinalEvaluator


def _write_tsv(path: Path, header: list[str], rows: list[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def test_transliteration_indic_scripts():
    """Verify Brahmic Indic scripts (Devanagari, Bengali, Punjabi) transliterate to Latin phonetics."""
    devanagari = "लोटस मीडिया प्राइवेट लिमिटेड"
    translit_deva = _transliterate_text(devanagari)
    assert "lots" in translit_deva
    assert "meediyaa" in translit_deva
    assert "limited" in translit_deva

    bengali = "জয় প্রডিউসার প্রাইভেট লিমিটেড"
    translit_beng = _transliterate_text(bengali)
    assert "praaibhet" in translit_beng or "limited" in translit_beng

    punjabi = "ਰਾਜ ਵਨ ਟੈਕਨਾਲੋਜੀ ਪ੍ਰਾ. ਲਿ."
    translit_punj = _transliterate_text(punjabi)
    assert "raaj" in translit_punj


def test_dba_alias_extraction_and_blocking(tmp_path):
    """Verify that Beloavi d/b/a Novent Owl PLLC generates alias keys and matches Novent Owl PLLC."""
    normalizer = TextNormalizer()
    rec = normalizer.normalize("S1_1", "Beloavi d/b/a Novent Owl PLLC", "100 Broadway", "US")
    assert "novent owl" in rec.name_aliases or any("novent owl" in a for a in rec.name_aliases)

    s1_file = tmp_path / "s1.tsv"
    s2_file = tmp_path / "s2.tsv"
    s3_file = tmp_path / "s3.tsv"
    gt_file = tmp_path / "gt.tsv"

    _write_tsv(s1_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "Beloavi d/b/a Novent Owl PLLC", "100 Broadway", "US"),
    ])
    _write_tsv(s2_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S2_1", "Novent Owl PLLC", "100 Broadway", "US"),
    ])
    _write_tsv(s3_file, ["entity_id", "business_name", "business_address", "country"], [])
    _write_tsv(gt_file, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1"),
    ])

    db_path = tmp_path / "blocking.sqlite"
    engine = BlockingEngine(db_path, BlockingConfig(enable_alias_blocking=True))
    engine.initialize("train")
    engine.build_target_index({"S2": s2_file, "S3": s3_file})
    stats = engine.generate_candidates(s1_file, "train")

    evaluator = BlockingRecallEvaluator(db_path)
    report = evaluator.evaluate(gt_file)
    assert report.covered_ground_truth_pairs == 1
    assert report.recall == 1.0


def test_cross_script_transliteration_blocking(tmp_path):
    """Verify Latin Lotus Media in S1 matches Devanagari in S2 through transliteration keys."""
    s1_file = tmp_path / "s1.tsv"
    s2_file = tmp_path / "s2.tsv"
    s3_file = tmp_path / "s3.tsv"
    gt_file = tmp_path / "gt.tsv"

    _write_tsv(s1_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "Lotus Media Private Limited", "MG Road Bangalore", "India"),
    ])
    _write_tsv(s2_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S2_1", "लोटस मीडिया प्राइवेट लिमिटेड", "Park Street Kolkata", "India"),
    ])
    _write_tsv(s3_file, ["entity_id", "business_name", "business_address", "country"], [])
    _write_tsv(gt_file, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1"),
    ])

    db_path = tmp_path / "blocking.sqlite"
    engine = BlockingEngine(db_path, BlockingConfig(enable_transliteration=True))
    engine.initialize("train")
    engine.build_target_index({"S2": s2_file, "S3": s3_file})
    engine.generate_candidates(s1_file, "train")

    evaluator = BlockingRecallEvaluator(db_path)
    report = evaluator.evaluate(gt_file)
    assert report.covered_ground_truth_pairs == 1
    assert "translit_exact_country" in report.strategy_recovered_counts or "translit_prefix_country" in report.strategy_recovered_counts


def test_scoped_country_fallback(tmp_path):
    """Verify that records with missing country can match via scoped countryless fallback."""
    s1_file = tmp_path / "s1.tsv"
    s2_file = tmp_path / "s2.tsv"
    s3_file = tmp_path / "s3.tsv"
    gt_file = tmp_path / "gt.tsv"

    _write_tsv(s1_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "Unique Global Tech Solutions", "100 Market St", ""),
    ])
    _write_tsv(s2_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S2_1", "Unique Global Tech Solutions", "100 Market St", ""),
    ])
    _write_tsv(s3_file, ["entity_id", "business_name", "business_address", "country"], [])
    _write_tsv(gt_file, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1"),
    ])

    db_path = tmp_path / "blocking.sqlite"
    engine = BlockingEngine(db_path, BlockingConfig(enable_country_fallback=True))
    engine.initialize("train")
    engine.build_target_index({"S2": s2_file, "S3": s3_file})
    engine.generate_candidates(s1_file, "train")

    evaluator = BlockingRecallEvaluator(db_path)
    report = evaluator.evaluate(gt_file)
    assert report.covered_ground_truth_pairs == 1


def test_smart_capping_recovers_matches(tmp_path):
    """Verify that when a generic prefix bucket exceeds max_bucket_size, smart overflow specificity recovers the match."""
    s1_file = tmp_path / "s1.tsv"
    s2_file = tmp_path / "s2.tsv"
    s3_file = tmp_path / "s3.tsv"
    gt_file = tmp_path / "gt.tsv"

    # S1 has name prefix 'amer' and street prefix 'main'
    _write_tsv(s1_file, ["entity_id", "business_name", "business_address", "country"], [
        ("S1_1", "American Logistics LLC", "500 Main Street 90210", "US"),
    ])

    # Create 5 targets with 'American ...' so bucket_size=5 exceeds max_bucket_size=2
    # Only one target has Main Street
    s2_rows = [
        ("S2_1", "American Logistics LLC", "500 Main Street 90210", "US"),
        ("S2_2", "American Bakery Corp", "10 Elm Street 90211", "US"),
        ("S2_3", "American Dental Care", "20 Oak Street 90212", "US"),
        ("S2_4", "American Fitness Club", "30 Pine Street 90213", "US"),
    ]
    _write_tsv(s2_file, ["entity_id", "business_name", "business_address", "country"], s2_rows)
    _write_tsv(s3_file, ["entity_id", "business_name", "business_address", "country"], [])
    _write_tsv(gt_file, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1"),
    ])

    db_path = tmp_path / "blocking.sqlite"
    # Set cap to 2 so generic name_prefix ('amer') is capped
    engine = BlockingEngine(db_path, BlockingConfig(max_bucket_size=2, enable_smart_capping=True))
    engine.initialize("train")
    engine.build_target_index({"S2": s2_file, "S3": s3_file})
    engine.generate_candidates(s1_file, "train")

    evaluator = BlockingRecallEvaluator(db_path)
    report = evaluator.evaluate(gt_file)
    assert report.covered_ground_truth_pairs == 1


def test_final_evaluator_cohort_resolution(tmp_path):
    """Verify that FinalEvaluator with evaluation_s1_source='feature' resolves cohort strictly to feature S1s."""
    feature_file = tmp_path / "features.tsv"
    source1_file = tmp_path / "s1_full.tsv"
    gt_file = tmp_path / "gt.tsv"

    # Feature file only has S1_1 and S1_2
    _write_tsv(feature_file, [
        "source1_id", "target_source", "target_id", "label", "raw_score", "calibrated_score",
        "strong_evidence_count", "country_conflict", "name_jaro_winkler", "name_token_jaccard",
        "address_token_jaccard", "domain_exact", "alias_exact", "blocking_hit_count"
    ], [
        ("S1_1", "S2", "S2_1", "1", "0.95", "0.95", "1", "0", "1.0", "1.0", "0.8", "0", "0", "1"),
        ("S1_2", "S2", "S2_2", "0", "0.01", "0.01", "0", "0", "0.2", "0.0", "0.1", "0", "0", "1"),
    ])

    # Source 1 full has 100 entities
    s1_rows = [(f"S1_{i}", f"Company {i}", f"{i} Street", "US") for i in range(1, 101)]
    _write_tsv(source1_file, ["entity_id", "business_name", "business_address", "country"], s1_rows)

    _write_tsv(gt_file, ["source1_entity_id", "matched_entity_ids"], [
        ("S1_1", "S2_1"),
    ])

    config = FinalEvaluationConfig(
        n_folds=2,
        holdout_fraction=0.2,
        evaluation_s1_source="feature",
    )
    evaluator = FinalEvaluator(config)
    s1_ids = evaluator._get_evaluated_s1_ids(feature_file, source1_file)
    assert set(s1_ids) == {"S1_1", "S1_2"}
    assert len(s1_ids) == 2  # NOT 100!
