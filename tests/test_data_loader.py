"""
Unit tests for Milestone 3 - Streaming Data Loader.
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.io import (
    EntityRecord,
    stream_records,
    stream_chunks,
    load_ground_truth,
    stream_ground_truth_pairs,
)


def test_entity_record():
    rec1 = EntityRecord("S1-001", "Apple Store", "1 Infinite Loop", "US")
    assert rec1.source == "S1"
    assert rec1.entity_id == "S1-001"
    assert rec1.country == "US"

    rec2 = EntityRecord("S2-099", "Le Bistro", "10 Rue de Paris", "France")
    assert rec2.source == "S2"
    assert rec2.business_name == "Le Bistro"


def test_stream_records():
    paths = ProjectPaths()
    records = list(stream_records(paths.train_source1, max_rows=50))
    assert len(records) == 50
    for r in records:
        assert r.source == "S1"
        assert r.country in {"US", "India"}


def test_stream_country_filtered():
    paths = ProjectPaths()
    # Test streaming only France from test_source1
    records_france = list(stream_records(paths.test_source1, country="France", max_rows=50))
    assert len(records_france) == 50
    for r in records_france:
        assert r.country == "France"
        assert r.source == "S1"


def test_stream_chunks():
    paths = ProjectPaths()
    chunks = list(stream_chunks(paths.train_source1, chunk_size=20, max_rows=50))
    assert len(chunks) == 3  # 20 + 20 + 10
    assert len(chunks[0]) == 20
    assert len(chunks[1]) == 20
    assert len(chunks[2]) == 10


def test_load_ground_truth():
    paths = ProjectPaths()
    s1_map, target_map = load_ground_truth(paths.train_ground_truth, max_rows=100)
    assert len(s1_map) == 100
    for s1_id, targets in s1_map.items():
        assert s1_id.startswith("S1-")
        for t in targets:
            assert target_map[t] == s1_id
            assert t.startswith(("S2-", "S3-"))


def test_stream_ground_truth_pairs():
    paths = ProjectPaths()
    pairs = list(stream_ground_truth_pairs(paths.train_ground_truth, max_rows=50))
    assert len(pairs) > 0
    for s1, target in pairs:
        assert s1.startswith("S1-")
        assert target.startswith(("S2-", "S3-"))


if __name__ == "__main__":
    print("Running streaming data loader tests...")
    test_entity_record()
    test_stream_records()
    test_stream_country_filtered()
    test_stream_chunks()
    test_load_ground_truth()
    test_stream_ground_truth_pairs()
    print("ALL STREAMING DATA LOADER TESTS PASSED!")
