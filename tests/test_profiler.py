"""
Unit tests for Milestone 2 - Dataset Profiler.
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.profiling.stats import StringStats, RunningStats
from business_entity_resolution.profiling.dataset_profiler import DatasetProfiler


def test_running_stats():
    rs = RunningStats()
    for val in range(1, 101):
        rs.update(val)
    d = rs.to_dict()
    assert d["count"] == 100
    assert d["min"] == 1.0
    assert d["max"] == 100.0
    assert abs(d["mean"] - 50.5) < 0.1
    assert abs(d["p50"] - 50.0) < 3.0


def test_string_stats():
    ss = StringStats()
    ss.update("Apple Inc")
    ss.update("Google LLC Mountain View")
    ss.update(None)
    ss.update("   ")
    d = ss.to_dict()
    assert d["total_records"] == 4
    assert d["missing_count"] == 1
    assert d["empty_count"] == 1
    assert d["missing_pct"] == 25.0


def test_dataset_profiler_sample():
    paths = ProjectPaths()
    config = PipelineConfig(sample_size=100)
    profiler = DatasetProfiler(paths, config)
    res = profiler.profile_source_tsv(paths.train_source1, max_rows=100)
    assert res["total_rows"] == 100
    assert "US" in res["country_distribution"] or "India" in res["country_distribution"]
    assert res["business_name"]["total_records"] == 100


HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def build_fixture(tmp_path):
    train = tmp_path / "train"
    test = tmp_path / "test"
    train.mkdir()
    test.mkdir()

    (train / "train_source1.tsv").write_text(
        HEADER
        + "S1-1\tAlpha Pvt Ltd\t12 MG Road Pune\tIndia\n"
        + "S1-2\tBeta Corp\t34 Main St Paris\tFrance\n"
        + "S1-3\t\t\tIndia\n",
        encoding="utf-8",
    )
    (train / "train_source2.tsv").write_text(
        HEADER
        + "S2-1\tAlpha Private Limited\t12 M G Road Pune\tIndia\n"
        + "S2-2\tBeta Corporation\t34 Main Street Paris\tFrance\n",
        encoding="utf-8",
    )
    (train / "train_source3.tsv").write_text(
        HEADER + "S3-1\tAlpha\t12 MG Rd Pune\tIndia\n",
        encoding="utf-8",
    )
    (train / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-1\n"
        "S1-2\tS2-2\n"
        "S1-3\t\n",
        encoding="utf-8",
    )

    for n in (1, 2, 3):
        (test / f"test_source{n}.tsv").write_text(
            HEADER + f"S{n}-T1\tGamma\t1 Main\tFrance\n",
            encoding="utf-8",
        )
    return train, test


def test_profile_counts(tmp_path):
    from business_entity_resolution.profiling.profile import DatasetProfiler as StreamProfiler
    train, test = build_fixture(tmp_path)
    result = StreamProfiler().profile_dataset(train, test)

    assert result["train"]["sources"]["S1"]["rows"] == 3
    assert result["test"]["sources"]["S3"]["rows"] == 1

    gt = result["train"]["ground_truth"]
    assert gt["rows"] == 3
    assert gt["zero_match_s1"] == 1
    assert gt["singleton_s1"] == 1
    assert gt["total_match_links"] == 3
    assert gt["match_count_distribution"] == {"0": 1, "1": 1, "2": 1}


def test_exact_id_uniqueness(tmp_path):
    from business_entity_resolution.profiling.profile import DatasetProfiler as StreamProfiler
    train, test = build_fixture(tmp_path)
    path = train / "train_source1.tsv"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "S1-1\tDuplicate\t1 Main\tIndia\n",
        encoding="utf-8",
    )
    result = StreamProfiler().profile_dataset(
        train, test, exact_id_uniqueness=True
    )
    s1 = result["train"]["sources"]["S1"]
    assert s1["entity_ids_seen"] == 4
    assert s1["entity_ids_unique"] == 3
    assert s1["duplicate_entity_ids"] == 1


if __name__ == "__main__":
    print("Running profiler tests...")
    test_running_stats()
    test_string_stats()
    test_dataset_profiler_sample()
    print("ALL PROFILER TESTS PASSED!")

