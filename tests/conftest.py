import pytest
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
import sys
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.paths import ProjectPaths

def pytest_collection_modifyitems(config, items):
    paths = ProjectPaths()
    dataset_missing = not paths.train_source1.is_file()
    if dataset_missing:
        skip_dataset = pytest.mark.skip(reason="Large dataset TSV files not present (run with full dataset mounted)")
        dataset_test_names = {
            "test_stream_records",
            "test_stream_country_filtered",
            "test_stream_chunks",
            "test_load_ground_truth",
            "test_stream_ground_truth_pairs",
            "test_paths",
            "test_dataset_profiler_sample",
            "test_stage_verify",
            "test_cli_verify_subprocess",
            "test_recall_and_reduction_ratio_evaluation",
        }
        for item in items:
            if item.name in dataset_test_names:
                item.add_marker(skip_dataset)
