"""
Unit tests for run_pipeline.py CLI and stage execution.
"""

import sys
import subprocess
from pathlib import Path

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from business_entity_resolution.paths import ProjectPaths  # noqa: E402
from business_entity_resolution.logging_config import setup_logger  # noqa: E402
from run_pipeline import stage_verify, stage_validate  # noqa: E402


def test_stage_verify():
    paths = ProjectPaths()
    logger = setup_logger("TestPipelineCLI")
    success = stage_verify(paths, logger)
    assert success is True, "stage_verify failed!"


def test_stage_validate_missing_file():
    paths = ProjectPaths()
    logger = setup_logger("TestPipelineCLI")
    # If matching_results.tsv doesn't exist yet, stage_validate must return False
    if not paths.matching_results.is_file():
        success = stage_validate(paths, logger)
        assert success is False, "stage_validate should return False when matching_results.tsv is missing!"


def test_cli_verify_subprocess():
    cmd = [sys.executable, str(root_dir / "run_pipeline.py"), "--stage", "verify"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"run_pipeline.py --stage verify failed: {res.stderr}"
    assert "Environment & Dataset Verification" in res.stdout or "Environment & Dataset Verification" in res.stderr


def test_cli_validate_missing_subprocess():
    paths = ProjectPaths()
    if not paths.matching_results.is_file():
        cmd = [sys.executable, str(root_dir / "run_pipeline.py"), "--stage", "validate"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 1, "run_pipeline.py --stage validate should exit with code 1 when output is missing!"


def test_cli_config_not_found():
    cmd = [sys.executable, str(root_dir / "run_pipeline.py"), "--config", "nonexistent_config_file_123.json"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1, "run_pipeline.py --config should exit with code 1 when config file does not exist!"
    assert "Configuration file not found" in res.stdout or "Configuration file not found" in res.stderr


def test_cli_features_requires_blocking_db(tmp_path, monkeypatch):
    from run_pipeline import run_stage_pairwise_features
    from business_entity_resolution.config import PipelineConfig
    paths = ProjectPaths(tmp_path)
    logger = setup_logger("TestPipelineCLI")
    success = run_stage_pairwise_features(paths, PipelineConfig(), logger)
    assert success is False, "run_stage_pairwise_features should fail when blocking db is absent"


if __name__ == "__main__":
    print("Running run_pipeline unit tests...")
    test_stage_verify()
    test_stage_validate_missing_file()
    test_cli_verify_subprocess()
    test_cli_validate_missing_subprocess()
    test_cli_config_not_found()
    print("ALL RUN_PIPELINE TESTS PASSED!")

