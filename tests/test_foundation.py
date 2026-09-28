"""
Unit tests for Milestone 1 foundation.
"""

import sys
from pathlib import Path

# Add src to pythonpath
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.logging_config import setup_logger


def test_paths():
    paths = ProjectPaths()
    paths.ensure_directories()
    assert paths.output_dir.exists()
    assert paths.artifacts_dir.exists()
    assert paths.diagnostics_dir.exists()
    
    verified = paths.verify_inputs()
    print("Paths verified:", verified)
    assert verified["train_source1"], "train_source1.tsv not found!"
    assert verified["test_source1"], "test_source1.tsv not found!"


def test_config():
    config = PipelineConfig(sample_size=1000)
    assert config.sample_size == 1000
    assert config.blocking.adaptive_k == 8
    
    paths = ProjectPaths()
    cfg_path = paths.diagnostics_dir / "test_config.json"
    config.save_json(cfg_path)
    assert cfg_path.exists()
    
    loaded = PipelineConfig.from_json(cfg_path)
    assert loaded.sample_size == 1000
    assert loaded.blocking.adaptive_k == 8
    cfg_path.unlink()


def test_logger():
    paths = ProjectPaths()
    log_file = paths.logs_dir / "test.log"
    logger = setup_logger("TestLogger", log_file=log_file)
    logger.info("Foundation test log message.")
    assert log_file.exists()
    # Close handlers to release Windows file lock
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    log_file.unlink()


if __name__ == "__main__":
    print("Running foundation tests...")
    test_paths()
    test_config()
    test_logger()
    print("ALL FOUNDATION TESTS PASSED!")
