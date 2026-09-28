"""
Path resolution and directory management for Business Entity Resolution.
"""

from pathlib import Path
import os
from typing import Dict


REQUIRED_SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
REQUIRED_GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")


def assert_tsv_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file but found: {path}")


def find_project_root() -> Path:
    """Resolve project root by inspecting current working directory or anchor files."""
    cwd = Path.cwd().resolve()
    # Check if run from project root
    if (cwd / "dataset").exists() or (cwd / "src").exists():
        return cwd
    # Check parent directories
    for parent in cwd.parents:
        if (parent / "dataset").exists() or (parent / "src").exists():
            return parent
    return cwd


PROJECT_ROOT = find_project_root()



def resolve_dataset_dirs(root: Path) -> Dict[str, Path]:
    """
    Locate train and test directories, checking both standard structure
    and student_resource structure.
    """
    env_dataset = os.environ.get("BER_DATASET_DIR")
    candidates_train = []
    candidates_test = []
    candidates_utils = [
        root / "utils",
        root / "dataset" / "student_resource" / "utils",
    ]

    if env_dataset:
        env_path = Path(env_dataset).resolve()
        if (env_path / "train").is_dir():
            candidates_train.append(env_path / "train")
        elif env_path.is_dir():
            candidates_train.append(env_path)

        if (env_path / "test").is_dir():
            candidates_test.append(env_path / "test")
        elif env_path.is_dir():
            candidates_test.append(env_path)

        if (env_path.parent / "utils").is_dir():
            candidates_utils.insert(0, env_path.parent / "utils")

    candidates_train.extend([
        root / "dataset" / "train",
        root / "dataset" / "student_resource" / "dataset" / "train",
    ])
    candidates_test.extend([
        root / "dataset" / "test",
        root / "dataset" / "student_resource" / "dataset" / "test",
    ])

    train_dir = next((p for p in candidates_train if p.exists() and p.is_dir()), candidates_train[0])
    test_dir = next((p for p in candidates_test if p.exists() and p.is_dir()), candidates_test[0])
    utils_dir = next((p for p in candidates_utils if p.exists() and p.is_dir()), candidates_utils[0])

    return {
        "train_dir": train_dir,
        "test_dir": test_dir,
        "utils_dir": utils_dir,
    }


class ProjectPaths:
    """Manages all directories and file locations in the pipeline."""

    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root.resolve()
        
        # Dataset dirs
        ds_dirs = resolve_dataset_dirs(self.root)
        self.train_dir = ds_dirs["train_dir"]
        self.test_dir = ds_dirs["test_dir"]
        self.utils_dir = ds_dirs["utils_dir"]

        # Output and Artifact directories
        self.output_dir = self.root / "output"
        self.artifacts_dir = self.root / "artifacts"
        self.indexes_dir = self.artifacts_dir / "indexes"
        self.models_dir = self.artifacts_dir / "models"
        self.features_dir = self.artifacts_dir / "features"
        self.diagnostics_dir = self.artifacts_dir / "diagnostics"
        self.predictions_dir = self.artifacts_dir / "predictions"
        self.logs_dir = self.artifacts_dir / "logs"
        self.scratch_dir = self.artifacts_dir / "scratch"
        self.temp_dir = self.scratch_dir / "temp"

        # Specific file paths
        self.train_source1 = self.train_dir / "train_source1.tsv"
        self.train_source2 = self.train_dir / "train_source2.tsv"
        self.train_source3 = self.train_dir / "train_source3.tsv"
        self.train_ground_truth = self.train_dir / "train_ground_truth.tsv"

        self.test_source1 = self.test_dir / "test_source1.tsv"
        self.test_source2 = self.test_dir / "test_source2.tsv"
        self.test_source3 = self.test_dir / "test_source3.tsv"

        self.matching_results = self.output_dir / "matching_results.tsv"
        self.candidate_pairs = self.output_dir / "candidate_pairs.tsv"
        self.validator_script = self.utils_dir / "validate_submission.py"

    def ensure_directories(self) -> None:
        """Create all required output and artifact directories if they do not exist."""
        for d in [
            self.output_dir,
            self.artifacts_dir,
            self.indexes_dir,
            self.models_dir,
            self.features_dir,
            self.diagnostics_dir,
            self.predictions_dir,
            self.logs_dir,
            self.scratch_dir,
            self.temp_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Force all libraries and subprocesses to use D: drive temp directory
        os.environ["TEMP"] = str(self.temp_dir)
        os.environ["TMP"] = str(self.temp_dir)
        os.environ["TMPDIR"] = str(self.temp_dir)

    def verify_inputs(self) -> Dict[str, bool]:
        """Verify presence of core input files."""
        return {
            "train_source1": self.train_source1.is_file(),
            "train_source2": self.train_source2.is_file(),
            "train_source3": self.train_source3.is_file(),
            "train_ground_truth": self.train_ground_truth.is_file(),
            "test_source1": self.test_source1.is_file(),
            "test_source2": self.test_source2.is_file(),
            "test_source3": self.test_source3.is_file(),
            "validator_script": self.validator_script.is_file(),
        }
