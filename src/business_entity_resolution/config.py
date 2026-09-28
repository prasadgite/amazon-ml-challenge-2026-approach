"""
Configuration dataclasses and parameter definitions.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json

from business_entity_resolution.paths import ProjectPaths


@dataclass
class BlockingConfig:
    """Parameters controlling candidate generation and blocking."""
    common_token_df_threshold: float = 0.005  # Suppress tokens appearing in > 0.5% records
    min_name_token_len: int = 3
    ngram_size: int = 3
    initial_top_k: int = 25  # Raw candidate cap before cheap scoring
    adaptive_k: int = 8      # Default target candidate budget per S1
    min_cheap_score: float = 2.0


@dataclass
class ModelConfig:
    """Parameters controlling model training and inference."""
    model_type: str = "lightgbm"  # 'lightgbm' or 'hist_gradient_boosting'
    n_estimators: int = 250
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 20
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    random_state: int = 42


@dataclass
class DecisionConfig:
    """Parameters controlling match set resolution and singleton protection."""
    base_threshold: float = 0.75
    singleton_margin: float = 0.15
    soft_conflict_resolution: bool = True
    beta: float = 0.5  # Official competition beta


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire entity resolution pipeline."""
    sample_size: int = 0  # 0 = Full dataset; >0 = Development mode slice
    chunk_size: int = 100_000
    random_seed: int = 42
    n_jobs: int = -1

    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Serialize configuration to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        blocking_data = data.pop("blocking", {})
        model_data = data.pop("model", {})
        decision_data = data.pop("decision", {})
        return cls(
            blocking=BlockingConfig(**blocking_data),
            model=ModelConfig(**model_data),
            decision=DecisionConfig(**decision_data),
            **data,
        )

    @classmethod
    def from_root(cls, root: Optional[Path] = None) -> "PipelineConfig":
        return cls()

    @property
    def paths(self) -> ProjectPaths:
        return ProjectPaths()

    @property
    def project_root(self) -> Path:
        return self.paths.root

    @property
    def dataset_dir(self) -> Path:
        return self.paths.train_dir.parent

    @property
    def train_dir(self) -> Path:
        return self.paths.train_dir

    @property
    def test_dir(self) -> Path:
        return self.paths.test_dir

    @property
    def output_dir(self) -> Path:
        return self.paths.output_dir

    @property
    def artifact_dir(self) -> Path:
        return self.paths.artifacts_dir

    @property
    def log_dir(self) -> Path:
        return self.paths.logs_dir

    @property
    def train_source1(self) -> Path:
        return self.paths.train_source1

    @property
    def train_source2(self) -> Path:
        return self.paths.train_source2

    @property
    def train_source3(self) -> Path:
        return self.paths.train_source3

    @property
    def train_ground_truth(self) -> Path:
        return self.paths.train_ground_truth

    @property
    def test_source1(self) -> Path:
        return self.paths.test_source1

    @property
    def test_source2(self) -> Path:
        return self.paths.test_source2

    @property
    def test_source3(self) -> Path:
        return self.paths.test_source3

    def ensure_directories(self) -> None:
        self.paths.ensure_directories()

    def required_train_files(self):
        return (self.train_source1, self.train_source2, self.train_source3, self.train_ground_truth)

    def required_test_files(self):
        return (self.test_source1, self.test_source2, self.test_source3)

