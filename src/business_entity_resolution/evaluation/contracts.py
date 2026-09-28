"""
Data contracts, experiment manifests, and schema guarantees for Industrial Entity Resolution.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExperimentContract:
    """Immutable contract governing an entity resolution experiment."""
    experiment_id: str
    population_id: str
    code_version: str
    created_at: str
    config_hash: str
    config_payload: dict[str, Any]
    parent_experiment: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        population_id: str,
        config: dict[str, Any],
        parent_experiment: str | None = None,
        tags: list[str] | None = None,
        code_version: str = "industrial-er-v1.0",
    ) -> ExperimentContract:
        cfg_json = json.dumps(config, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:16]
        return cls(
            experiment_id=experiment_id,
            population_id=population_id,
            code_version=code_version,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            config_hash=cfg_hash,
            config_payload=config,
            parent_experiment=parent_experiment,
            tags=tags or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass
class StageResultContract:
    """Contract capturing inputs, outputs, invariants, and metrics for a pipeline stage."""
    stage_name: str
    contract: ExperimentContract
    input_paths: dict[str, str]
    output_paths: dict[str, str]
    invariants_passed: bool
    invariant_violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def assert_invariants(self) -> None:
        if not self.invariants_passed:
            violations_str = "; ".join(self.invariant_violations)
            raise ValueError(f"Stage '{self.stage_name}' failed invariant contract: {violations_str}")
