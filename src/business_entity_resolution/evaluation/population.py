"""
Evaluation population control and integrity validation for Industrial Entity Resolution.
Guarantees feature population matches evaluation population and singletons are cleanly accounted for.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class PopulationMismatchError(ValueError):
    """Raised when the feature population diverges from the evaluation population."""
    pass


@dataclass
class EvaluationPopulation:
    """Explicitly defined population of Source 1 entities for an experiment."""
    population_id: str
    s1_ids: set[str]
    truth: dict[str, set[str]]
    singletons: set[str] = field(init=False)

    def __post_init__(self):
        self.singletons = {s1 for s1 in self.s1_ids if len(self.truth.get(s1, set())) == 0}

    @property
    def total_s1(self) -> int:
        return len(self.s1_ids)

    @property
    def total_singletons(self) -> int:
        return len(self.singletons)

    @property
    def total_matched_entities(self) -> int:
        return self.total_s1 - self.total_singletons

    @property
    def total_ground_truth_pairs(self) -> int:
        return sum(len(self.truth.get(s1, set())) for s1 in self.s1_ids)

    @classmethod
    def load_ground_truth(cls, truth_path: Path) -> dict[str, set[str]]:
        truth: dict[str, set[str]] = {}
        with truth_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s1_id = row["source1_entity_id"].strip()
                matches = [m.strip() for m in row.get("matched_entity_ids", "").split(",") if m.strip()]
                truth[s1_id] = set(matches)
        return truth

    @classmethod
    def from_feature_file(
        cls,
        feature_path: Path,
        truth_path: Path,
        population_id: str | None = None,
    ) -> EvaluationPopulation:
        """Derive the evaluation population strictly from S1 IDs represented in the feature file."""
        s1_ids: set[str] = set()
        with feature_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s1_id = row.get("source1_id", "").strip()
                if s1_id:
                    s1_ids.add(s1_id)

        truth = cls.load_ground_truth(truth_path)
        pop_id = population_id or f"feat_{feature_path.stem}_{len(s1_ids)}"
        return cls(population_id=pop_id, s1_ids=s1_ids, truth=truth)

    @classmethod
    def from_source1_file(
        cls,
        source1_path: Path,
        truth_path: Path,
        population_id: str | None = None,
        max_entities: int = 0,
    ) -> EvaluationPopulation:
        """Load evaluation population from a dedicated Source 1 TSV file."""
        s1_ids: set[str] = set()
        with source1_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s1_id = row.get("entity_id", "").strip()
                if s1_id:
                    s1_ids.add(s1_id)
                if max_entities > 0 and len(s1_ids) >= max_entities:
                    break

        truth = cls.load_ground_truth(truth_path)
        pop_id = population_id or f"s1_{source1_path.stem}_{len(s1_ids)}"
        return cls(population_id=pop_id, s1_ids=s1_ids, truth=truth)

    def validate_feature_population(self, feature_s1_ids: set[str]) -> None:
        """Enforce the hard invariant: feature_population == evaluation_population."""
        missing_in_features = self.s1_ids - feature_s1_ids
        extra_in_features = feature_s1_ids - self.s1_ids

        # True singletons are legitimately permitted to have 0 candidate pairs.
        # But non-singletons missing from features indicates a severe cohort leak.
        unmatched_non_singletons = missing_in_features - self.singletons

        if unmatched_non_singletons:
            sample_ids = sorted(list(unmatched_non_singletons))[:5]
            raise PopulationMismatchError(
                f"Feature population does not match evaluation population '{self.population_id}'! "
                f"{len(unmatched_non_singletons):,} non-singleton S1 entities have zero features. "
                f"Examples: {sample_ids}. "
                f"Evaluating against mismatched populations causes synthetic false negatives."
            )

    def split_assignment(
        self,
        seed: int = 42,
        holdout_fraction: float = 0.15,
        n_folds: int = 5,
    ) -> dict[str, tuple[str, int | None]]:
        """Assign every S1 entity deterministically to either holdout or an OOF development fold."""
        assignments: dict[str, tuple[str, int | None]] = {}
        for s1 in sorted(self.s1_ids):
            raw = f"{seed}:{s1}".encode("utf-8")
            digest = hashlib.blake2b(raw, digest_size=8).digest()
            h = int.from_bytes(digest, "big") / float(2**64)

            if h < holdout_fraction:
                assignments[s1] = ("holdout", None)
            else:
                dev_h = (h - holdout_fraction) / (1.0 - holdout_fraction)
                fold = min(n_folds - 1, int(dev_h * n_folds))
                assignments[s1] = ("development", fold)
        return assignments
