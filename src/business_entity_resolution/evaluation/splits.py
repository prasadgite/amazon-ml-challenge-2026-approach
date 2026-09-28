"""
Deterministic, entity-level dataset splitting for Industrial Entity Resolution.
Ensures zero S1 leakage across train, validation, and holdout partitions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class PartitionConfig:
    seed: int = 42
    holdout_fraction: float = 0.15
    n_folds: int = 5


@dataclass
class DatasetSplits:
    config: PartitionConfig
    holdout_s1: set[str]
    development_s1: set[str]
    folds: dict[int, set[str]]  # fold_idx -> set of S1 IDs
    assignments: dict[str, tuple[str, int | None]]

    @classmethod
    def create(
        cls,
        s1_ids: set[str],
        config: PartitionConfig | None = None,
    ) -> DatasetSplits:
        cfg = config or PartitionConfig()
        holdout_s1: set[str] = set()
        development_s1: set[str] = set()
        folds: dict[int, set[str]] = {i: set() for i in range(cfg.n_folds)}
        assignments: dict[str, tuple[str, int | None]] = {}

        for s1 in sorted(s1_ids):
            raw = f"{cfg.seed}:{s1}".encode("utf-8")
            digest = hashlib.blake2b(raw, digest_size=8).digest()
            h = int.from_bytes(digest, "big") / float(2**64)

            if h < cfg.holdout_fraction:
                holdout_s1.add(s1)
                assignments[s1] = ("holdout", None)
            else:
                development_s1.add(s1)
                dev_h = (h - cfg.holdout_fraction) / (1.0 - cfg.holdout_fraction)
                fold = min(cfg.n_folds - 1, int(dev_h * cfg.n_folds))
                folds[fold].add(s1)
                assignments[s1] = ("development", fold)

        # Invariant Assertions
        assert holdout_s1.isdisjoint(development_s1), "Leakage detected: S1 in both holdout and development!"
        assert (holdout_s1 | development_s1) == s1_ids, "Integrity failure: Not all S1 entities assigned!"

        return cls(
            config=cfg,
            holdout_s1=holdout_s1,
            development_s1=development_s1,
            folds=folds,
            assignments=assignments,
        )

    def train_s1_for_fold(self, fold_idx: int) -> set[str]:
        """Return the training S1 IDs for a given OOF fold (all other dev folds)."""
        train_s1 = set()
        for idx, s1_set in self.folds.items():
            if idx != fold_idx:
                train_s1.update(s1_set)
        return train_s1

    def val_s1_for_fold(self, fold_idx: int) -> set[str]:
        """Return the validation S1 IDs for a given OOF fold."""
        return self.folds[fold_idx]
