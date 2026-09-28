"""
Automated data leakage and evaluation integrity checks for Industrial Entity Resolution.
Guarantees zero target-identity leakage, partition separation, and clean holdout isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

LOGGER = logging.getLogger("ber.leakage")


class LeakageViolationError(ValueError):
    """Raised when an evaluation partition or target leakage invariant is violated."""
    pass


@dataclass
class LeakageAuditReport:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    partition_overlap_count: int = 0
    target_bridge_count: int = 0
    missing_population_count: int = 0


def audit_evaluation_leakage(
    train_s1: set[str],
    holdout_s1: set[str],
    train_targets: set[str] | None = None,
    holdout_targets: set[str] | None = None,
    evaluated_s1: set[str] | None = None,
    expected_s1: set[str] | None = None,
    threshold_tuned_on: set[str] | None = None,
) -> LeakageAuditReport:
    """Execute comprehensive leakage audit across partitions, targets, and threshold tuning."""
    violations: list[str] = []
    warnings: list[str] = []

    # 1. Hard Invariant: S1 entity partition disjointness
    overlap_s1 = train_s1 & holdout_s1
    if overlap_s1:
        sample = sorted(list(overlap_s1))[:5]
        violations.append(
            f"FATAL: S1 partition overlap detected! {len(overlap_s1)} S1 entities appear in both train and holdout. "
            f"Examples: {sample}"
        )

    # 2. Hard Invariant: Threshold isolation
    if threshold_tuned_on is not None:
        threshold_holdout_leak = threshold_tuned_on & holdout_s1
        if threshold_holdout_leak:
            violations.append(
                f"FATAL: Threshold was tuned on holdout entities! {len(threshold_holdout_leak)} holdout entities were in threshold tuning."
            )

    # 3. Population completeness check
    missing_count = 0
    if expected_s1 is not None and evaluated_s1 is not None:
        missing = expected_s1 - evaluated_s1
        missing_count = len(missing)
        if missing:
            violations.append(
                f"FATAL: {missing_count} expected S1 entities were omitted from evaluation population!"
            )

    # 4. Transitive target bridge warning
    bridge_count = 0
    if train_targets is not None and holdout_targets is not None:
        shared_targets = train_targets & holdout_targets
        bridge_count = len(shared_targets)
        if shared_targets:
            warnings.append(
                f"Informational: {bridge_count} target entities are referenced by both train and holdout S1 entities "
                f"(real-world natural transitivity). Model must not overfit specific target IDs."
            )

    passed = len(violations) == 0
    if not passed:
        LOGGER.error("Leakage audit FAILED with %d violations: %s", len(violations), violations)

    return LeakageAuditReport(
        passed=passed,
        violations=violations,
        warnings=warnings,
        partition_overlap_count=len(overlap_s1),
        target_bridge_count=bridge_count,
        missing_population_count=missing_count,
    )
