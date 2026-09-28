from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class S1PolicyConfig:
    """
    Configuration for S1-level adaptive decision policy.

    The frozen global threshold remains the primary acceptance rule.

    Rescue logic is only considered when no candidate survives the
    primary threshold.
    """

    global_threshold: float = 0.8625

    enable_argmax_rescue: bool = False

    rescue_min_score: float = 0.70

    rescue_min_margin: float = 0.05

    require_positive_margin: bool = True

    reject_country_conflict: bool = True

    require_strong_evidence_for_rescue: bool = False

    min_strong_evidence_count: int = 1

    min_positive_evidence_count: int = 1

    max_rescue_candidates_per_s1: int = 1


@dataclass(frozen=True)
class CandidateScore:
    source1_id: str
    target_source: str
    target_id: str
    calibrated_score: float
    raw_score: float = 0.0
    country_conflict: bool = False
    strong_evidence_count: int = 0
    combined_evidence_count: int = 0
    label: int | None = None

    @property
    def target_key(self) -> tuple[str, str]:
        return (
            self.target_source,
            self.target_id,
        )


@dataclass(frozen=True)
class S1Decision:
    source1_id: str
    target_source: str | None
    target_id: str | None
    decision: str
    reason: str
    calibrated_score: float | None
    margin: float | None
    rescue_used: bool


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_from_row(row: dict[str, Any]) -> CandidateScore:
    """
    Convert an evaluator TSV row into the normalized policy representation.

    Supports the current M9 decision/prediction column names and tolerates
    missing optional evidence fields.
    """

    source1_id = (
        row.get("source1_id")
        or row.get("source1_entity_id")
        or ""
    ).strip()

    target_source = (
        row.get("target_source")
        or ""
    ).strip()

    target_id = (
        row.get("target_id")
        or ""
    ).strip()

    return CandidateScore(
        source1_id=source1_id,
        target_source=target_source,
        target_id=target_id,
        calibrated_score=_as_float(
            row.get("calibrated_score"),
        ),
        raw_score=_as_float(
            row.get("raw_score"),
        ),
        country_conflict=_as_bool(
            row.get("country_conflict"),
        ),
        strong_evidence_count=_as_int(
            row.get("strong_evidence_count"),
        ),
        combined_evidence_count=(
            _as_int(row.get("combined_evidence_count"))
            if row.get("combined_evidence_count") not in (None, "")
            else (1 if _as_float(row.get("calibrated_score")) > 0 else 0)
        ),
        label=(
            _as_int(row["label"])
            if row.get("label") not in (None, "")
            else None
        ),
    )



def group_candidates(
    rows: Iterable[dict[str, Any]],
) -> dict[str, list[CandidateScore]]:
    grouped: dict[str, list[CandidateScore]] = {}

    for row in rows:
        candidate = candidate_from_row(row)

        if not candidate.source1_id:
            continue

        grouped.setdefault(
            candidate.source1_id,
            [],
        ).append(candidate)

    for source1_id in grouped:
        grouped[source1_id].sort(
            key=lambda candidate: (
                candidate.calibrated_score,
                candidate.raw_score,
                candidate.target_source,
                candidate.target_id,
            ),
            reverse=True,
        )

    return grouped


def _margin(
    ordered_candidates: list[CandidateScore],
) -> float:
    if not ordered_candidates:
        return 0.0

    if len(ordered_candidates) == 1:
        return ordered_candidates[0].calibrated_score

    return (
        ordered_candidates[0].calibrated_score
        - ordered_candidates[1].calibrated_score
    )


def decide_s1(
    source1_id: str,
    candidates: list[CandidateScore],
    config: S1PolicyConfig,
) -> S1Decision:
    """
    Apply the S1-level policy.

    Existing high-confidence matches always win.

    Rescue logic is used only if the global threshold produces no match.
    """

    if not candidates:
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="no_candidates",
            calibrated_score=None,
            margin=None,
            rescue_used=False,
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.calibrated_score,
            candidate.raw_score,
            candidate.target_source,
            candidate.target_id,
        ),
        reverse=True,
    )

    accepted = [
        candidate
        for candidate in ordered
        if candidate.calibrated_score
        >= config.global_threshold
        and not (
            config.reject_country_conflict
            and candidate.country_conflict
        )
    ]

    if accepted:
        best = accepted[0]

        return S1Decision(
            source1_id=source1_id,
            target_source=best.target_source,
            target_id=best.target_id,
            decision="MATCH",
            reason="global_threshold",
            calibrated_score=best.calibrated_score,
            margin=_margin(ordered),
            rescue_used=False,
        )

    if not config.enable_argmax_rescue:
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="below_global_threshold",
            calibrated_score=ordered[0].calibrated_score,
            margin=_margin(ordered),
            rescue_used=False,
        )

    best = ordered[0]

    if (
        config.reject_country_conflict
        and best.country_conflict
    ):
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="rescue_country_conflict",
            calibrated_score=best.calibrated_score,
            margin=_margin(ordered),
            rescue_used=False,
        )

    if best.calibrated_score < config.rescue_min_score:
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="rescue_score_below_floor",
            calibrated_score=best.calibrated_score,
            margin=_margin(ordered),
            rescue_used=False,
        )

    margin = _margin(ordered)

    if (
        config.require_positive_margin
        and len(ordered) > 1
        and margin < config.rescue_min_margin
    ):
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="rescue_margin_too_small",
            calibrated_score=best.calibrated_score,
            margin=margin,
            rescue_used=False,
        )

    if (
        config.require_strong_evidence_for_rescue
        and best.strong_evidence_count
        < config.min_strong_evidence_count
    ):
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="rescue_missing_strong_evidence",
            calibrated_score=best.calibrated_score,
            margin=margin,
            rescue_used=False,
        )

    if (
        best.combined_evidence_count
        < config.min_positive_evidence_count
    ):
        return S1Decision(
            source1_id=source1_id,
            target_source=None,
            target_id=None,
            decision="NO_MATCH",
            reason="rescue_insufficient_evidence",
            calibrated_score=best.calibrated_score,
            margin=margin,
            rescue_used=False,
        )

    return S1Decision(
        source1_id=source1_id,
        target_source=best.target_source,
        target_id=best.target_id,
        decision="MATCH",
        reason="s1_argmax_rescue",
        calibrated_score=best.calibrated_score,
        margin=margin,
        rescue_used=True,
    )


def decide_s1_candidates(
    source1_id: str,
    candidates: list[CandidateScore],
    config: S1PolicyConfig,
) -> list[S1Decision]:
    """
    Apply S1 policy to all candidates.
    If any candidates meet the primary global threshold, all accepted candidates
    are retained (preserving the frozen multi-target match decisions).
    If no candidate meets the threshold, the single best candidate is evaluated for rescue.
    """
    if not candidates:
        return [decide_s1(source1_id, candidates, config)]

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.calibrated_score,
            candidate.raw_score,
            candidate.target_source,
            candidate.target_id,
        ),
        reverse=True,
    )

    accepted = [
        candidate
        for candidate in ordered
        if candidate.calibrated_score
        >= config.global_threshold
        and not (
            config.reject_country_conflict
            and candidate.country_conflict
        )
    ]

    if accepted:
        margin = _margin(ordered)
        return [
            S1Decision(
                source1_id=source1_id,
                target_source=c.target_source,
                target_id=c.target_id,
                decision="MATCH",
                reason="global_threshold",
                calibrated_score=c.calibrated_score,
                margin=margin,
                rescue_used=False,
            )
            for c in accepted
        ]

    return [decide_s1(source1_id, candidates, config)]


def apply_s1_policy(
    rows: Iterable[dict[str, Any]],
    config: S1PolicyConfig,
) -> list[S1Decision]:
    grouped = group_candidates(rows)

    decisions = []

    for source1_id in sorted(grouped):
        decisions.extend(
            decide_s1_candidates(
                source1_id,
                grouped[source1_id],
                config,
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# Frozen M9.4 P1-A Decision Policy
# ---------------------------------------------------------------------------

P1_A_POLICY_NAME = "P1-A"
P1_A_THRESHOLD = 0.8750
P1_A_STRONG_EVIDENCE_THRESHOLD = 0.7000

P1_A_CONFIG = {
    "name": P1_A_POLICY_NAME,
    "primary_threshold": P1_A_THRESHOLD,
    "strong_evidence_threshold": P1_A_STRONG_EVIDENCE_THRESHOLD,
    "minimum_strong_evidence_count": 1,
}


def accept_p1_a(
    calibrated_probability: float,
    strong_evidence_count: int,
) -> bool:
    """
    Frozen M9.4 P1-A decision policy.

    Accept when:
      1. calibrated probability reaches the primary threshold, OR
      2. at least one strong evidence signal exists and the probability
         reaches the controlled rescue threshold.
    """

    if calibrated_probability >= P1_A_THRESHOLD:
        return True

    if (
        strong_evidence_count >= 1
        and calibrated_probability >= P1_A_STRONG_EVIDENCE_THRESHOLD
    ):
        return True

    return False

