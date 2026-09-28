from __future__ import annotations

from business_entity_resolution.decision.s1_policy import (
    CandidateScore,
    S1PolicyConfig,
    decide_s1,
)


def candidate(
    target_id: str,
    score: float,
    *,
    country_conflict: bool = False,
    strong: int = 0,
    combined: int = 1,
):
    return CandidateScore(
        source1_id="S1-1",
        target_source="S2",
        target_id=target_id,
        calibrated_score=score,
        raw_score=score,
        country_conflict=country_conflict,
        strong_evidence_count=strong,
        combined_evidence_count=combined,
    )


def test_global_threshold_match():
    config = S1PolicyConfig(
        global_threshold=0.8625,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate("S2-1", 0.95),
            candidate("S2-2", 0.80),
        ],
        config,
    )

    assert result.decision == "MATCH"
    assert result.target_id == "S2-1"
    assert result.reason == "global_threshold"
    assert result.rescue_used is False


def test_below_threshold_without_rescue():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=False,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate("S2-1", 0.85),
            candidate("S2-2", 0.60),
        ],
        config,
    )

    assert result.decision == "NO_MATCH"
    assert result.reason == "below_global_threshold"


def test_argmax_rescue():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=True,
        rescue_min_score=0.70,
        rescue_min_margin=0.05,
        require_positive_margin=True,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate("S2-1", 0.85),
            candidate("S2-2", 0.70),
        ],
        config,
    )

    assert result.decision == "MATCH"
    assert result.target_id == "S2-1"
    assert result.reason == "s1_argmax_rescue"
    assert result.rescue_used is True


def test_margin_blocks_ambiguous_rescue():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=True,
        rescue_min_score=0.70,
        rescue_min_margin=0.05,
        require_positive_margin=True,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate("S2-1", 0.80),
            candidate("S2-2", 0.78),
        ],
        config,
    )

    assert result.decision == "NO_MATCH"
    assert result.reason == "rescue_margin_too_small"


def test_country_conflict_blocks_rescue():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=True,
        rescue_min_score=0.70,
        reject_country_conflict=True,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate(
                "S2-1",
                0.90,
                country_conflict=True,
            ),
        ],
        config,
    )

    assert result.decision == "NO_MATCH"
    assert result.reason == "rescue_country_conflict"


def test_strong_evidence_rescue():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=True,
        rescue_min_score=0.70,
        rescue_min_margin=0.05,
        require_positive_margin=True,
        require_strong_evidence_for_rescue=True,
        min_strong_evidence_count=1,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate(
                "S2-1",
                0.82,
                strong=2,
            ),
            candidate(
                "S2-2",
                0.70,
                strong=0,
            ),
        ],
        config,
    )

    assert result.decision == "MATCH"
    assert result.target_id == "S2-1"


def test_weak_candidate_rejected_by_strong_evidence_policy():
    config = S1PolicyConfig(
        global_threshold=0.8625,
        enable_argmax_rescue=True,
        rescue_min_score=0.70,
        rescue_min_margin=0.05,
        require_positive_margin=True,
        require_strong_evidence_for_rescue=True,
        min_strong_evidence_count=1,
    )

    result = decide_s1(
        "S1-1",
        [
            candidate(
                "S2-1",
                0.82,
                strong=0,
            ),
            candidate(
                "S2-2",
                0.70,
                strong=0,
            ),
        ],
        config,
    )

    assert result.decision == "NO_MATCH"
    assert result.reason == "rescue_missing_strong_evidence"


def test_no_candidates():
    config = S1PolicyConfig()

    result = decide_s1(
        "S1-1",
        [],
        config,
    )

    assert result.decision == "NO_MATCH"
    assert result.reason == "no_candidates"
