from __future__ import annotations

import pytest

from business_entity_resolution.decision.s1_policy import (
    P1_A_POLICY_NAME,
    P1_A_THRESHOLD,
    P1_A_STRONG_EVIDENCE_THRESHOLD,
    P1_A_CONFIG,
    accept_p1_a,
)


def test_primary_threshold_accepts():
    assert accept_p1_a(0.8750, 0) is True


def test_primary_threshold_below_rejects_without_evidence():
    assert accept_p1_a(0.8749, 0) is False


def test_strong_evidence_rescue_accepts():
    assert accept_p1_a(0.7000, 1) is True


def test_strong_evidence_below_rescue_threshold_rejects():
    assert accept_p1_a(0.6999, 1) is False


def test_no_strong_evidence_does_not_rescue():
    assert accept_p1_a(0.8000, 0) is False


def test_multiple_strong_evidence_accepts():
    assert accept_p1_a(0.7000, 2) is True


def test_known_false_positive_boundary_rejects():
    # M9.4 audit maximum strong-evidence FP score = 0.5769.
    assert accept_p1_a(0.5769, 1) is False


def test_p1_a_metadata():
    assert P1_A_POLICY_NAME == "P1-A"
    assert P1_A_THRESHOLD == 0.8750
    assert P1_A_STRONG_EVIDENCE_THRESHOLD == 0.7000
    assert P1_A_CONFIG["name"] == "P1-A"
    assert P1_A_CONFIG["primary_threshold"] == 0.8750
    assert P1_A_CONFIG["strong_evidence_threshold"] == 0.7000
    assert P1_A_CONFIG["minimum_strong_evidence_count"] == 1
