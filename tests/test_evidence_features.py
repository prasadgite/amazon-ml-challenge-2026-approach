from __future__ import annotations

import math

from business_entity_resolution.features.evidence import (
    combined_name_domain_evidence,
    domain_base,
    domain_name_tokens,
    domain_to_name_similarity,
    evidence_tokens,
    normalize_evidence_text,
    secondary_evidence,
    soft_token_coverage,
    soft_token_similarity,
    token_jaccard,
)


def test_normalization_is_unicode_safe() -> None:
    assert normalize_evidence_text(
        "Société Générale"
    ) == "societe generale"


def test_domain_base() -> None:
    assert (
        domain_base(
            "https://www.AggieENagle.com/path"
        )
        == "aggieenagle.com"
    )


def test_domain_tokens() -> None:
    assert domain_name_tokens(
        "aggie-e-nagle.com"
    ) == (
        "aggie",
        "e",
        "nagle",
    )


def test_exact_token_jaccard() -> None:
    assert math.isclose(
        token_jaccard(
            ("alpha", "beta"),
            ("alpha", "gamma"),
        ),
        1.0 / 3.0,
    )


def test_soft_token_similarity() -> None:
    score = soft_token_similarity(
        ("aggie", "nagle"),
        ("aggie", "nagl"),
    )

    assert score > 0.80


def test_soft_token_coverage() -> None:
    score = soft_token_coverage(
        ("aggie", "nagle"),
        ("aggie", "nagl"),
        threshold=0.80,
    )

    assert score == 1.0


def test_domain_name_alignment() -> None:
    score = domain_to_name_similarity(
        "aggieenagle.com",
        "Aggie E Nagle",
    )

    assert score >= 0.70


def test_combined_name_domain_evidence() -> None:
    score = combined_name_domain_evidence(
        "Aggie E Nagle",
        "aggieenagle.com",
    )

    assert score >= 0.70


def test_secondary_evidence_schema() -> None:
    result = secondary_evidence(
        name_left="Aggie E Nagle",
        name_right="Aggie E. Nagle",
        domain_left="aggieenagle.com",
        domain_right="aggieenagle.com",
    )

    expected = {
        "soft_token_similarity",
        "soft_token_coverage",
        "domain_name_similarity_left_domain",
        "domain_name_similarity_right_domain",
        "domain_name_coverage_left_domain",
        "domain_name_coverage_right_domain",
        "name_domain_coverage_left_name",
        "name_domain_coverage_right_name",
        "combined_name_domain_evidence",
    }

    assert set(result) == expected

    for value in result.values():
        assert 0.0 <= value <= 1.0


def test_empty_evidence_is_zero() -> None:
    result = secondary_evidence()

    assert all(
        value == 0.0
        for value in result.values()
    )


def test_cross_script_text_is_not_crashed() -> None:
    result = secondary_evidence(
        name_left="मुंबई ट्रेडर्स",
        name_right="Mumbai Traders",
    )

    assert all(
        0.0 <= value <= 1.0
        for value in result.values()
    )
