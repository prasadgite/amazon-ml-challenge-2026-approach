from __future__ import annotations

import csv
from pathlib import Path

from business_entity_resolution.features.evidence_augmentation import (
    EXTRA_FEATURES,
    augment_feature_file,
    augment_feature_rows,
    extract_embedded_domain,
    pair_evidence_features,
)


def test_extract_embedded_domain():
    assert (
        extract_embedded_domain(
            "Acme Corp *** acme-example.com"
        )
        == "acme-example.com"
    )

    assert (
        extract_embedded_domain(
            "No domain here"
        )
        == ""
    )


def test_pair_evidence_schema():

    values = pair_evidence_features(
        "Acme Trading",
        "Acme Tradng",
    )

    assert set(values) == set(
        EXTRA_FEATURES
    )

    assert (
        values["soft_token_similarity"]
        > 0.8
    )

    assert (
        values["soft_token_coverage"]
        == 1.0
    )


def test_domain_evidence():

    values = pair_evidence_features(
        "Acme Holdings acme.com",
        "Acme Holdings acme.com",
    )

    assert (
        values["embedded_domain_present"]
        == 1.0
    )

    assert (
        values[
            "embedded_domain_left_present"
        ]
        == 1.0
    )

    assert (
        values[
            "embedded_domain_right_present"
        ]
        == 1.0
    )

    assert (
        values[
            "embedded_domain_name_similarity"
        ]
        == 1.0
    )


def test_e0_adds_zero_features():

    rows = list(
        augment_feature_rows(
            [
                {
                    "source1_id": "S1-1",
                    "target_id": "S2-1",
                    "label": "1",
                }
            ],
            {
                "S1-1": "Acme",
                "S2-1": "Acme",
            },
            "e0",
        )
    )

    assert all(
        float(rows[0][key]) == 0.0
        for key in EXTRA_FEATURES
    )


def test_e1_only_fuzzy_features():

    rows = list(
        augment_feature_rows(
            [
                {
                    "source1_id": "S1-1",
                    "target_id": "S2-1",
                    "label": "1",
                }
            ],
            {
                "S1-1": "Acme Trading",
                "S2-1": "Acme Tradng acme.com",
            },
            "e1",
        )
    )

    assert (
        float(
            rows[0][
                "soft_token_similarity"
            ]
        )
        > 0.0
    )

    assert (
        float(
            rows[0][
                "embedded_domain_present"
            ]
        )
        == 0.0
    )


def test_e2_only_domain_features():

    rows = list(
        augment_feature_rows(
            [
                {
                    "source1_id": "S1-1",
                    "target_id": "S2-1",
                    "label": "1",
                }
            ],
            {
                "S1-1": "Acme acme.com",
                "S2-1": "Acme acme.com",
            },
            "e2",
        )
    )

    assert (
        float(
            rows[0][
                "soft_token_similarity"
            ]
        )
        == 0.0
    )

    assert (
        float(
            rows[0][
                "embedded_domain_present"
            ]
        )
        == 1.0
    )


def test_augment_file_preserves_base_columns(
    tmp_path: Path,
):

    source = tmp_path / "in.tsv"
    target = tmp_path / "out.tsv"

    with source.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source1_id",
                "target_id",
                "label",
            ],
            delimiter="\t",
        )

        writer.writeheader()

        writer.writerow(
            {
                "source1_id": "S1-1",
                "target_id": "S2-1",
                "label": "1",
            }
        )

    count = augment_feature_file(
        source,
        target,
        {
            "S1-1": "Acme",
            "S2-1": "Acme",
        },
        "e3",
    )

    assert count == 1

    with target.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        row = next(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    assert row["label"] == "1"

    assert set(
        EXTRA_FEATURES
    ).issubset(row)
