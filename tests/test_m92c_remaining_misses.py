from pathlib import Path

from business_entity_resolution.blocking.miss_analysis import (
    analyze_remaining_misses,
    write_misses,
)


def write_tsv(
    path: Path,
    rows: list[dict[str, str]],
):
    fields = list(rows[0])

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        handle.write(
            "\t".join(fields)
            + "\n"
        )

        for row in rows:
            handle.write(
                "\t".join(
                    row[field]
                    for field in fields
                )
                + "\n"
            )


def test_remaining_miss_detection(
    tmp_path: Path,
):

    gt = tmp_path / "gt.tsv"
    s1 = tmp_path / "s1.tsv"
    s2 = tmp_path / "s2.tsv"
    s3 = tmp_path / "s3.tsv"
    b0 = tmp_path / "b0.tsv"
    b5 = tmp_path / "b5.tsv"

    write_tsv(
        gt,
        [
            {
                "source1_entity_id": "S1-1",
                "matched_entity_ids": "S2-1",
            }
        ],
    )

    write_tsv(
        s1,
        [
            {
                "entity_id": "S1-1",
                "business_name": "Alpha",
                "business_address": "A",
                "country": "US",
            }
        ],
    )

    write_tsv(
        s2,
        [
            {
                "entity_id": "S2-1",
                "business_name": "Alpha",
                "business_address": "A",
                "country": "US",
            }
        ],
    )

    write_tsv(
        s3,
        [
            {
                "entity_id": "S3-1",
                "business_name": "Other",
                "business_address": "B",
                "country": "US",
            }
        ],
    )

    write_tsv(
        b0,
        [
            {
                "source1_id": "S1-1",
                "target_id": "S3-1",
            }
        ],
    )

    write_tsv(
        b5,
        [
            {
                "source1_id": "S1-1",
                "target_id": "S3-1",
            }
        ],
    )

    misses = analyze_remaining_misses(
        gt,
        s1,
        s2,
        s3,
        b0,
        b5,
    )

    assert len(misses) == 1

    assert (
        misses[0].source1_id
        == "S1-1"
    )

    assert (
        misses[0].target_id
        == "S2-1"
    )

    assert (
        misses[0].failure_reason
        == "NO_B0_NO_B5"
    )


def test_recovered_pair_is_not_a_miss(
    tmp_path: Path,
):

    gt = tmp_path / "gt.tsv"
    s1 = tmp_path / "s1.tsv"
    s2 = tmp_path / "s2.tsv"
    s3 = tmp_path / "s3.tsv"
    b0 = tmp_path / "b0.tsv"
    b5 = tmp_path / "b5.tsv"

    write_tsv(
        gt,
        [
            {
                "source1_entity_id": "S1-1",
                "matched_entity_ids": "S2-1",
            }
        ],
    )

    for path, rows in [
        (
            s1,
            [
                {
                    "entity_id": "S1-1",
                    "business_name": "Alpha",
                    "business_address": "A",
                    "country": "US",
                }
            ],
        ),
        (
            s2,
            [
                {
                    "entity_id": "S2-1",
                    "business_name": "Alpha",
                    "business_address": "A",
                    "country": "US",
                }
            ],
        ),
        (
            s3,
            [
                {
                    "entity_id": "S3-1",
                    "business_name": "Other",
                    "business_address": "B",
                    "country": "US",
                }
            ],
        ),
    ]:
        write_tsv(path, rows)

    write_tsv(
        b0,
        [
            {
                "source1_id": "S1-1",
                "target_id": "S3-1",
            }
        ],
    )

    write_tsv(
        b5,
        [
            {
                "source1_id": "S1-1",
                "target_id": "S2-1",
            }
        ],
    )

    misses = analyze_remaining_misses(
        gt,
        s1,
        s2,
        s3,
        b0,
        b5,
    )

    assert misses == []
