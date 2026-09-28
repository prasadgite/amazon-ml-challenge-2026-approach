from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlockingMiss:
    source1_id: str
    target_id: str
    target_source: str

    source1_name: str
    target_name: str

    source1_address: str
    target_address: str

    source1_country: str
    target_country: str

    baseline_present: bool
    enhanced_present: bool

    failure_reason: str


def read_rows(
    path: Path,
    needed_ids: set[str] | None = None,
) -> dict[str, dict[str, str]]:
    result = {}
    path = Path(path)
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as handle:
        header_line = handle.readline()
        if not header_line:
            return result
        header = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]

        for line in handle:
            tab_pos = line.find("\t")
            if tab_pos == -1:
                continue
            eid = line[:tab_pos].strip()
            if needed_ids is not None:
                if eid not in needed_ids:
                    continue
            parts = [p.strip() for p in line.rstrip("\r\n").split("\t")]
            row = dict(zip(header, parts))
            result[eid] = row
            if needed_ids is not None and len(result) >= len(needed_ids):
                break

    return result


def read_ground_truth(
    path: Path,
) -> list[tuple[str, str, str]]:
    pairs = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:

            source1_id = (
                row[
                    "source1_entity_id"
                ]
                .strip()
            )

            matched = (
                row[
                    "matched_entity_ids"
                ]
                .strip()
            )

            if not matched:
                continue

            for target_id in matched.split(","):

                target_id = target_id.strip()

                if not target_id:
                    continue

                if target_id.startswith("S2-"):
                    source = "S2"
                elif target_id.startswith("S3-"):
                    source = "S3"
                else:
                    continue

                pairs.append(
                    (
                        source1_id,
                        source,
                        target_id,
                    )
                )

    return pairs


def read_candidate_pairs(
    path: Path,
) -> set[tuple[str, str]]:
    pairs = set()

    path = Path(path)

    if path.suffix in {".sqlite", ".db"}:
        import sqlite3
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cursor.fetchall()}
        target_table = "candidates" if "candidates" in tables else list(tables)[0]
        for row in cursor.execute(f"SELECT source1_id, target_id FROM {target_table}"):
            pairs.add((str(row[0]).strip(), str(row[1]).strip()))
        conn.close()
        return pairs

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        fields = set(
            reader.fieldnames or []
        )

        if {
            "source1_id",
            "target_id",
        }.issubset(fields):

            for row in reader:

                pairs.add(
                    (
                        row["source1_id"].strip(),
                        row["target_id"].strip(),
                    )
                )

        elif {
            "source1_entity_id",
            "target_entity_id",
        }.issubset(fields):

            for row in reader:

                pairs.add(
                    (
                        row[
                            "source1_entity_id"
                        ].strip(),
                        row[
                            "target_entity_id"
                        ].strip(),
                    )
                )

        elif {
            "source1_entity_id",
            "candidate_entity_ids",
        }.issubset(fields):

            for row in reader:
                s1 = row["source1_entity_id"].strip()
                raw_cands = row.get("candidate_entity_ids", "") or ""
                for target_id in raw_cands.split(","):
                    target_id = target_id.strip()
                    if target_id:
                        pairs.add((s1, target_id))

        else:

            raise ValueError(
                "Unsupported candidate-pair schema: "
                f"{sorted(fields)}"
            )

    return pairs


def analyze_remaining_misses(
    ground_truth_path: Path,
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    baseline_candidates_path: Path,
    enhanced_candidates_path: Path,
) -> list[BlockingMiss]:

    truth = read_ground_truth(
        ground_truth_path
    )

    baseline = read_candidate_pairs(
        baseline_candidates_path
    )

    enhanced = read_candidate_pairs(
        enhanced_candidates_path
    )

    missed_pairs = [
        (s1_id, target_source, target_id)
        for (s1_id, target_source, target_id) in truth
        if (s1_id, target_id) not in enhanced
    ]

    needed_s1 = {s1_id for s1_id, _, _ in missed_pairs}
    needed_targets = {target_id for _, _, target_id in missed_pairs}

    source1 = read_rows(
        source1_path,
        needed_ids=needed_s1,
    )

    source2 = read_rows(
        source2_path,
        needed_ids=needed_targets,
    )

    source3 = read_rows(
        source3_path,
        needed_ids=needed_targets,
    )

    targets = {
        **source2,
        **source3,
    }

    result = []

    for (
        source1_id,
        target_source,
        target_id,
    ) in missed_pairs:

        pair = (
            source1_id,
            target_id,
        )

        left = source1.get(
            source1_id,
            {},
        )

        right = targets.get(
            target_id,
            {},
        )

        result.append(
            BlockingMiss(
                source1_id=source1_id,
                target_id=target_id,
                target_source=target_source,
                source1_name=left.get(
                    "business_name",
                    "",
                ),
                target_name=right.get(
                    "business_name",
                    "",
                ),
                source1_address=left.get(
                    "business_address",
                    "",
                ),
                target_address=right.get(
                    "business_address",
                    "",
                ),
                source1_country=left.get(
                    "country",
                    "",
                ),
                target_country=right.get(
                    "country",
                    "",
                ),
                baseline_present=(
                    pair in baseline
                ),
                enhanced_present=(
                    pair in enhanced
                ),
                failure_reason=(
                    "NO_B0_NO_B5"
                ),
            )
        )

    return result


def write_misses(
    path: Path,
    misses: list[BlockingMiss],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "source1_id",
        "target_id",
        "target_source",
        "source1_name",
        "target_name",
        "source1_address",
        "target_address",
        "source1_country",
        "target_country",
        "baseline_present",
        "enhanced_present",
        "failure_reason",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
        )

        writer.writeheader()

        for miss in misses:
            writer.writerow(
                {
                    field: getattr(
                        miss,
                        field,
                    )
                    for field in fields
                }
            )


TOKEN_RE = re.compile(
    r"[^\W_]+",
    re.UNICODE,
)


def tokens(value: str) -> set[str]:
    return {
        x.casefold()
        for x in TOKEN_RE.findall(
            value or ""
        )
    }


def classify_name(
    left: str,
    right: str,
) -> str:
    a = tokens(left)
    b = tokens(right)

    if not a or not b:
        return "NAME_EMPTY"

    if a == b:
        return "NAME_TOKEN_EXACT"

    if a.issubset(b) or b.issubset(a):
        return "NAME_TOKEN_CONTAINMENT"

    overlap = len(a & b)

    if overlap:
        return "NAME_PARTIAL_OVERLAP"

    if any(
        x[:4] == y[:4]
        for x in a
        for y in b
        if len(x) >= 4 and len(y) >= 4
    ):
        return "NAME_PREFIX_SIMILAR"

    return "NAME_NO_TOKEN_OVERLAP"


def classify_country(
    left: str,
    right: str,
) -> str:
    left = (left or "").strip().casefold()
    right = (right or "").strip().casefold()

    if not left or not right:
        return "COUNTRY_MISSING"

    if left == right:
        return "COUNTRY_MATCH"

    return "COUNTRY_CONFLICT"

