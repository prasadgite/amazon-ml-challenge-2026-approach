from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


EXTRA_FEATURES = (
    "soft_token_similarity",
    "soft_token_coverage",
    "embedded_domain_present",
    "embedded_domain_left_present",
    "embedded_domain_right_present",
    "embedded_domain_name_similarity",
    "embedded_domain_name_coverage",
    "combined_name_domain_evidence",
)

_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9-])"
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"([A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z]{2,63})+)"
    r"(?![A-Za-z0-9-])",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for token in _TOKEN_RE.findall(value or "")
        if token
    )


def _token_similarity(a: str, b: str) -> float:
    left = set(_tokens(a))
    right = set(_tokens(b))

    if not left or not right:
        return 0.0

    matched = 0.0

    for token in left:
        matched += max(
            (
                SequenceMatcher(None, token, candidate).ratio()
                for candidate in right
            ),
            default=0.0,
        )

    return matched / len(left)


def _token_coverage(a: str, b: str) -> float:
    left = set(_tokens(a))
    right = set(_tokens(b))

    if not left or not right:
        return 0.0

    matched = sum(
        1
        for token in left
        if max(
            (
                SequenceMatcher(None, token, candidate).ratio()
                for candidate in right
            ),
            default=0.0,
        ) >= 0.85
    )

    return matched / len(left)


def extract_embedded_domain(value: str) -> str:
    """Extract the first domain embedded in a business-name string."""

    match = _DOMAIN_RE.search(value or "")

    if not match:
        return ""

    return match.group(1).casefold()


def _domain_name(domain: str) -> str:
    if not domain:
        return ""

    host = domain.split(":", 1)[0].casefold()
    parts = host.split(".")

    if len(parts) < 2:
        return ""

    return parts[0].replace("-", " ")


def _as_text(value: object) -> str:
    return str(value or "")


def pair_evidence_features(
    left_name: str,
    right_name: str,
) -> dict[str, float]:

    left_domain = extract_embedded_domain(left_name)
    right_domain = extract_embedded_domain(right_name)

    left_domain_name = _domain_name(left_domain)
    right_domain_name = _domain_name(right_domain)

    domain_present = bool(left_domain or right_domain)

    if left_domain_name and right_domain_name:
        domain_name_similarity = SequenceMatcher(
            None,
            left_domain_name,
            right_domain_name,
        ).ratio()
    else:
        domain_name_similarity = 0.0

    domain_name_coverage = _token_coverage(
        left_domain_name,
        right_domain_name,
    )

    soft_similarity = _token_similarity(
        left_name,
        right_name,
    )

    return {
        "soft_token_similarity": round(
            soft_similarity,
            8,
        ),
        "soft_token_coverage": round(
            _token_coverage(left_name, right_name),
            8,
        ),
        "embedded_domain_present": float(
            domain_present
        ),
        "embedded_domain_left_present": float(
            bool(left_domain)
        ),
        "embedded_domain_right_present": float(
            bool(right_domain)
        ),
        "embedded_domain_name_similarity": round(
            domain_name_similarity,
            8,
        ),
        "embedded_domain_name_coverage": round(
            domain_name_coverage,
            8,
        ),
        "combined_name_domain_evidence": round(
            max(
                soft_similarity,
                domain_name_similarity,
            ),
            8,
        ),
    }


def build_entity_name_index(
    paths: Iterable[Path],
) -> dict[str, str]:

    index: dict[str, str] = {}

    for path in paths:
        with Path(path).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:

            reader = csv.DictReader(
                handle,
                delimiter="\t",
            )

            fields = set(
                reader.fieldnames or ()
            )

            if {
                "entity_id",
                "business_name",
            } - fields:

                raise ValueError(
                    f"Expected entity_id and business_name "
                    f"in {path}; got {sorted(fields)}"
                )

            for row in reader:
                entity_id = _as_text(
                    row.get("entity_id")
                ).strip()

                if entity_id:
                    index[entity_id] = _as_text(
                        row.get("business_name")
                    )

    return index


def augment_feature_rows(
    rows: Iterable[dict[str, str]],
    name_index: dict[str, str],
    variant: str,
) -> Iterable[dict[str, str]]:

    variant = variant.lower()

    if variant not in {"e0", "e1", "e2", "e3"}:
        raise ValueError(
            "variant must be e0, e1, e2, or e3"
        )

    use_fuzzy = variant in {"e1", "e3"}
    use_domain = variant in {"e2", "e3"}

    for row in rows:

        output = dict(row)

        extras = {
            name: 0.0
            for name in EXTRA_FEATURES
        }

        if use_fuzzy or use_domain:

            values = pair_evidence_features(
                name_index.get(
                    _as_text(row.get("source1_id")),
                    "",
                ),
                name_index.get(
                    _as_text(row.get("target_id")),
                    "",
                ),
            )

            if use_fuzzy:
                extras[
                    "soft_token_similarity"
                ] = values[
                    "soft_token_similarity"
                ]

                extras[
                    "soft_token_coverage"
                ] = values[
                    "soft_token_coverage"
                ]

            if use_domain:
                for key in EXTRA_FEATURES[2:]:
                    extras[key] = values[key]

        output.update(
            {
                key: str(value)
                for key, value in extras.items()
            }
        )

        yield output


def augment_feature_file(
    input_path: Path,
    output_path: Path,
    name_index: dict[str, str],
    variant: str,
) -> int:

    with Path(input_path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source:

        reader = csv.DictReader(
            source,
            delimiter="\t",
        )

        base_fields = list(
            reader.fieldnames or []
        )

        required = {
            "source1_id",
            "target_id",
            "label",
        }

        missing = required - set(base_fields)

        if missing:
            raise ValueError(
                "Feature file is missing required "
                f"columns: {sorted(missing)}"
            )

        fields = (
            base_fields
            + [
                feature
                for feature in EXTRA_FEATURES
                if feature not in base_fields
            ]
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = 0

        with Path(output_path).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as destination:

            writer = csv.DictWriter(
                destination,
                fieldnames=fields,
                delimiter="\t",
                extrasaction="ignore",
            )

            writer.writeheader()

            for row in augment_feature_rows(
                reader,
                name_index,
                variant,
            ):
                writer.writerow(row)
                count += 1

    return count
