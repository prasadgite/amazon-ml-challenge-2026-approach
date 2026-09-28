from __future__ import annotations

import csv
import json
import logging
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from difflib import SequenceMatcher

from ..normalization.normalizer import NormalizedRecord, TextNormalizer, iter_normalized_tsv

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for deterministic pairwise feature generation."""

    progress_every: int = 500_000
    batch_size: int = 5_000
    include_labels: bool = True


class FeatureSchema:
    """Stable ordered feature schema used by training and inference."""

    CATEGORICAL = ()
    BINARY = (
        "country_match", "country_conflict", "country_missing_either",
        "name_exact", "address_exact", "name_address_exact",
        "house_exact", "postal_exact", "domain_exact", "alias_exact",
        "name_nonempty_both", "address_nonempty_both", "domain_available_both",
        "alias_available_both", "name_address_both_available",
        "blocking_hit_count", "blocking_strategy_count",
    )
    NUMERIC = (
        "name_jaro_winkler", "name_edit_similarity", "name_token_jaccard",
        "name_token_containment", "name_sorted_token_jaccard",
        "name_char3_jaccard", "name_char4_jaccard", "name_length_ratio",
        "address_edit_similarity", "address_token_jaccard",
        "address_token_containment", "address_char3_jaccard", "address_char4_jaccard",
        "address_length_ratio", "house_overlap", "postal_overlap",
        "domain_jaccard", "alias_jaccard", "combined_evidence_count",
        "strong_evidence_count", "missing_field_count",
    )

    @classmethod
    def columns(cls, include_label: bool = True, include_labels: bool | None = None) -> list[str]:
        use_label = include_label if include_labels is None else include_labels
        columns = [
            "source1_id", "target_source", "target_id", *cls.CATEGORICAL,
            *cls.BINARY, *cls.NUMERIC,
        ]
        if use_label:
            columns.append("label")
        return columns


@dataclass(frozen=True)
class _Record:
    source: str
    record: NormalizedRecord


def _safe_ratio(a: int, b: int) -> float:
    if not a or not b:
        return 0.0
    return min(a, b) / max(a, b)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def _containment(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _overlap(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        current = [i]
        for j, ca in enumerate(a, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


def _edit_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    distance = _levenshtein(a, b)
    return max(0.0, 1.0 - distance / max(len(a), len(b)))


def _jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0 if a else 0.0
    if not a or not b:
        return 0.0
    match_distance = max(len(a), len(b)) // 2 - 1
    match_distance = max(match_distance, 0)
    a_match = [False] * len(a)
    b_match = [False] * len(b)
    matches = 0
    for i, ca in enumerate(a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(b))
        for j in range(start, end):
            if b_match[j] or ca != b[j]:
                continue
            a_match[i] = True
            b_match[j] = True
            matches += 1
            break
    if not matches:
        return 0.0
    a_seq = [a[i] for i, ok in enumerate(a_match) if ok]
    b_seq = [b[j] for j, ok in enumerate(b_match) if ok]
    transpositions = sum(x != y for x, y in zip(a_seq, b_seq)) / 2.0
    return (
        matches / len(a) + matches / len(b) +
        (matches - transpositions) / matches
    ) / 3.0


def _jaro_winkler(a: str, b: str) -> float:
    score = _jaro(a, b)
    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        prefix += 1
    if score > 0.7:
        score += prefix * 0.1 * (1.0 - score)
    return min(score, 1.0)


def _json_tuple(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _record_payload(record: NormalizedRecord) -> tuple:
    return (
        record.source if hasattr(record, "source") else "",
        record.entity_id,
        record.country,
        record.name_core,
        _json_tuple(record.name_tokens),
        _json_tuple(record.name_sorted_tokens),
        _json_tuple(record.name_aliases),
        _json_tuple(record.domains),
        _json_tuple(record.name_char3),
        _json_tuple(record.name_char4),
        record.address_street_core,
        _json_tuple(record.address_street_tokens),
        _json_tuple(record.house_numbers),
        _json_tuple(record.postal_codes),
        _json_tuple(record.address_char3),
        _json_tuple(record.address_char4),
    )


class PairFeatureEngine:
    """Build labeled pairwise features from M4 candidates without RAM growth.

    A compact SQLite record store is built for all three normalized sources.
    Candidate pairs are streamed from the M4 blocking database and emitted as
    TSV. Ground-truth links are loaded into SQLite so labels do not require an
    in-memory set proportional to the number of training links.
    """

    RECORD_DDL = """
        CREATE TABLE IF NOT EXISTS records (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            country TEXT NOT NULL,
            name_core TEXT NOT NULL,
            name_tokens TEXT NOT NULL,
            name_sorted_tokens TEXT NOT NULL,
            name_aliases TEXT NOT NULL,
            domains TEXT NOT NULL,
            name_char3 TEXT NOT NULL,
            name_char4 TEXT NOT NULL,
            address_street_core TEXT NOT NULL,
            address_street_tokens TEXT NOT NULL,
            house_numbers TEXT NOT NULL,
            postal_codes TEXT NOT NULL,
            address_char3 TEXT NOT NULL,
            address_char4 TEXT NOT NULL,
            PRIMARY KEY(source, entity_id)
        )
    """

    GT_DDL = """
        CREATE TABLE IF NOT EXISTS ground_truth (
            source1_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            PRIMARY KEY(source1_id, target_id)
        )
    """

    def __init__(self, feature_db_path: Path, config: FeatureConfig | None = None) -> None:
        self.feature_db_path = Path(feature_db_path)
        self.config = config or FeatureConfig()
        self.feature_db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.feature_db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(self.RECORD_DDL)
            conn.execute(self.GT_DDL)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_records_id ON records(entity_id)")
            conn.commit()

    def build_record_store(
        self,
        source_paths: dict[str, Path],
        normalizer: TextNormalizer | None = None,
        max_rows: int = 0,
    ) -> int:
        normalizer = normalizer or TextNormalizer()
        inserted = 0
        with self._connect() as conn:
            for source, path in source_paths.items():
                batch = []
                for record in iter_normalized_tsv(path, normalizer, max_rows=max_rows):
                    batch.append((
                        source, record.entity_id, record.country, record.name_core,
                        _json_tuple(record.name_tokens), _json_tuple(record.name_sorted_tokens),
                        _json_tuple(record.name_aliases), _json_tuple(record.domains),
                        _json_tuple(record.name_char3), _json_tuple(record.name_char4),
                        record.address_street_core, _json_tuple(record.address_street_tokens),
                        _json_tuple(record.house_numbers), _json_tuple(record.postal_codes),
                        _json_tuple(record.address_char3), _json_tuple(record.address_char4),
                    ))
                    inserted += 1
                    if len(batch) >= self.config.batch_size:
                        conn.executemany(
                            "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
                        )
                        conn.commit()
                        batch.clear()
                    if inserted % self.config.progress_every == 0:
                        LOGGER.info("Feature record store: %,d records", inserted)
                if batch:
                    conn.executemany(
                        "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
                    )
                    conn.commit()
        return inserted

    def load_ground_truth(self, ground_truth_path: Path) -> int:
        inserted = 0
        with self._connect() as conn:
            conn.execute("DELETE FROM ground_truth")
            s1_in_store = {r[0] for r in conn.execute("SELECT entity_id FROM records WHERE source='S1'")}
            batch = []
            with ground_truth_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                required = {"source1_entity_id", "matched_entity_ids"}
                if set(reader.fieldnames or []) != required:
                    raise ValueError(
                        f"Unexpected ground-truth columns: {reader.fieldnames}; expected {sorted(required)}"
                    )
                for row in reader:
                    source1_id = (row.get("source1_entity_id") or "").strip()
                    if s1_in_store and source1_id not in s1_in_store:
                        continue
                    matches = [x.strip() for x in (row.get("matched_entity_ids") or "").split(",") if x.strip()]
                    for target_id in matches:
                        batch.append((source1_id, target_id))
                        inserted += 1
                        if len(batch) >= self.config.batch_size:
                            conn.executemany("INSERT OR IGNORE INTO ground_truth VALUES (?,?)", batch)
                            conn.commit()
                            batch.clear()
            if batch:
                conn.executemany("INSERT OR IGNORE INTO ground_truth VALUES (?,?)", batch)
                conn.commit()
        return inserted

    @staticmethod
    def _load_record(row: sqlite3.Row) -> dict:
        def js(value: str) -> tuple[str, ...]:
            return tuple(json.loads(value)) if value else ()
        return {
            "source": row[0], "entity_id": row[1], "country": row[2],
            "name_core": row[3], "name_tokens": js(row[4]), "name_sorted_tokens": js(row[5]),
            "name_aliases": js(row[6]), "domains": js(row[7]), "name_char3": js(row[8]),
            "name_char4": js(row[9]), "address_street_core": row[10],
            "address_street_tokens": js(row[11]), "house_numbers": js(row[12]),
            "postal_codes": js(row[13]), "address_char3": js(row[14]), "address_char4": js(row[15]),
        }

    @staticmethod
    def _similarity_features(a: dict, b: dict, hit_count: int, strategy_count: int) -> dict:
        name_a, name_b = a["name_core"], b["name_core"]
        addr_a, addr_b = a["address_street_core"], b["address_street_core"]
        countries_missing = not a["country"] or not b["country"]
        country_match = bool(a["country"] and b["country"] and a["country"] == b["country"])
        country_conflict = bool(a["country"] and b["country"] and a["country"] != b["country"])
        name_exact = bool(name_a and name_b and name_a == name_b)
        address_exact = bool(addr_a and addr_b and addr_a == addr_b)
        house_exact = bool(set(a["house_numbers"]) & set(b["house_numbers"]))
        postal_exact = bool(set(a["postal_codes"]) & set(b["postal_codes"]))
        domain_exact = bool(set(a["domains"]) & set(b["domains"]))
        alias_exact = bool(set(a["name_aliases"]) & set(b["name_aliases"]))

        name_jw = _jaro_winkler(name_a, name_b)
        name_edit = _edit_similarity(name_a, name_b)
        address_edit = _edit_similarity(addr_a, addr_b)
        name_token_j = _jaccard(a["name_tokens"], b["name_tokens"])
        name_token_c = _containment(a["name_tokens"], b["name_tokens"])
        sorted_token_j = _jaccard(a["name_sorted_tokens"], b["name_sorted_tokens"])
        addr_token_j = _jaccard(a["address_street_tokens"], b["address_street_tokens"])
        addr_token_c = _containment(a["address_street_tokens"], b["address_street_tokens"])
        domain_j = _jaccard(a["domains"], b["domains"])
        alias_j = _jaccard(a["name_aliases"], b["name_aliases"])

        strong = sum([
            name_exact, address_exact, house_exact and postal_exact,
            domain_exact, alias_exact,
        ])
        combined = sum([
            name_exact,
            address_exact,
            name_token_j >= 0.8,
            addr_token_j >= 0.8,
            house_exact,
            postal_exact,
            domain_exact,
            alias_exact,
            country_match,
        ])
        missing = sum([
            not name_a or not name_b,
            not addr_a or not addr_b,
            not a["domains"] or not b["domains"],
            not a["name_aliases"] or not b["name_aliases"],
        ])

        return {
            "country_match": int(country_match),
            "country_conflict": int(country_conflict),
            "country_missing_either": int(countries_missing),
            "name_exact": int(name_exact),
            "address_exact": int(address_exact),
            "name_address_exact": int(name_exact and address_exact),
            "house_exact": int(house_exact),
            "postal_exact": int(postal_exact),
            "domain_exact": int(domain_exact),
            "alias_exact": int(alias_exact),
            "name_nonempty_both": int(bool(name_a and name_b)),
            "address_nonempty_both": int(bool(addr_a and addr_b)),
            "domain_available_both": int(bool(a["domains"] and b["domains"])),
            "alias_available_both": int(bool(a["name_aliases"] and b["name_aliases"])),
            "name_address_both_available": int(bool(name_a and name_b and addr_a and addr_b)),
            "blocking_hit_count": hit_count,
            "blocking_strategy_count": strategy_count,
            "name_jaro_winkler": round(name_jw, 8),
            "name_edit_similarity": round(name_edit, 8),
            "name_token_jaccard": round(name_token_j, 8),
            "name_token_containment": round(name_token_c, 8),
            "name_sorted_token_jaccard": round(sorted_token_j, 8),
            "name_char3_jaccard": round(_jaccard(a["name_char3"], b["name_char3"]), 8),
            "name_char4_jaccard": round(_jaccard(a["name_char4"], b["name_char4"]), 8),
            "name_length_ratio": round(_safe_ratio(len(name_a), len(name_b)), 8),
            "address_edit_similarity": round(address_edit, 8),
            "address_token_jaccard": round(addr_token_j, 8),
            "address_token_containment": round(addr_token_c, 8),
            "address_char3_jaccard": round(_jaccard(a["address_char3"], b["address_char3"]), 8),
            "address_char4_jaccard": round(_jaccard(a["address_char4"], b["address_char4"]), 8),
            "address_length_ratio": round(_safe_ratio(len(addr_a), len(addr_b)), 8),
            "house_overlap": round(_overlap(a["house_numbers"], b["house_numbers"]), 8),
            "postal_overlap": round(_overlap(a["postal_codes"], b["postal_codes"]), 8),
            "domain_jaccard": round(domain_j, 8),
            "alias_jaccard": round(alias_j, 8),
            "combined_evidence_count": combined,
            "strong_evidence_count": strong,
            "missing_field_count": missing,
        }

    def generate_features(
        self,
        blocking_db_path: Path,
        output_path: Path,
        include_labels: bool | None = None,
    ) -> dict:
        """Stream M4 candidates through one SQLite join and write pairwise features.

        The blocking database is attached to the feature database. Candidate rows,
        S1 records, target records, blocking-hit counts, and (for train) labels are
        therefore retrieved by a single indexed SQL plan rather than one query per
        candidate pair.
        """
        include_labels = self.config.include_labels if include_labels is None else include_labels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS blk", (str(Path(blocking_db_path).resolve()),))

        columns = FeatureSchema.columns(include_labels)
        counts = Counter()
        query = """
            SELECT
                c.source1_id,
                c.target_source,
                c.target_id,
                r1.source AS s1_source,
                r1.entity_id AS s1_entity_id,
                r1.country AS s1_country,
                r1.name_core AS s1_name_core,
                r1.name_tokens AS s1_name_tokens,
                r1.name_sorted_tokens AS s1_name_sorted_tokens,
                r1.name_aliases AS s1_name_aliases,
                r1.domains AS s1_domains,
                r1.name_char3 AS s1_name_char3,
                r1.name_char4 AS s1_name_char4,
                r1.address_street_core AS s1_address_street_core,
                r1.address_street_tokens AS s1_address_street_tokens,
                r1.house_numbers AS s1_house_numbers,
                r1.postal_codes AS s1_postal_codes,
                r1.address_char3 AS s1_address_char3,
                r1.address_char4 AS s1_address_char4,
                r2.source AS target_source_db,
                r2.entity_id AS target_entity_id,
                r2.country AS target_country,
                r2.name_core AS target_name_core,
                r2.name_tokens AS target_name_tokens,
                r2.name_sorted_tokens AS target_name_sorted_tokens,
                r2.name_aliases AS target_name_aliases,
                r2.domains AS target_domains,
                r2.name_char3 AS target_name_char3,
                r2.name_char4 AS target_name_char4,
                r2.address_street_core AS target_address_street_core,
                r2.address_street_tokens AS target_address_street_tokens,
                r2.house_numbers AS target_house_numbers,
                r2.postal_codes AS target_postal_codes,
                r2.address_char3 AS target_address_char3,
                r2.address_char4 AS target_address_char4,
                COALESCE(h.hit_count, 0) AS blocking_hit_count,
                CASE WHEN gt.source1_id IS NOT NULL THEN 1 ELSE 0 END AS label
            FROM blk.candidates c
            JOIN records r1
              ON r1.source = 'S1' AND r1.entity_id = c.source1_id
            JOIN records r2
              ON r2.source = c.target_source AND r2.entity_id = c.target_id
            LEFT JOIN (
                SELECT source1_id, target_source, target_id, COUNT(*) AS hit_count
                FROM blk.candidate_hits
                GROUP BY source1_id, target_source, target_id
            ) h
              ON h.source1_id = c.source1_id
             AND h.target_source = c.target_source
             AND h.target_id = c.target_id
            LEFT JOIN ground_truth gt
              ON gt.source1_id = c.source1_id
             AND gt.target_id = c.target_id
        """
        if not include_labels:
            query = query.replace(
                "CASE WHEN gt.source1_id IS NOT NULL THEN 1 ELSE 0 END AS label,",
                "0 AS label,",
            ).replace(
                "            LEFT JOIN ground_truth gt\n              ON gt.source1_id = c.source1_id\n             AND gt.target_id = c.target_id\n",
                "",
            )

        def make_record(row: sqlite3.Row, prefix: str) -> dict:
            def js(name: str) -> tuple[str, ...]:
                value = row[name]
                return tuple(json.loads(value)) if value else ()
            return {
                "country": row[f"{prefix}_country"],
                "name_core": row[f"{prefix}_name_core"],
                "name_tokens": js(f"{prefix}_name_tokens"),
                "name_sorted_tokens": js(f"{prefix}_name_sorted_tokens"),
                "name_aliases": js(f"{prefix}_name_aliases"),
                "domains": js(f"{prefix}_domains"),
                "name_char3": js(f"{prefix}_name_char3"),
                "name_char4": js(f"{prefix}_name_char4"),
                "address_street_core": row[f"{prefix}_address_street_core"],
                "address_street_tokens": js(f"{prefix}_address_street_tokens"),
                "house_numbers": js(f"{prefix}_house_numbers"),
                "postal_codes": js(f"{prefix}_postal_codes"),
                "address_char3": js(f"{prefix}_address_char3"),
                "address_char4": js(f"{prefix}_address_char4"),
            }

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in conn.execute(query):
                a = make_record(row, "s1")
                b = make_record(row, "target")
                hit_count = int(row["blocking_hit_count"])
                feats = self._similarity_features(a, b, hit_count, hit_count)
                out = {
                    "source1_id": row["source1_id"],
                    "target_source": row["target_source"],
                    "target_id": row["target_id"],
                    **feats,
                }
                if include_labels:
                    out["label"] = int(row["label"])
                    counts["positive"] += out["label"]
                    counts["negative"] += 1 - out["label"]
                writer.writerow(out)
                counts["pairs"] += 1
                if counts["pairs"] % self.config.progress_every == 0:
                    LOGGER.info("Pairwise features: %,d candidate pairs", counts["pairs"])

        conn.close()
        return dict(counts)
