from __future__ import annotations

import csv
import json
import logging
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ..normalization.normalizer import NormalizedRecord, TextNormalizer, iter_normalized_tsv

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockingConfig:
    """Configuration for deterministic, bounded candidate generation."""

    name_prefix_length: int = 4
    street_prefix_length: int = 3
    max_bucket_size: int = 500
    progress_every: int = 500_000

    # M9.2 multi-channel enhancement flags
    enable_alias_blocking: bool = True
    enable_transliteration: bool = True
    enable_country_fallback: bool = True
    enable_smart_capping: bool = True
    enable_secondary_identity: bool = False


@dataclass
class BlockingStats:
    split: str
    s1_rows: int = 0
    target_rows: int = 0
    candidate_pairs: int = 0
    candidate_hits: int = 0
    capped_bucket_keys: int = 0
    capped_bucket_rows: int = 0
    empty_block_keys: int = 0
    strategy_hits: Counter = None

    def __post_init__(self) -> None:
        if self.strategy_hits is None:
            self.strategy_hits = Counter()

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "s1_rows": self.s1_rows,
            "target_rows": self.target_rows,
            "candidate_pairs": self.candidate_pairs,
            "candidate_hits": self.candidate_hits,
            "capped_bucket_keys": self.capped_bucket_keys,
            "capped_bucket_rows": self.capped_bucket_rows,
            "empty_block_keys": self.empty_block_keys,
            "strategy_hits": dict(sorted(self.strategy_hits.items())),
            "candidate_reduction_ratio": (
                1.0 - self.candidate_pairs / (self.s1_rows * self.target_rows)
                if self.s1_rows and self.target_rows else 0.0
            ),
            "mean_candidates_per_s1": (
                self.candidate_pairs / self.s1_rows if self.s1_rows else 0.0
            ),
        }


class BlockingEngine:
    """SQLite-backed multi-pass blocking engine.

    Targets are indexed once. S1 records are then streamed and queried against
    bounded block buckets. Candidate pairs are deduplicated in SQLite, so RAM
    usage does not grow with the number of candidates.
    """

    SCHEMAS = {
        "meta": """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """,
        "targets": """
            CREATE TABLE IF NOT EXISTS targets (
                source TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                country TEXT NOT NULL,
                name_core TEXT NOT NULL,
                address_street_core TEXT NOT NULL,
                house_numbers TEXT NOT NULL,
                postal_codes TEXT NOT NULL,
                domains TEXT NOT NULL,
                name_aliases TEXT NOT NULL,
                name_char4 TEXT NOT NULL,
                address_char4 TEXT NOT NULL,
                PRIMARY KEY (source, entity_id)
            )
        """,
        "blocks": """
            CREATE TABLE IF NOT EXISTS blocks (
                strategy TEXT NOT NULL,
                block_key TEXT NOT NULL,
                target_source TEXT NOT NULL,
                target_id TEXT NOT NULL,
                PRIMARY KEY (strategy, block_key, target_source, target_id)
            )
        """,
        "bucket_stats": """
            CREATE TABLE IF NOT EXISTS bucket_stats (
                strategy TEXT NOT NULL,
                block_key TEXT NOT NULL,
                bucket_size INTEGER NOT NULL,
                capped INTEGER NOT NULL,
                PRIMARY KEY (strategy, block_key)
            )
        """,
        "candidates": """
            CREATE TABLE IF NOT EXISTS candidates (
                source1_id TEXT NOT NULL,
                target_source TEXT NOT NULL,
                target_id TEXT NOT NULL,
                PRIMARY KEY (source1_id, target_source, target_id)
            )
        """,
        "candidate_hits": """
            CREATE TABLE IF NOT EXISTS candidate_hits (
                source1_id TEXT NOT NULL,
                target_source TEXT NOT NULL,
                target_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                PRIMARY KEY (source1_id, target_source, target_id, strategy)
            )
        """,
    }

    def __init__(self, db_path: Path, config: BlockingConfig | None = None) -> None:
        self.db_path = Path(db_path)
        self.config = config or BlockingConfig()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self, split: str, config_snapshot: dict | None = None) -> None:
        with self._connect() as conn:
            for ddl in self.SCHEMAS.values():
                conn.execute(ddl)
            conn.execute("DELETE FROM meta")
            conn.execute("INSERT INTO meta(key,value) VALUES (?,?)", ("split", split))
            conn.execute(
                "INSERT INTO meta(key,value) VALUES (?,?)",
                ("config", json.dumps(config_snapshot or asdict(self.config), sort_keys=True)),
            )
            conn.commit()

    @staticmethod
    def _json_tuple(values: Iterable[str]) -> str:
        return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))

    def _specific_keys_for_record(self, record: NormalizedRecord) -> Iterator[tuple[str, str]]:
        country = record.country or "__unknown__"
        prefix_n = record.blocking_name_prefix(self.config.name_prefix_length)
        if prefix_n:
            prefix_s = record.blocking_street_prefix(self.config.street_prefix_length)
            if prefix_s:
                yield "name_street_prefix_spec", f"{country}|{prefix_n}|{prefix_s}"
            if record.postal_codes:
                for pc in record.postal_codes[:1]:
                    if len(pc) >= 3:
                        yield "name_prefix_postal_spec", f"{country}|{prefix_n}|{pc[:3]}"
            for tok in record.address_tokens:
                if len(tok) >= 5 and not tok.isdigit():
                    yield "name_prefix_addr_tok_spec", f"{country}|{prefix_n}|{tok}"
                    break

    def _block_keys(self, record: NormalizedRecord) -> Iterator[tuple[str, str]]:
        country = record.country or "__unknown__"

        # 1. Primary exact keys
        if record.name_core:
            yield "name_exact_country", f"{country}|{record.name_core}"
            prefix = record.blocking_name_prefix(self.config.name_prefix_length)
            if prefix:
                yield "name_prefix_country", f"{country}|{prefix}"
            for gram in record.name_char4:
                if gram:
                    yield "name_char4_country", f"{country}|{gram}"

        if record.address_street_core:
            yield "address_exact_country", f"{country}|{record.address_street_core}"
            prefix = record.blocking_street_prefix(self.config.street_prefix_length)
            if prefix:
                yield "street_prefix_country", f"{country}|{prefix}"
            for gram in record.address_char4:
                if gram:
                    yield "address_char4_country", f"{country}|{gram}"

        if record.name_core and record.address_street_core:
            yield (
                "name_address_exact_country",
                f"{country}|{record.name_core}|{record.address_street_core}",
            )

        if record.house_numbers and record.postal_codes:
            for house in record.house_numbers:
                for postal in record.postal_codes:
                    yield "house_postal_country", f"{country}|{house}|{postal}"

        for domain in record.domains:
            yield "domain_country", f"{country}|{domain}"

        # 2. Change 3: Enhanced Alias / DBA & Domain Base keys
        if self.config.enable_alias_blocking:
            all_aliases = list(record.name_aliases) + list(getattr(record, "name_domain_names", ()))
            for alias in all_aliases:
                clean_alias = alias.strip()
                if clean_alias:
                    yield "alias_exact_country", f"{country}|{clean_alias}"
                    compact_a = "".join(clean_alias.split())
                    if len(compact_a) >= 4:
                        prefix_a = compact_a[:self.config.name_prefix_length]
                        yield "alias_prefix_country", f"{country}|{prefix_a}"
                        yield "name_prefix_country", f"{country}|{prefix_a}"
                    for tok in clean_alias.split():
                        if len(tok) >= 4:
                            yield "alias_token_country", f"{country}|{tok}"
        else:
            for alias in record.name_aliases:
                yield "alias_country", f"{country}|{alias}"

        # 3. Change 4: Cross-script transliteration keys
        if self.config.enable_transliteration:
            translit = getattr(record, "name_transliterated", "")
            if translit:
                yield "translit_exact_country", f"{country}|{translit}"
                compact_t = "".join(translit.split())
                if len(compact_t) >= 3:
                    yield "translit_prefix_country", f"{country}|{compact_t[:3]}"
                for tok in getattr(record, "name_transliterated_tokens", ()):
                    if len(tok) >= 4:
                        yield "translit_token_country", f"{country}|{tok}"
            elif record.name_core:
                yield "translit_exact_country", f"{country}|{record.name_core}"
                compact_n = "".join(record.name_core.split())
                if len(compact_n) >= 3:
                    yield "translit_prefix_country", f"{country}|{compact_n[:3]}"
                for tok in record.name_tokens:
                    if len(tok) >= 4:
                        yield "translit_token_country", f"{country}|{tok}"

        # 4. Change 5: Scoped country fallback keys
        if self.config.enable_country_fallback:
            if not record.country or record.country in {"", "unknown", "__unknown__"}:
                if record.name_core:
                    yield "name_exact_countryless", f"__any__|{record.name_core}"
            if record.house_numbers and record.postal_codes:
                for house in record.house_numbers:
                    for postal in record.postal_codes:
                        yield "house_postal_countryless", f"__any__|{house}|{postal}"

        # 5. Change 2: Smart Capping specificity sub-keys
        if self.config.enable_smart_capping:
            for spec_strat, spec_key in self._specific_keys_for_record(record):
                yield spec_strat, spec_key

        # 6. Change 6: Secondary Identity channel (B6)
        if getattr(self.config, "enable_secondary_identity", False):
            from .secondary_identity import secondary_identity_keys
            raw_name = getattr(record, "name_original", "") or record.name_clean
            for sec_key in secondary_identity_keys(raw_name):
                yield "secondary_identity_country", f"{country}|{sec_key}"


    def build_target_index(
        self,
        source_paths: dict[str, Path],
        normalizer: TextNormalizer | None = None,
        max_rows: int = 0,
    ) -> int:
        normalizer = normalizer or TextNormalizer()
        inserted = 0
        with self._connect() as conn:
            for source, path in source_paths.items():
                batch_targets = []
                batch_blocks = []
                for record in iter_normalized_tsv(path, normalizer, max_rows=max_rows):
                    batch_targets.append((
                        source, record.entity_id, record.country, record.name_core,
                        record.address_street_core, self._json_tuple(record.house_numbers),
                        self._json_tuple(record.postal_codes), self._json_tuple(record.domains),
                        self._json_tuple(record.name_aliases), self._json_tuple(record.name_char4),
                        self._json_tuple(record.address_char4),
                    ))
                    for strategy, key in self._block_keys(record):
                        batch_blocks.append((strategy, key, source, record.entity_id))
                    inserted += 1
                    if len(batch_targets) >= 10_000:
                        conn.executemany(
                            "INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            batch_targets,
                        )
                        conn.executemany(
                            "INSERT OR IGNORE INTO blocks VALUES (?,?,?,?)",
                            batch_blocks,
                        )
                        conn.commit()
                        batch_targets.clear()
                        batch_blocks.clear()
                    if inserted % self.config.progress_every == 0:
                        LOGGER.info("Indexed %,d target records", inserted)
                if batch_targets:
                    conn.executemany(
                        "INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        batch_targets,
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO blocks VALUES (?,?,?,?)",
                        batch_blocks,
                    )
                    conn.commit()

            conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_lookup ON blocks(strategy, block_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_s1 ON candidates(source1_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_target ON candidates(target_source, target_id)")
            conn.commit()

            # Bucket statistics are computed once after all target indexing.
            conn.execute("DELETE FROM bucket_stats")
            conn.execute("""
                INSERT INTO bucket_stats(strategy, block_key, bucket_size, capped)
                SELECT strategy, block_key, COUNT(*),
                       CASE WHEN COUNT(*) > ? THEN 1 ELSE 0 END
                FROM blocks
                GROUP BY strategy, block_key
            """, (self.config.max_bucket_size,))
            conn.commit()
        return inserted

    def _query_candidates(
        self,
        conn: sqlite3.Connection,
        record: NormalizedRecord,
    ) -> Iterator[tuple[str, str, str, str]]:
        """Yield unique candidate/strategy hits for one S1 record."""
        seen: set[tuple[str, str, str]] = set()
        for strategy, key in self._block_keys(record):
            row = conn.execute(
                "SELECT bucket_size, capped FROM bucket_stats WHERE strategy=? AND block_key=?",
                (strategy, key),
            ).fetchone()
            if row is None:
                continue
            bucket_size, capped = row
            if capped:
                if self.config.enable_smart_capping and strategy in {
                    "name_prefix_country", "street_prefix_country",
                    "name_char4_country", "address_char4_country",
                }:
                    for spec_strat, spec_key in self._specific_keys_for_record(record):
                        spec_row = conn.execute(
                            "SELECT bucket_size, capped FROM bucket_stats WHERE strategy=? AND block_key=?",
                            (spec_strat, spec_key),
                        ).fetchone()
                        if spec_row is not None and not spec_row[1]:
                            for target_source, target_id in conn.execute(
                                "SELECT target_source, target_id FROM blocks WHERE strategy=? AND block_key=?",
                                (spec_strat, spec_key),
                            ):
                                hit = (record.entity_id, target_source, target_id)
                                if hit not in seen:
                                    seen.add(hit)
                                    yield record.entity_id, target_source, target_id, f"{strategy}_smart_overflow"
                continue
            for target_source, target_id in conn.execute(
                "SELECT target_source, target_id FROM blocks WHERE strategy=? AND block_key=?",
                (strategy, key),
            ):
                hit = (record.entity_id, target_source, target_id)
                if hit not in seen:
                    seen.add(hit)
                    yield record.entity_id, target_source, target_id, strategy

    def generate_candidates(
        self,
        source1_path: Path,
        split: str,
        normalizer: TextNormalizer | None = None,
        max_rows: int = 0,
    ) -> BlockingStats:
        normalizer = normalizer or TextNormalizer()
        stats = BlockingStats(split=split)
        with self._connect() as conn:
            for record in iter_normalized_tsv(source1_path, normalizer, max_rows=max_rows):
                stats.s1_rows += 1
                candidate_rows: list[tuple[str, str, str, str]] = list(
                    self._query_candidates(conn, record)
                )
                if candidate_rows:
                    batch_candidates = list(dict.fromkeys(
                        (s1, target_source, target_id)
                        for s1, target_source, target_id, _strategy in candidate_rows
                    ))
                    conn.executemany(
                        "INSERT OR IGNORE INTO candidates VALUES (?,?,?)",
                        batch_candidates,
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO candidate_hits VALUES (?,?,?,?)",
                        candidate_rows,
                    )
                    for _s1, _source, _target, strategy in candidate_rows:
                        stats.strategy_hits[strategy] += 1
                    stats.candidate_hits += len(batch_candidates)
                if stats.s1_rows % self.config.progress_every == 0:
                    LOGGER.info("Generated candidates for %,d S1 records", stats.s1_rows)
            conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)", ("s1_rows", str(stats.s1_rows)))
            conn.commit()
            stats.target_rows = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            stats.candidate_pairs = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            stats.capped_bucket_keys, stats.capped_bucket_rows = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(bucket_size),0) FROM bucket_stats WHERE capped=1"
            ).fetchone()
            stats.empty_block_keys = conn.execute(
                "SELECT COUNT(*) FROM bucket_stats WHERE bucket_size=0"
            ).fetchone()[0]
        return stats

    def bucket_statistics(self) -> dict:
        """Return overall and per-strategy bucket-size diagnostics."""
        def percentiles(values: list[int]) -> dict:
            if not values:
                return {"count": 0, "min": 0, "median": 0, "p95": 0, "p99": 0, "max": 0}
            values = sorted(values)
            def pct(q: float) -> int:
                idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
                return values[idx]
            return {
                "count": len(values), "min": values[0], "median": pct(0.50),
                "p95": pct(0.95), "p99": pct(0.99), "max": values[-1],
            }

        with self._connect() as conn:
            overall = [r[0] for r in conn.execute("SELECT bucket_size FROM bucket_stats")]
            per_strategy = {}
            for strategy, in conn.execute("SELECT DISTINCT strategy FROM bucket_stats ORDER BY strategy"):
                per_strategy[strategy] = percentiles([
                    r[0] for r in conn.execute(
                        "SELECT bucket_size FROM bucket_stats WHERE strategy=?", (strategy,)
                    )
                ])
            return {"overall": percentiles(overall), "by_strategy": per_strategy}

    def write_candidates_tsv(self, output_path: Path) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn, output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("source1_entity_id", "target_source", "target_entity_id"))
            count = 0
            for row in conn.execute(
                "SELECT source1_id, target_source, target_id FROM candidates ORDER BY source1_id, target_source, target_id"
            ):
                writer.writerow(row)
                count += 1
        return count

    def write_report(self, output_path: Path, stats: BlockingStats) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(stats.as_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def close(self) -> None:
        # Connections are short-lived by design; this method exists for API symmetry.
        return None
