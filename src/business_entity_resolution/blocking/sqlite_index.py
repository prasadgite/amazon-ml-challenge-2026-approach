"""
SQLite-backed blocking and inverted index engine for scalable multi-pass candidate generation.
Avoids holding millions of posting lists in Python RAM.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable, Union
import json
import logging
import statistics

from business_entity_resolution.normalization.normalizer import NormalizedRecord
from business_entity_resolution.blocking.keys import generate_multi_pass_keys

logger = logging.getLogger("SQLiteBlockingIndex")


class SQLiteBlockingIndex:
    """
    Disk or memory-backed SQLite inverted index for candidate generation.
    Stores postings as (block_key, entity_id, source, strategy) with fast B-tree lookups.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_database()

    def _setup_database(self) -> None:
        """Configures high-speed pragmas and initializes index tables."""
        cur = self.conn.cursor()
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        cur.execute("PRAGMA cache_size = -64000;")  # 64MB cache

        cur.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                block_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                source TEXT NOT NULL,
                strategy TEXT NOT NULL,
                PRIMARY KEY (block_key, entity_id, strategy)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_key ON postings (block_key);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_postings_entity ON postings (entity_id);")
        self.conn.commit()

    def add_target(self, rec: NormalizedRecord) -> None:
        """Indexes a single target record across all 10 multi-pass blocking strategies."""
        keys_by_strat = generate_multi_pass_keys(rec)
        source = rec.entity_id[:2] if len(rec.entity_id) >= 2 else "UNKNOWN"
        rows = []
        for strat, keys in keys_by_strat.items():
            for k in keys:
                rows.append((k, rec.entity_id, source, strat))

        if rows:
            self.conn.executemany(
                "INSERT OR IGNORE INTO postings (block_key, entity_id, source, strategy) VALUES (?, ?, ?, ?)",
                rows,
            )
            self.conn.commit()

    def bulk_insert_targets(
        self,
        records: Iterable[NormalizedRecord],
        batch_size: int = 10_000,
    ) -> int:
        """
        Streams and indexes target records in batches for high throughput.
        Returns total postings inserted.
        """
        total_postings = 0
        batch: List[tuple] = []
        cur = self.conn.cursor()

        for rec in records:
            source = rec.entity_id[:2] if len(rec.entity_id) >= 2 else "UNKNOWN"
            keys_by_strat = generate_multi_pass_keys(rec)
            for strat, keys in keys_by_strat.items():
                for k in keys:
                    batch.append((k, rec.entity_id, source, strat))

            if len(batch) >= batch_size:
                cur.executemany(
                    "INSERT OR IGNORE INTO postings (block_key, entity_id, source, strategy) VALUES (?, ?, ?, ?)",
                    batch,
                )
                total_postings += len(batch)
                batch.clear()

        if batch:
            cur.executemany(
                "INSERT OR IGNORE INTO postings (block_key, entity_id, source, strategy) VALUES (?, ?, ?, ?)",
                batch,
            )
            total_postings += len(batch)

        self.conn.commit()
        return total_postings

    def get_bucket_statistics(self, max_bucket_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculates bucket statistics: min, median, P95, P99, max, and capped bucket count.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT entity_id) FROM postings;")
        total_entities = cur.fetchone()[0] or 0

        cur.execute("SELECT COUNT(*) FROM postings;")
        total_postings = cur.fetchone()[0] or 0

        # Query sizes for all block keys
        cur.execute("SELECT COUNT(DISTINCT entity_id) AS sz FROM postings GROUP BY block_key;")
        sizes = [row[0] for row in cur.fetchall()]

        if not sizes:
            return {
                "total_unique_keys": 0,
                "total_postings": 0,
                "total_indexed_entities": total_entities,
                "min": 0,
                "median": 0,
                "p95": 0,
                "p99": 0,
                "max": 0,
                "capped_keys": 0,
                "max_bucket_size": max_bucket_size,
            }

        sizes.sort()
        n = len(sizes)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)

        capped_count = sum(1 for s in sizes if s > max_bucket_size) if max_bucket_size else 0

        return {
            "total_unique_keys": n,
            "total_postings": total_postings,
            "total_indexed_entities": total_entities,
            "min": sizes[0],
            "median": round(statistics.median(sizes), 2),
            "p95": sizes[p95_idx],
            "p99": sizes[p99_idx],
            "max": sizes[-1],
            "capped_keys": capped_count,
            "max_bucket_size": max_bucket_size,
        }

    def query_candidates(
        self,
        query_rec: NormalizedRecord,
        max_bucket_size: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Queries all candidate targets for a query record across all 10 passes.
        Applies max_bucket_size cap to prevent candidate explosion.
        Returns deduplicated candidates with hits count and strategy provenance:
        {target_id: {'hits': count, 'strategies': [strategy_names]}}
        """
        keys_by_strat = generate_multi_pass_keys(query_rec)
        candidates: Dict[str, Dict[str, Any]] = {}
        cur = self.conn.cursor()

        for strat, keys in keys_by_strat.items():
            for key in keys:
                cur.execute(
                    "SELECT entity_id, strategy FROM postings WHERE block_key = ?",
                    (key,),
                )
                rows = cur.fetchall()

                # Cap bucket size: skip if bucket exceeds cap
                if len(rows) > max_bucket_size:
                    continue

                for row in rows:
                    target_id = row["entity_id"]
                    matched_strat = row["strategy"]
                    if target_id not in candidates:
                        candidates[target_id] = {
                            "hits": 0,
                            "strategies": set(),
                        }
                    candidates[target_id]["hits"] += 1
                    candidates[target_id]["strategies"].add(matched_strat)

        # Convert set of strategies to sorted list for serialization
        for c in candidates.values():
            c["strategies"] = sorted(list(c["strategies"]))

        return candidates

    def save_diagnostics(self, out_json_path: Path, max_bucket_size: int = 1000) -> Dict[str, Any]:
        """Exports bucket statistics to JSON."""
        stats = self.get_bucket_statistics(max_bucket_size=max_bucket_size)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        return stats

    def close(self) -> None:
        """Closes SQLite connection."""
        try:
            self.conn.close()
        except Exception:
            pass
