"""
High-performance Inverted Index for candidate generation.
"""

from collections import defaultdict
from typing import Dict, List, Optional

from business_entity_resolution.normalization.normalizer import NormalizedRecord
from business_entity_resolution.blocking.keys import generate_blocking_keys


class InvertedIndex:
    """
    Inverted index mapping blocking_key -> list of target entity IDs.
    Supports multi-channel indexing and query retrieval with hit counts.
    """

    def __init__(self, max_bucket_size: int = 5_000):
        # key -> list of target_ids
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.max_bucket_size = max_bucket_size
        self.total_postings = 0

    def add_target(
        self,
        rec: NormalizedRecord,
        active_channels: Optional[List[str]] = None,
    ) -> None:
        """Indexes a target record (from S2 or S3)."""
        key_dict = generate_blocking_keys(rec)
        target_id = rec.entity_id

        channels_to_use = active_channels if active_channels else key_dict.keys()
        for ch in channels_to_use:
            for k in key_dict.get(ch, []):
                # Guard against huge posting lists
                bucket = self.index[k]
                if len(bucket) < self.max_bucket_size:
                    bucket.append(target_id)
                    self.total_postings += 1

    def query_candidates(
        self,
        rec: NormalizedRecord,
        active_channels: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Retrieves all candidate target IDs for a query S1 record.
        Returns:
            Dict[target_id, blocking_hit_count]
        """
        key_dict = generate_blocking_keys(rec)
        candidates: Dict[str, int] = defaultdict(int)

        channels_to_use = active_channels if active_channels else key_dict.keys()
        for ch in channels_to_use:
            for k in key_dict.get(ch, []):
                bucket = self.index.get(k)
                if bucket:
                    for tid in bucket:
                        candidates[tid] += 1

        return dict(candidates)

    def get_bucket_statistics(self) -> Dict[str, float]:
        """Calculates bucket size distribution across the index."""
        if not self.index:
            return {"num_keys": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}

        sizes = sorted(len(v) for v in self.index.values())
        n = len(sizes)
        return {
            "num_unique_keys": n,
            "total_postings": self.total_postings,
            "mean_bucket_size": round(sum(sizes) / n, 2),
            "p50_bucket_size": sizes[int(0.50 * n)],
            "p90_bucket_size": sizes[int(0.90 * n)],
            "p95_bucket_size": sizes[int(0.95 * n)],
            "p99_bucket_size": sizes[int(0.99 * n)],
            "max_bucket_size": sizes[-1],
        }
