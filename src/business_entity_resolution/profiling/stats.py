"""
Streaming statistical accumulators for dataset profiling.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import math
import random


@dataclass
class RunningStats:
    """Tracks count, min, max, sum, and approximate percentiles via reservoir sampling."""
    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    reservoir_size: int = 10_000
    sample: List[float] = field(default_factory=list)

    def update(self, val: float) -> None:
        self.count += 1
        self.total += val
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

        # Reservoir sampling
        if len(self.sample) < self.reservoir_size:
            self.sample.append(val)
        else:
            idx = random.randint(0, self.count - 1)
            if idx < self.reservoir_size:
                self.sample[idx] = val

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        """Returns approximate percentile p in [0, 100]."""
        if not self.sample:
            return 0.0
        sorted_s = sorted(self.sample)
        idx = int(math.ceil((p / 100.0) * len(sorted_s))) - 1
        idx = max(0, min(idx, len(sorted_s) - 1))
        return sorted_s[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min_val if self.count > 0 else 0.0,
            "max": self.max_val if self.count > 0 else 0.0,
            "mean": round(self.mean, 2),
            "p50": round(self.percentile(50), 2),
            "p95": round(self.percentile(95), 2),
            "p99": round(self.percentile(99), 2),
        }


@dataclass
class StringStats:
    """Tracks length and token count statistics for string fields."""
    length_stats: RunningStats = field(default_factory=RunningStats)
    token_stats: RunningStats = field(default_factory=RunningStats)
    missing_count: int = 0
    empty_count: int = 0
    total_records: int = 0

    def update(self, text: str) -> None:
        self.total_records += 1
        if text is None:
            self.missing_count += 1
            return
        cleaned = text.strip()
        if not cleaned:
            self.empty_count += 1
            return

        self.length_stats.update(len(cleaned))
        token_count = len(cleaned.split())
        self.token_stats.update(token_count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "missing_count": self.missing_count,
            "missing_pct": round(self.missing_count / self.total_records * 100, 2) if self.total_records > 0 else 0.0,
            "empty_count": self.empty_count,
            "empty_pct": round(self.empty_count / self.total_records * 100, 2) if self.total_records > 0 else 0.0,
            "char_length": self.length_stats.to_dict(),
            "token_count": self.token_stats.to_dict(),
        }
