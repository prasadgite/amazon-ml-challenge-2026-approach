"""
I/O module for high-throughput streaming dataset loading.
"""

from business_entity_resolution.io.record import EntityRecord
from business_entity_resolution.io.tsv_loader import (
    stream_records,
    stream_chunks,
    load_ground_truth,
    stream_ground_truth_pairs,
)

__all__ = [
    "EntityRecord",
    "stream_records",
    "stream_chunks",
    "load_ground_truth",
    "stream_ground_truth_pairs",
]
