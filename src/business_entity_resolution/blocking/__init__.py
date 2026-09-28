"""
Blocking module for multi-pass candidate generation and recall evaluation.
"""

from business_entity_resolution.blocking.keys import (
    generate_multi_pass_keys,
    generate_blocking_keys,
    ALL_BLOCKING_STRATEGIES,
    COMMON_STOPWORDS,
)
from business_entity_resolution.blocking.index import InvertedIndex
from business_entity_resolution.blocking.sqlite_index import SQLiteBlockingIndex
from business_entity_resolution.blocking.evaluator import BlockingEvaluator

__all__ = [
    "generate_multi_pass_keys",
    "generate_blocking_keys",
    "ALL_BLOCKING_STRATEGIES",
    "COMMON_STOPWORDS",
    "InvertedIndex",
    "SQLiteBlockingIndex",
    "BlockingEvaluator",
]
