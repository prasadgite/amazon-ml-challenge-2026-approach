"""
Compact in-memory record definitions with minimal overhead.
"""

from typing import NamedTuple, Optional


class EntityRecord:
    """
    Memory-compact representation of a single business entity record.
    Uses __slots__ to eliminate per-instance dictionary overhead (saving >75% RAM).
    """
    __slots__ = ("entity_id", "business_name", "business_address", "country")

    def __init__(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ):
        self.entity_id = entity_id
        self.business_name = business_name
        self.business_address = business_address
        self.country = country

    @property
    def source(self) -> str:
        """Derive source prefix ('S1', 'S2', or 'S3') from entity_id."""
        if self.entity_id.startswith("S1-"):
            return "S1"
        elif self.entity_id.startswith("S2-"):
            return "S2"
        elif self.entity_id.startswith("S3-"):
            return "S3"
        return "UNKNOWN"

    def __repr__(self) -> str:
        return f"EntityRecord(id={self.entity_id}, country={self.country}, name={self.business_name!r})"
