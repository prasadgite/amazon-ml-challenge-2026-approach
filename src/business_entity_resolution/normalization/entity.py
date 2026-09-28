"""
Unified normalized entity representation combining multi-view name and address.
"""

from typing import List, Optional
from business_entity_resolution.io.record import EntityRecord
from business_entity_resolution.normalization.names import NormalizedName
from business_entity_resolution.normalization.addresses import NormalizedAddress


class NormalizedEntity:
    """
    Combines parsed multi-view name and address representations for a business entity.
    Uses __slots__ for memory efficiency across millions of records.
    """
    __slots__ = ("record", "name", "address")

    def __init__(self, record: EntityRecord):
        self.record = record
        self.name = NormalizedName(record.business_name)
        self.address = NormalizedAddress(record.business_address, record.country)

    @property
    def entity_id(self) -> str:
        return self.record.entity_id

    @property
    def country(self) -> str:
        return self.record.country

    @property
    def source(self) -> str:
        return self.record.source

    def __repr__(self) -> str:
        return f"NormalizedEntity(id={self.entity_id}, country={self.country}, name_core={self.name.core!r})"
