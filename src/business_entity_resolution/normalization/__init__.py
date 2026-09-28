"""
Normalization module providing multi-view representations of business names and addresses.
"""

from business_entity_resolution.normalization.unicode_utils import normalize_unicode, clean_string
from business_entity_resolution.normalization.names import (
    NormalizedName,
    strip_legal_suffix,
    extract_aliases,
    extract_domain,
    extract_char_ngrams,
)
from business_entity_resolution.normalization.addresses import (
    NormalizedAddress,
    extract_house_numbers,
    extract_postal_code,
    standardize_street_terms,
    normalize_house_number,
)
from business_entity_resolution.normalization.entity import NormalizedEntity
from business_entity_resolution.normalization.normalizer import (
    NormalizedRecord,
    TextNormalizer,
    iter_normalized_tsv,
    export_normalization_samples,
    canonicalize_country,
)

__all__ = [
    "normalize_unicode",
    "clean_string",
    "NormalizedName",
    "strip_legal_suffix",
    "extract_aliases",
    "extract_domain",
    "extract_char_ngrams",
    "NormalizedAddress",
    "extract_house_numbers",
    "extract_postal_code",
    "standardize_street_terms",
    "normalize_house_number",
    "NormalizedEntity",
    "NormalizedRecord",
    "TextNormalizer",
    "iter_normalized_tsv",
    "export_normalization_samples",
    "canonicalize_country",
]
