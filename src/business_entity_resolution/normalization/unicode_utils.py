"""
Unicode and string normalization utilities.
"""

import unicodedata
import re
from typing import Optional

# Punctuation to whitespace regex
_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_REGEX = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """
    Normalizes unicode to NFKD form and removes non-spacing mark diacritics
    (e.g., converts French 'é', 'à', 'è', 'ç', 'ô' to 'e', 'a', 'e', 'c', 'o').
    """
    if not text:
        return ""
    # Normalize unicode characters
    nfkd = unicodedata.normalize("NFKD", text)
    # Strip diacritic combining marks
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return without_accents


def clean_string(text: str) -> str:
    """
    Performs lowercasing, unicode normalization, punctuation removal,
    and whitespace consolidation.
    """
    if not text:
        return ""
    text = normalize_unicode(text.lower())
    text = _PUNCT_REGEX.sub(" ", text)
    text = _MULTI_SPACE_REGEX.sub(" ", text).strip()
    return text
