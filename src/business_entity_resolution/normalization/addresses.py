"""
Multi-representation address normalization, number extraction, and component parsing.
"""

import re
from typing import List, Set, Optional, Tuple
from business_entity_resolution.normalization.unicode_utils import normalize_unicode, clean_string

# Street abbreviation dictionary
_STREET_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave?\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bfl\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bno\b": "number",
    # French abbreviations
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\ball\b": "allee",
    r"\bav\b": "avenue",
}


def normalize_house_number(raw_num: str) -> str:
    """Removes leading zeroes: '0189' -> '189', '004' -> '4'."""
    if not raw_num:
        return ""
    cleaned = re.sub(r"[^\d/-]", "", raw_num).strip()
    if "/" in cleaned:
        parts = cleaned.split("/")
        return "/".join(str(int(p)) if p.isdigit() else p for p in parts)
    if "-" in cleaned:
        parts = cleaned.split("-")
        return "-".join(str(int(p)) if p.isdigit() else p for p in parts)
    if cleaned.isdigit():
        return str(int(cleaned))
    return cleaned

def extract_house_numbers(text: str) -> List[str]:
    """Extracts all candidate house/street numbers from an address string."""
    if not text:
        return []
    tokens = re.findall(r"\b\d+(?:[/-]\d+)?\b", text)
    res: List[str] = []
    for t in tokens:
        # Ignore postal codes (5 or 6 digits without slash/dash)
        if len(t) in (5, 6) and "/" not in t and "-" not in t:
            continue
        # Strip leading zeros
        if t.isdigit():
            norm = str(int(t))
        elif "/" in t:
            norm = "/".join(str(int(p)) if p.isdigit() else p for p in t.split("/"))
        elif "-" in t:
            norm = "-".join(str(int(p)) if p.isdigit() else p for p in t.split("-"))
        else:
            norm = t
        if norm and norm not in res:
            res.append(norm)
    return res


# Postal code patterns
_PIN_INDIA_REGEX = re.compile(r"\b[1-9]\d{5}\b")  # 6-digit Indian PIN code
_ZIP_US_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")  # 5-digit US ZIP (+ optional 4)
_POSTAL_FR_REGEX = re.compile(r"\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b")  # 5-digit French postal code


def extract_postal_code(text: str, country: str = "") -> Optional[str]:
    """Extracts country-appropriate postal code."""
    if not text:
        return None
    c_upper = country.strip().upper() if country else ""
    if c_upper == "INDIA":
        m = _PIN_INDIA_REGEX.search(text)
        return m.group(0) if m else None
    elif c_upper == "FRANCE":
        m = _POSTAL_FR_REGEX.search(text)
        return m.group(0) if m else None
    elif c_upper == "US":
        m = _ZIP_US_REGEX.search(text)
        return m.group(0)[:5] if m else None
    else:
        # Generic fallback
        m = re.search(r"\b\d{5,6}\b", text)
        return m.group(0) if m else None


def standardize_street_terms(text: str) -> str:
    """Standardizes street abbreviations to full words."""
    result = text
    for pattern, repl in _STREET_REPLACEMENTS.items():
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)
    return result


class NormalizedAddress:
    """Multi-view structured container for business addresses."""

    __slots__ = (
        "raw",
        "clean",
        "tokens",
        "house_numbers",
        "postal_code",
        "street_core",
        "char3",
    )

    def __init__(self, raw: str, country: str = ""):
        self.raw = raw or ""
        cl = clean_string(self.raw)
        self.clean = standardize_street_terms(cl)
        
        # Numbers
        self.house_numbers = extract_house_numbers(self.raw)
        self.postal_code = extract_postal_code(self.raw, country)

        # Tokens
        tokens = [t for t in self.clean.split() if len(t) >= 2]
        self.tokens = tokens

        # Street core (first 2-3 non-number tokens)
        non_num_tokens = [t for t in tokens if not t.isdigit() and len(t) >= 3]
        self.street_core = " ".join(non_num_tokens[:3]) if non_num_tokens else ""

        # Character 3-grams
        compact = self.clean.replace(" ", "")
        self.char3 = [compact[i : i + 3] for i in range(len(compact) - 2)] if len(compact) >= 3 else []

    def __repr__(self) -> str:
        return f"NormalizedAddress(clean={self.clean!r}, numbers={self.house_numbers}, postal={self.postal_code})"
