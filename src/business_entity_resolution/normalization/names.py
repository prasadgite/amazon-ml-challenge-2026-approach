"""
Multi-representation business name normalization and alias/domain extraction.
"""

import re
from typing import List, Set, Tuple, Optional
from business_entity_resolution.normalization.unicode_utils import normalize_unicode, clean_string

# Domain regex (e.g., foo.com, www.foo-bar.org)
_DOMAIN_REGEX = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|org|net|in|fr|co|io|biz|info)\b",
    re.IGNORECASE,
)

# Alias / DBA patterns
_ALIAS_SPLIT_REGEX = re.compile(
    r"\b(?:aka|d/?b/?a|f/?k/?a|doing\s+business\s+as|formerly\s+known\s+as)\b",
    re.IGNORECASE,
)

# Legal suffixes across US, India, France, and Devanagari
_LEGAL_SUFFIXES_ORDERED = [
    # Multi-word suffixes first
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt. ltd",
    "pvt limited",
    "co. ltd",
    "co. ltd.",
    "co ltd",
    "co limited",
    "& sons",
    "and sons",
    "& co",
    "and co",
    "et fils",
    "& fils",
    # Single word suffixes
    "limited",
    "private",
    "corporation",
    "incorporated",
    "corp.",
    "corp",
    "inc.",
    "inc",
    "llc.",
    "llc",
    "l.l.c.",
    "l.l.c",
    "ltd.",
    "ltd",
    "llp.",
    "llp",
    "sarl",
    "s.a.r.l.",
    "s.a.r.l",
    "sasu",
    "s.a.s.u.",
    "sas",
    "s.a.s.",
    "sci",
    "s.c.i.",
    "amicale",
    "groupe",
    # Devanagari suffixes
    "प्राइवेट लिमिटेड",
    "लिमिटेड",
    "प्राइवेट",
]

# Precompiled regex for stripping legal suffixes from end of string
_SUFFIX_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES_ORDERED) + r")\b",
    re.IGNORECASE,
)


def extract_domain(text: str) -> Optional[str]:
    """
    Detects if the business name is or contains a web domain.
    E.g., 'wilfordhancock.com' -> 'wilford hancock'
    """
    if not text:
        return None
    match = _DOMAIN_REGEX.search(text)
    if match:
        domain_base = match.group(1).replace("-", " ")
        return clean_string(domain_base)
    return None


def extract_aliases(text: str) -> List[str]:
    """
    Extracts alternative business names / DBAs.
    E.g. 'Halodelta aka Clemons Silver Eastern Inc' -> ['halodelta', 'clemons silver eastern']
    """
    if not text:
        return []
    parts = _ALIAS_SPLIT_REGEX.split(text)
    if len(parts) > 1:
        aliases = [clean_string(p) for p in parts if clean_string(p)]
        return aliases

    # Check parentheses e.g. "BS Projects (Kanpur)"
    if "(" in text and ")" in text:
        in_paren = re.findall(r"\((.*?)\)", text)
        before_paren = re.sub(r"\(.*?\)", "", text)
        res = [clean_string(before_paren)]
        for p in in_paren:
            cl = clean_string(p)
            if len(cl) > 2:
                res.append(cl)
        return [r for r in res if r]

    return []


def strip_legal_suffix(text_clean: str) -> str:
    """
    Removes common legal structures from a normalized string.
    E.g., 'reliance industries pvt ltd' -> 'reliance industries'
    """
    if not text_clean:
        return ""
    # Strip matching suffix words
    stripped = _SUFFIX_PATTERN.sub(" ", text_clean)
    # Consolidate whitespace
    return re.sub(r"\s+", " ", stripped).strip()


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extracts contiguous character n-grams from a string."""
    if not text or len(text) < n:
        return [text] if text else []
    compact = text.replace(" ", "")
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


class NormalizedName:
    """Multi-view structured container for business names."""

    __slots__ = (
        "raw",
        "clean",
        "core",
        "tokens",
        "token_sorted",
        "char3",
        "prefix4",
        "domain",
        "aliases",
    )

    def __init__(self, raw: str):
        self.raw = raw or ""
        self.clean = clean_string(self.raw)
        self.core = strip_legal_suffix(self.clean) or self.clean
        
        # Tokens
        tokens = [t for t in self.clean.split() if len(t) >= 2]
        self.tokens = tokens
        self.token_sorted = sorted(set(tokens))

        # Character n-grams & prefix
        self.char3 = extract_char_ngrams(self.core, 3)
        self.prefix4 = self.core.replace(" ", "")[:4]

        # Domain and aliases
        self.domain = extract_domain(self.raw)
        self.aliases = extract_aliases(self.raw)

    def __repr__(self) -> str:
        return f"NormalizedName(core={self.core!r}, clean={self.clean!r})"
