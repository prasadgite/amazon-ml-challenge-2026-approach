from __future__ import annotations

import csv
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple, Dict

LOGGER = logging.getLogger(__name__)

# Corporate/legal designators. We remove them from the CORE representation but
# preserve the original CLEAN representation. This prevents loss of signal.
LEGAL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "default": (
        "incorporated", "corporation", "company", "limited", "llc", "inc",
        "corp", "ltd", "co", "plc", "llp", "lp", "sa", "ag", "gmbh",
    ),
    "us": ("incorporated", "corporation", "company", "limited", "llc", "inc", "corp", "ltd", "co"),
    "in": ("private limited", "pvt ltd", "private ltd", "limited", "llp", "ltd", "pvt", "inc"),
    "india": ("private limited", "pvt ltd", "private ltd", "limited", "llp", "ltd", "pvt", "inc"),
    "fr": ("sarl", "sasu", "sas", "sci", "sa", "fils", "groupe", "llc", "ltd", "inc", "corp", "co"),
    "france": ("sarl", "sasu", "sas", "sci", "sa", "fils", "groupe", "llc", "ltd", "inc", "corp", "co"),
}

STREET_TERMS = {
    "street": "st",
    "st": "st",
    "st.": "st",
    "road": "rd",
    "rd": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln": "ln",
    "ln.": "ln",
    "roadway": "rd",
    "highway": "hwy",
    "hwy": "hwy",
    "parkway": "pkwy",
    "pkwy": "pkwy",
    "place": "pl",
    "pl": "pl",
    "square": "sq",
    "sq": "sq",
    "terrace": "ter",
    "ter": "ter",
    "rue": "rue",
    "r": "rue",
    "route": "rte",
    "rte": "rte",
    "chemin": "che",
    "ch": "che",
}

# Explicit aliases that occur in business records. We only treat these as
# aliases when the marker is clearly present; otherwise text remains intact.
ALIAS_MARKER_RE = re.compile(
    r"\b(?:dba|d/b/a|doing\s+business\s+as|aka|a/k/a|trading\s+as|t/a)\b\s*[:\-]?\s*",
    re.IGNORECASE,
)


DOMAIN_RE = re.compile(
    r"(?<![\w@])(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9.-]*\.[a-z]{2,})(?![\w.-])",
    re.IGNORECASE,
)

# A conservative house-number pattern. It handles 0189, 12-A, 12/14 and
# common "No. 4" forms without attempting to interpret every address grammar.
HOUSE_NUMBER_RE = re.compile(
    r"(?<!\w)(?:no\.?\s*)?(\d{1,7})(?:\s*([-/])\s*(\d{1,7}))?(?:\s*[a-z])?(?!\w)",
    re.IGNORECASE,
)

# Postal-code patterns are intentionally conservative. A false postal code is
# worse for blocking than a missing one, so unsupported formats are ignored.
POSTAL_PATTERNS = (
    re.compile(r"(?<!\d)\d{5,6}(?!\d)"),              # US/France 5-digit, India 6-digit
    re.compile(r"(?<![A-Z0-9])\d{5}-\d{4}(?!\d)"),  # US ZIP+4
    re.compile(r"(?<![A-Z0-9])\d{3}\s?\d{2}(?!\d)"),  # some country variants
    re.compile(r"(?<![A-Z0-9])[A-Z]\d[A-Z]\s?\d[A-Z]\d(?!\w)"),  # Canada
)

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def _ascii_fold(value: str) -> str:
    """NFKD + remove combining marks; keeps non-Latin scripts intact."""
    value = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _normalize_unicode(value: str) -> str:
    value = _ascii_fold(value.casefold())
    # Keep letters/numbers; convert punctuation and symbols to spaces.
    chars = []
    for ch in value:
        category = unicodedata.category(ch)
        if category[0] in {"L", "N"}:
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(value))


def _dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _char_ngrams(value: str, n: int = 3) -> tuple[str, ...]:
    compact = value.replace(" ", "")
    if len(compact) < n:
        return (compact,) if compact else ()
    return tuple(sorted({compact[i:i+n] for i in range(len(compact) - n + 1)}))


def _canonical_country(country: str) -> str:
    c = _normalize_unicode(country)
    aliases = {
        "united states": "us", "usa": "us", "u s a": "us", "us": "us",
        "united states of america": "us",
        "india": "in", "in": "in",
        "france": "fr", "fr": "fr",
    }
    return aliases.get(c, c)


def canonicalize_country(raw_country: str) -> str:
    return _canonical_country(raw_country)


def _strip_legal_suffixes(tokens: tuple[str, ...], country: str) -> tuple[str, ...]:
    """Strip legal designators from the END of a business name only."""
    suffixes = LEGAL_SUFFIXES.get(country, LEGAL_SUFFIXES["default"])
    normalized_suffixes = [tuple(_tokens(_normalize_unicode(s))) for s in suffixes]
    current = list(tokens)

    changed = True
    while current and changed:
        changed = False
        for suffix in sorted(normalized_suffixes, key=len, reverse=True):
            if suffix and tuple(current[-len(suffix):]) == suffix:
                del current[-len(suffix):]
                changed = True
                break
    return tuple(current)


def _normalize_street_tokens(tokens: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for token in tokens:
        token = STREET_TERMS.get(token, token)
        if token.isdigit():
            token = str(int(token))
        normalized.append(token)
    return tuple(normalized)


def _extract_house_numbers(address_clean: str) -> tuple[str, ...]:
    numbers = []
    postal_spans = []
    for pattern in POSTAL_PATTERNS:
        postal_spans.extend(m.span() for m in pattern.finditer(address_clean.upper()))

    def overlaps_postal(start: int, end: int) -> bool:
        return any(start < p_end and end > p_start for p_start, p_end in postal_spans)

    for match in HOUSE_NUMBER_RE.finditer(address_clean):
        if overlaps_postal(*match.span()):
            continue
        g1, sep, g2 = match.group(1), match.group(2), match.group(3)
        if g1 and g2 and sep == "/":
            numbers.append(f"{int(g1)}/{int(g2)}")
            numbers.append(str(int(g1)))
            numbers.append(str(int(g2)))
        elif g1 and g2 and sep == "-":
            numbers.append(str(int(g1)))
            numbers.append(str(int(g2)))
        elif g1:
            numbers.append(str(int(g1)))
    return _dedupe_preserve_order(numbers)


def _extract_postal_codes(address_clean: str) -> tuple[str, ...]:
    codes = []
    upper = address_clean.upper()
    for pattern in POSTAL_PATTERNS:
        codes.extend(m.group(0) for m in pattern.finditer(upper))
    return _dedupe_preserve_order(code.replace(" ", "") for code in codes)


def _extract_domains(original: str) -> tuple[str, ...]:
    domains = []
    for match in DOMAIN_RE.finditer(original):
        domain = match.group(1).casefold().strip(".")
        if domain:
            domains.append(domain)
    return _dedupe_preserve_order(domains)


def _domain_base_names(domains: Iterable[str]) -> tuple[str, ...]:
    result = []
    for domain in domains:
        labels = domain.split(".")
        if len(labels) >= 2:
            base = labels[-2]
            result.append(_normalize_unicode(base))
    return _dedupe_preserve_order(result)


BRAHMIC_MAP = {
    0x01: "n", 0x02: "m", 0x03: "h",
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "au", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3C: "", # nukta
    0x3E: "aa", 0x3F: "i", 0x40: "ee", 0x41: "u", 0x42: "oo", 0x43: "ri",
    0x47: "e", 0x48: "ai", 0x4B: "o", 0x4C: "au",
    0x4D: "", # halant / virama
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
    0x70: "", 0x71: "",
}

CYRILLIC_MAP = {
    0x0410: "A", 0x0411: "B", 0x0412: "V", 0x0413: "G", 0x0414: "D", 0x0415: "E", 0x0401: "Yo",
    0x0416: "Zh", 0x0417: "Z", 0x0418: "I", 0x0419: "Y", 0x041A: "K", 0x041B: "L", 0x041C: "M",
    0x041D: "N", 0x041E: "O", 0x041F: "P", 0x0420: "R", 0x0421: "S", 0x0422: "T", 0x0423: "U",
    0x0424: "F", 0x0425: "Kh", 0x0426: "Ts", 0x0427: "Ch", 0x0428: "Sh", 0x0429: "Shch",
    0x042B: "Y", 0x042D: "E", 0x042E: "Yu", 0x042F: "Ya",
    0x0430: "a", 0x0431: "b", 0x0432: "v", 0x0433: "g", 0x0434: "d", 0x0435: "e", 0x0451: "yo",
    0x0436: "zh", 0x0437: "z", 0x0438: "i", 0x0439: "y", 0x043A: "k", 0x043B: "l", 0x043C: "m",
    0x043D: "n", 0x043E: "o", 0x043F: "p", 0x0440: "r", 0x0441: "s", 0x0442: "t", 0x0443: "u",
    0x0444: "f", 0x0445: "kh", 0x0446: "ts", 0x0447: "ch", 0x0448: "sh", 0x0449: "shch",
    0x044B: "y", 0x044D: "e", 0x044E: "yu", 0x044F: "ya",
}


def _transliterate_text(text: str) -> str:
    """Deterministic script-aware transliteration for Indic (Brahmic) and Cyrillic scripts."""
    if not text:
        return ""
    has_non_latin = False
    for ch in text:
        cp = ord(ch)
        if (0x0900 <= cp <= 0x0D7F) or (0x0400 <= cp <= 0x04FF):
            has_non_latin = True
            break
    if not has_non_latin:
        return text

    chars = []
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x0D7F:
            offset = (cp - 0x0900) % 0x80
            chars.append(BRAHMIC_MAP.get(offset, ""))
        elif 0x0400 <= cp <= 0x04FF:
            chars.append(CYRILLIC_MAP.get(cp, ""))
        else:
            chars.append(ch)
    return "".join(chars)


def _extract_aliases(original_name: str, clean_name: str, country: str = "default") -> tuple[str, ...]:
    if not ALIAS_MARKER_RE.search(original_name):
        return ()
    parts = ALIAS_MARKER_RE.split(original_name)
    if len(parts) < 2:
        return ()
    aliases = []
    # Primary name before DBA marker
    primary_part = DOMAIN_RE.sub(" ", parts[0])
    norm_primary = _normalize_unicode(primary_part)
    if norm_primary:
        aliases.append(norm_primary)
        core_p = " ".join(_strip_legal_suffixes(_tokens(norm_primary), country))
        if core_p and core_p != norm_primary:
            aliases.append(core_p)

    for part in parts[1:]:
        part_without_domains = DOMAIN_RE.sub(" ", part)
        normalized = _normalize_unicode(part_without_domains)
        if normalized:
            aliases.append(normalized)
            core_a = " ".join(_strip_legal_suffixes(_tokens(normalized), country))
            if core_a and core_a != normalized:
                aliases.append(core_a)
    return _dedupe_preserve_order(aliases)


@dataclass(frozen=True)
class NormalizedRecord:
    entity_id: str
    country: str
    name_original: str
    address_original: str

    # Business-name representations.
    name_clean: str
    name_core: str
    name_tokens: tuple[str, ...]
    name_sorted_tokens: tuple[str, ...]
    name_aliases: tuple[str, ...]
    name_domain_names: tuple[str, ...]
    domains: tuple[str, ...]
    name_char3: tuple[str, ...]
    name_char4: tuple[str, ...]

    # Address representations.
    address_clean: str
    address_tokens: tuple[str, ...]
    address_street_tokens: tuple[str, ...]
    address_street_core: str
    house_numbers: tuple[str, ...]
    postal_codes: tuple[str, ...]
    address_char3: tuple[str, ...]
    address_char4: tuple[str, ...]

    # M9.2 transliterated representations (blocking only).
    name_transliterated: str = ""
    name_transliterated_tokens: tuple[str, ...] = ()
    name_transliterated_char3: tuple[str, ...] = ()
    name_transliterated_char4: tuple[str, ...] = ()

    def to_json_dict(self) -> dict:
        return asdict(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def blocking_name_prefix(self, n: int = 4) -> str:
        compact = "".join(self.name_core.split())
        return compact[:n]

    def blocking_street_prefix(self, n: int = 3) -> str:
        compact = "".join(self.address_street_core.split())
        return compact[:n]


class TextNormalizer:
    """Deterministic normalizer used by all subsequent pipeline stages."""

    def normalize_name(self, value: str, country: str) -> dict:
        original = value or ""
        clean = _normalize_unicode(original)
        domains = _extract_domains(original)
        name_without_domains = DOMAIN_RE.sub(" ", original)
        name_without_domains_clean = _normalize_unicode(name_without_domains)
        tokens = _tokens(name_without_domains_clean)
        core_tokens = _strip_legal_suffixes(tokens, country)
        core = " ".join(core_tokens)
        aliases = _extract_aliases(original, clean, country)
        domain_names = _domain_base_names(domains)

        # M9.2: Script-aware transliteration for cross-script blocking
        raw_transliterated = _transliterate_text(original)
        translit_clean = _normalize_unicode(raw_transliterated) if raw_transliterated != original else ""
        if translit_clean and translit_clean != clean:
            translit_tokens = _strip_legal_suffixes(_tokens(translit_clean), country)
            translit_core = " ".join(translit_tokens)
            translit_c3 = _char_ngrams(translit_core, 3)
            translit_c4 = _char_ngrams(translit_core, 4)
        else:
            translit_core = ""
            translit_tokens = ()
            translit_c3 = ()
            translit_c4 = ()

        return {
            "original": original,
            "clean": clean,
            "core": core,
            "tokens": core_tokens,
            "sorted_tokens": tuple(sorted(core_tokens)),
            "aliases": aliases,
            "domains": domains,
            "domain_names": domain_names,
            "char3": _char_ngrams(core, 3),
            "char4": _char_ngrams(core, 4),
            "transliterated": translit_core,
            "transliterated_tokens": translit_tokens,
            "transliterated_char3": translit_c3,
            "transliterated_char4": translit_c4,
        }

    def normalize_address(self, value: str) -> dict:
        original = value or ""
        clean = _normalize_unicode(original)
        raw_tokens = _tokens(clean)
        street_tokens = _normalize_street_tokens(raw_tokens)
        street_core = " ".join(street_tokens)
        return {
            "original": original,
            "clean": " ".join(street_tokens),
            "tokens": street_tokens,
            "street_tokens": street_tokens,
            "street_core": street_core,
            "house_numbers": _extract_house_numbers(original),
            "postal_codes": _extract_postal_codes(original),
            "char3": _char_ngrams(street_core, 3),
            "char4": _char_ngrams(street_core, 4),
        }

    def normalize(
        self,
        entity_id_or_record: Any,
        business_name: Optional[str] = None,
        business_address: Optional[str] = None,
        country: Optional[str] = None,
    ) -> NormalizedRecord:
        if hasattr(entity_id_or_record, "entity_id"):
            entity_id = entity_id_or_record.entity_id
            b_name = getattr(entity_id_or_record, "business_name", "") or ""
            b_addr = getattr(entity_id_or_record, "business_address", "") or ""
            b_country = getattr(entity_id_or_record, "country", "") or ""
        else:
            entity_id = str(entity_id_or_record)
            b_name = business_name or ""
            b_addr = business_address or ""
            b_country = country or ""

        country_key = _canonical_country(b_country)
        name = self.normalize_name(b_name, country_key)
        address = self.normalize_address(b_addr)

        return NormalizedRecord(
            entity_id=entity_id,
            country=country_key,
            name_original=b_name,
            address_original=b_addr,
            name_clean=name["clean"],
            name_core=name["core"],
            name_tokens=name["tokens"],
            name_sorted_tokens=name["sorted_tokens"],
            name_aliases=name["aliases"],
            name_domain_names=name["domain_names"],
            domains=name["domains"],
            name_char3=name["char3"],
            name_char4=name["char4"],
            address_clean=address["clean"],
            address_tokens=address["tokens"],
            address_street_tokens=address["street_tokens"],
            address_street_core=address["street_core"],
            house_numbers=address["house_numbers"],
            postal_codes=address["postal_codes"],
            address_char3=address["char3"],
            address_char4=address["char4"],
            name_transliterated=name.get("transliterated", ""),
            name_transliterated_tokens=name.get("transliterated_tokens", ()),
            name_transliterated_char3=name.get("transliterated_char3", ()),
            name_transliterated_char4=name.get("transliterated_char4", ()),
        )


_DEFAULT_NORMALIZER = TextNormalizer()


def normalize_record(
    entity_id: str,
    business_name: str,
    business_address: str,
    country: str,
) -> NormalizedRecord:
    return _DEFAULT_NORMALIZER.normalize(entity_id, business_name, business_address, country)


def iter_normalized_tsv(
    file_path: Path,
    normalizer: Any = None,
    country: Optional[str] = None,
    max_rows: int = 0,
) -> Iterator[NormalizedRecord]:
    norm = None
    c_filter = country
    if isinstance(normalizer, TextNormalizer):
        norm = normalizer
    elif isinstance(normalizer, str):
        c_filter = normalizer
    norm = norm or _DEFAULT_NORMALIZER

    with Path(file_path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(f"Unexpected TSV schema in {file_path}: {reader.fieldnames}")
        count = 0
        for row in reader:
            row_country = row["country"].strip()
            if c_filter and row_country != c_filter:
                continue
            yield norm.normalize(
                row["entity_id"].strip(),
                row["business_name"].strip(),
                row["business_address"].strip(),
                row_country,
            )
            count += 1
            if max_rows > 0 and count >= max_rows:
                break


def write_normalized_jsonl(
    records: Iterable[NormalizedRecord],
    output_path: Path,
    limit: int | None = None,
) -> int:
    """Optional diagnostic export. Not required by later stages."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def export_normalization_samples(
    paths,
    sample_size: int = 100,
    logger=None,
) -> None:
    """Exports sample normalized records to artifacts/diagnostics/normalization_samples/."""
    out_dir = paths.diagnostics_dir / "normalization_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = TextNormalizer()

    targets = [
        ("train_source1.jsonl", paths.train_source1),
        ("train_source2.jsonl", paths.train_source2),
        ("train_source3.jsonl", paths.train_source3),
        ("test_source1.jsonl", paths.test_source1),
        ("test_source2.jsonl", paths.test_source2),
        ("test_source3.jsonl", paths.test_source3),
    ]

    for fname, fpath in targets:
        out_path = out_dir / fname
        count = write_normalized_jsonl(
            iter_normalized_tsv(fpath, normalizer=normalizer),
            out_path,
            limit=sample_size,
        )
        if logger:
            logger.info("Exported %d normalized sample records to %s", count, out_path.name)
