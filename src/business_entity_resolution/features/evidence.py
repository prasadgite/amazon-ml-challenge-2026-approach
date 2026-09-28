"""
M9.3 pairwise evidence features.

This module contains secondary evidence signals that are intentionally
separate from the existing primary pairwise feature set.

Design goals:
    1. Do not replace existing features.
    2. Do not change normalization semantics.
    3. Preserve deterministic behavior.
    4. Make domain -> name and fuzzy token evidence explicit.
    5. Keep the feature functions independently testable.
    6. Never use target labels while computing evidence.

These features are intended for:
    - FN diagnostics
    - feature ablation
    - later model retraining

They are NOT decision-policy rules.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Sequence


_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([^/:?#]+)",
    flags=re.IGNORECASE,
)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_evidence_text(value: object) -> str:
    """
    Conservative normalization used only by secondary evidence.

    Important:
        This does not replace the project's canonical normalizer.

    It exists so that evidence computation remains deterministic even
    when it receives raw source values.
    """
    text = _as_text(value)
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    # Remove combining marks while retaining non-Latin scripts.
    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.casefold()

    # Preserve letters/numbers from all scripts.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def evidence_tokens(value: object) -> tuple[str, ...]:
    """
    Return unique deterministic tokens.
    """
    text = normalize_evidence_text(value)

    if not text:
        return ()

    tokens = _WORD_RE.findall(text)

    # Preserve first occurrence ordering while removing duplicates.
    seen: set[str] = set()
    result: list[str] = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)

    return tuple(result)


def domain_base(value: object) -> str:
    """
    Extract a domain-like host and remove the public www prefix.

    Examples:
        https://www.example.com/path -> example.com
        www.example.com              -> example.com
        example.com                  -> example.com
    """
    text = _as_text(value).casefold()

    if not text:
        return ""

    match = _DOMAIN_RE.match(text)

    if not match:
        return ""

    host = match.group(1).strip(".")

    if host.startswith("www."):
        host = host[4:]

    return host


def domain_name_tokens(value: object) -> tuple[str, ...]:
    """
    Convert a domain into lexical tokens useful for name comparison.

    Example:
        "aggie-e-nagle.com"
            -> ("aggie", "e", "nagle")

    TLD is excluded.
    """
    domain = domain_base(value)

    if not domain:
        return ()

    host = domain.split(".", 1)[0]

    parts = re.split(r"[^a-z0-9]+", host)

    tokens: list[str] = []

    for part in parts:
        part = part.strip()

        if part:
            tokens.append(part)

    return tuple(dict.fromkeys(tokens))


def sequence_similarity(left: object, right: object) -> float:
    """
    Deterministic normalized sequence similarity.

    Returns:
        [0.0, 1.0]
    """
    a = normalize_evidence_text(left)
    b = normalize_evidence_text(right)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(
    left: Sequence[str] | Iterable[str],
    right: Sequence[str] | Iterable[str],
) -> float:
    """
    Exact token Jaccard similarity.
    """
    a = set(left)
    b = set(right)

    if not a and not b:
        return 0.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def token_containment(
    left: Sequence[str] | Iterable[str],
    right: Sequence[str] | Iterable[str],
) -> float:
    """
    Symmetric maximum containment.

    Useful when one representation contains fewer tokens than the other.
    """
    a = set(left)
    b = set(right)

    if not a or not b:
        return 0.0

    intersection = len(a & b)

    return max(
        intersection / len(a),
        intersection / len(b),
    )


def soft_token_similarity(
    left: Sequence[str] | Iterable[str],
    right: Sequence[str] | Iterable[str],
) -> float:
    """
    Fuzzy token overlap.

    For every token in the smaller token set, find its best similarity
    against the other token set.

    This deliberately uses a conservative symmetric formulation.
    """
    a = tuple(dict.fromkeys(left))
    b = tuple(dict.fromkeys(right))

    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    scores: list[float] = []

    for token_a in a:
        best = 0.0

        for token_b in b:
            score = SequenceMatcher(
                None,
                token_a,
                token_b,
            ).ratio()

            if score > best:
                best = score

        scores.append(best)

    return sum(scores) / len(scores)


def soft_token_coverage(
    left: Sequence[str] | Iterable[str],
    right: Sequence[str] | Iterable[str],
    threshold: float = 0.80,
) -> float:
    """
    Fraction of tokens from the smaller side that have a fuzzy match
    >= threshold on the other side.
    """
    a = tuple(dict.fromkeys(left))
    b = tuple(dict.fromkeys(right))

    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    covered = 0

    for token_a in a:
        best = max(
            (
                SequenceMatcher(
                    None,
                    token_a,
                    token_b,
                ).ratio()
                for token_b in b
            ),
            default=0.0,
        )

        if best >= threshold:
            covered += 1

    return covered / len(a)


def domain_to_name_similarity(
    domain: object,
    name: object,
) -> float:
    """
    Compare domain lexical tokens against business-name tokens.

    Example:

        domain:
            aggieenagle.com

        name:
            Aggie E. Nagle

    produces a high value despite the representations being structurally
    different.

    This is deliberately not an exact-match feature.
    """
    d_tokens = domain_name_tokens(domain)
    n_tokens = evidence_tokens(name)

    if not d_tokens or not n_tokens:
        return 0.0

    token_sim = soft_token_similarity(d_tokens, n_tokens)
    compact_sim = SequenceMatcher(
        None,
        "".join(d_tokens),
        "".join(n_tokens),
    ).ratio()

    return max(token_sim, compact_sim)


def domain_to_name_coverage(
    domain: object,
    name: object,
    threshold: float = 0.80,
) -> float:
    """
    Fraction of domain tokens represented in the business name.
    """
    d_tokens = domain_name_tokens(domain)
    n_tokens = evidence_tokens(name)

    if not d_tokens or not n_tokens:
        return 0.0

    compact_d = "".join(d_tokens)
    compact_n = "".join(n_tokens)

    if SequenceMatcher(None, compact_d, compact_n).ratio() >= threshold:
        return 1.0

    token_cov = soft_token_coverage(
        d_tokens,
        n_tokens,
        threshold=threshold,
    )

    return token_cov


def name_to_domain_coverage(
    name: object,
    domain: object,
    threshold: float = 0.80,
) -> float:
    """
    Fraction of business-name tokens represented by the domain.

    This is intentionally directional and complements
    domain_to_name_coverage().
    """
    n_tokens = evidence_tokens(name)
    d_tokens = domain_name_tokens(domain)

    if not n_tokens or not d_tokens:
        return 0.0

    compact_d = "".join(d_tokens)
    compact_n = "".join(n_tokens)

    if SequenceMatcher(None, compact_n, compact_d).ratio() >= threshold:
        return 1.0

    covered = 0
    for token_n in n_tokens:
        if token_n in compact_d:
            covered += 1
        else:
            best = max(
                (
                    SequenceMatcher(None, token_n, token_d).ratio()
                    for token_d in d_tokens
                ),
                default=0.0,
            )
            if best >= threshold:
                covered += 1

    return covered / len(n_tokens)



def combined_name_domain_evidence(
    name: object,
    domain: object,
) -> float:
    """
    Conservative aggregate of domain/name evidence.

    We use the maximum directional coverage rather than a simple average
    because domains often omit legal/entity tokens.

    This function is descriptive evidence only; it does not make a
    match decision.
    """
    similarity = domain_to_name_similarity(domain, name)

    d_to_n = domain_to_name_coverage(
        domain,
        name,
    )

    n_to_d = name_to_domain_coverage(
        name,
        domain,
    )

    return max(
        similarity,
        d_to_n,
        n_to_d,
    )


def secondary_evidence(
    *,
    name_left: object = "",
    name_right: object = "",
    domain_left: object = "",
    domain_right: object = "",
) -> dict[str, float]:
    """
    Compute the complete M9.3 secondary evidence vector.
    """
    left_tokens = evidence_tokens(name_left)
    right_tokens = evidence_tokens(name_right)

    return {
        "soft_token_similarity": soft_token_similarity(
            left_tokens,
            right_tokens,
        ),
        "soft_token_coverage": soft_token_coverage(
            left_tokens,
            right_tokens,
        ),
        "domain_name_similarity_left_domain": domain_to_name_similarity(
            domain_left,
            name_right,
        ),
        "domain_name_similarity_right_domain": domain_to_name_similarity(
            domain_right,
            name_left,
        ),
        "domain_name_coverage_left_domain": domain_to_name_coverage(
            domain_left,
            name_right,
        ),
        "domain_name_coverage_right_domain": domain_to_name_coverage(
            domain_right,
            name_left,
        ),
        "name_domain_coverage_left_name": name_to_domain_coverage(
            name_left,
            domain_right,
        ),
        "name_domain_coverage_right_name": name_to_domain_coverage(
            name_right,
            domain_left,
        ),
        "combined_name_domain_evidence": max(
            combined_name_domain_evidence(
                name_right,
                domain_left,
            ),
            combined_name_domain_evidence(
                name_left,
                domain_right,
            ),
        ),
    }


def evidence_column_names() -> tuple[str, ...]:
    """
    Stable schema for downstream feature generation.
    """
    return (
        "soft_token_similarity",
        "soft_token_coverage",
        "domain_name_similarity_left_domain",
        "domain_name_similarity_right_domain",
        "domain_name_coverage_left_domain",
        "domain_name_coverage_right_domain",
        "name_domain_coverage_left_name",
        "name_domain_coverage_right_name",
        "combined_name_domain_evidence",
    )
