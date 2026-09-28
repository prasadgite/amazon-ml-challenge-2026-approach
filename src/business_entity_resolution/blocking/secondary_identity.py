from __future__ import annotations

import re


DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9-])"
    r"(?:https?://)?"
    r"(?:www\.)?"
    r"([A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z]{2,63})+)"
    r"(?![A-Za-z0-9-])",
    re.IGNORECASE,
)

ALIAS_PATTERNS = (
    re.compile(
        r"\b(?:dba|d/b/a|doing business as)\b"
        r"\s*[:\-]?\s*(.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:aka|a/k/a|also known as)\b"
        r"\s*[:\-]?\s*(.+)$",
        re.IGNORECASE,
    ),
)


def normalize_key(
    value: str,
) -> str:

    value = (
        value or ""
    ).casefold()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def extract_domain(
    value: str,
) -> str:

    match = DOMAIN_RE.search(
        value or ""
    )

    if not match:
        return ""

    return normalize_key(
        match.group(1)
    )


def extract_aliases(
    value: str,
) -> tuple[str, ...]:

    aliases: list[str] = []

    text = value or ""

    domain = DOMAIN_RE.search(text)
    domain_str = domain.group(0) if domain else ""

    for pattern in ALIAS_PATTERNS:

        match = pattern.search(text)

        if not match:
            continue

        raw_alias = match.group(1)
        if domain_str:
            raw_alias = raw_alias.replace(domain_str, "")

        alias = normalize_key(
            raw_alias
        )

        if alias:
            aliases.append(alias)

    return tuple(
        dict.fromkeys(aliases)
    )


def secondary_identity_keys(
    business_name: str,
) -> tuple[str, ...]:

    keys: list[str] = []

    domain = extract_domain(
        business_name
    )

    if domain:
        keys.append(
            f"domain:{domain}"
        )

    for alias in extract_aliases(
        business_name
    ):
        keys.append(
            f"alias:{alias}"
        )

    return tuple(
        dict.fromkeys(keys)
    )
