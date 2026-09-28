"""
Multi-channel and multi-pass blocking key generators with dynamic token suppression.
Implements the 10 distinct blocking passes for Milestone 4.
"""

from typing import List, Set, Dict
from business_entity_resolution.normalization.normalizer import NormalizedRecord

# High-frequency generic name stopwords across US, India, and France
COMMON_NAME_STOPWORDS: Set[str] = {
    "limited", "private", "llc", "inc", "ltd", "pvt", "corp", "corporation",
    "services", "service", "center", "solutions", "enterprises", "enterprise",
    "international", "group", "holdings", "company", "and", "the", "for", "with",
    # Hindi / Devanagari common terms
    "लिमिटेड", "प्राइवेट", "कंपनी", "एंटरप्राइजेज",
    # French corporate & general
    "sarl", "sasu", "sas", "sci", "sa", "amicale", "groupe", "france", "paris",
}

# High-frequency generic address stopwords
COMMON_ADDR_STOPWORDS: Set[str] = {
    "road", "street", "drive", "avenue", "lane", "court", "floor", "unit",
    "suite", "north", "south", "east", "west", "near", "delhi", "maharashtra",
    "karnataka", "texas", "california", "new", "number", "shop", "block", "sector",
    "india", "france", "states",
}

COMMON_STOPWORDS = COMMON_NAME_STOPWORDS

# All 10 distinct multi-pass blocking strategies
ALL_BLOCKING_STRATEGIES = [
    "exact_name",
    "exact_addr",
    "exact_name_addr",
    "name_prefix",
    "street_prefix",
    "name_4grams",
    "addr_4grams",
    "hn_postal",
    "domain",
    "dba_alias",
]


def generate_multi_pass_keys(
    rec: NormalizedRecord,
    suppress_stopwords: bool = True,
) -> Dict[str, List[str]]:
    """
    Generates keys for the 10 distinct blocking passes:
      1. exact normalized name + country
      2. exact address + country
      3. exact name + address + country
      4. name prefix + country
      5. street prefix + country
      6. name 4-grams
      7. address 4-grams
      8. house number + postal code
      9. domain + country
      10. DBA/AKA alias + country
    """
    country = rec.country
    keys: Dict[str, List[str]] = {strat: [] for strat in ALL_BLOCKING_STRATEGIES}

    # Pass 1: exact normalized name + country
    if rec.name_clean:
        keys["exact_name"].append(f"{country}_en_{rec.name_clean}")
    if rec.name_core and rec.name_core != rec.name_clean:
        keys["exact_name"].append(f"{country}_en_{rec.name_core}")

    # Pass 2: exact address + country
    if rec.address_clean:
        keys["exact_addr"].append(f"{country}_ea_{rec.address_clean}")
    if rec.address_street_core and rec.address_street_core != rec.address_clean:
        keys["exact_addr"].append(f"{country}_ea_{rec.address_street_core}")

    # Pass 3: exact name + address + country
    if rec.name_clean and rec.address_clean:
        keys["exact_name_addr"].append(f"{country}_ena_{rec.name_clean}_{rec.address_clean}")

    # Pass 4: name prefix + country
    name_compact = (rec.name_core or rec.name_clean).replace(" ", "")
    if len(name_compact) >= 4:
        keys["name_prefix"].append(f"{country}_np_{name_compact[:6]}")

    # Pass 5: street prefix + country
    street_compact = (rec.address_street_core or rec.address_clean).replace(" ", "")
    if len(street_compact) >= 4:
        keys["street_prefix"].append(f"{country}_sp_{street_compact[:8]}")

    # Pass 6: name 4-grams
    for g in rec.name_char4:
        keys["name_4grams"].append(f"{country}_n4g_{g}")

    # Pass 7: address 4-grams
    for g in rec.address_char4:
        keys["addr_4grams"].append(f"{country}_a4g_{g}")

    # Pass 8: house number + postal code
    for hn in rec.house_numbers:
        for pc in rec.postal_codes:
            keys["hn_postal"].append(f"{country}_hnpc_{hn}_{pc}")

    # Pass 9: domain + country
    for d in rec.domains:
        clean_d = d.replace(".com", "").replace(".org", "").replace(".net", "").replace(".", "_")
        keys["domain"].append(f"{country}_dom_{clean_d}")

    # Pass 10: DBA/AKA alias + country
    for a in rec.name_aliases:
        clean_a = a.strip()
        if clean_a:
            keys["dba_alias"].append(f"{country}_alias_{clean_a}")

    return keys


def generate_blocking_keys(
    rec: NormalizedRecord,
    suppress_stopwords: bool = True,
) -> Dict[str, List[str]]:
    """
    Backward-compatible key generator returning both classic multi-channel keys
    and the 10 multi-pass keys.
    """
    country = rec.country
    keys: Dict[str, List[str]] = {
        "address_core": [],
        "postal_name": [],
        "name_token": [],
        "name_prefix": [],
        "address_token": [],
        "domain_alias": [],
    }

    # 1. Address Core Channel
    for h in rec.house_numbers:
        if len(h) >= 3 or "/" in h:
            keys["address_core"].append(f"{country}_hn_{h}")
        if rec.address_street_core:
            keys["address_core"].append(f"{country}_h_{h}_{rec.address_street_core[:4]}")

    # 2. Postal + Name Channel
    for p in rec.postal_codes:
        keys["postal_name"].append(f"{country}_p_{p}")
        if rec.name_core:
            keys["postal_name"].append(f"{country}_p_{p}_{rec.name_core[:3]}")

    # 3. Name Token Channel
    for tok in rec.name_tokens[:3]:
        if len(tok) >= 3:
            if suppress_stopwords and tok in COMMON_NAME_STOPWORDS:
                continue
            keys["name_token"].append(f"{country}_n_{tok}")

    # 4. Name Prefix Channel
    core_compact = rec.name_core.replace(" ", "")
    if len(core_compact) >= 4:
        keys["name_prefix"].append(f"{country}_np_{core_compact[:4]}")

    # 5. Distinctive Address Token Channel
    for tok in rec.address_tokens:
        if len(tok) >= 5 and not tok.isdigit():
            if suppress_stopwords and tok in COMMON_ADDR_STOPWORDS:
                continue
            keys["address_token"].append(f"{country}_at_{tok}")

    # 6. Domain / Alias Channel
    for d in rec.domains:
        clean_d = d.replace(".com", "").replace(".org", "").replace(".net", "").replace(".", "_")
        keys["domain_alias"].append(f"{country}_dom_{clean_d}")

    for a in rec.name_aliases:
        alias_tokens = [t for t in a.split() if len(t) >= 3 and (not suppress_stopwords or t not in COMMON_NAME_STOPWORDS)]
        for at in alias_tokens[:2]:
            keys["domain_alias"].append(f"{country}_alias_{at}")

    # Also merge in multi-pass keys
    multi_keys = generate_multi_pass_keys(rec, suppress_stopwords=suppress_stopwords)
    keys.update(multi_keys)

    return keys
