"""
Unit tests for Milestone 4 - Normalization Engine.
"""

import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from business_entity_resolution.io.record import EntityRecord
from business_entity_resolution.normalization import (
    normalize_unicode,
    clean_string,
    NormalizedName,
    strip_legal_suffix,
    extract_aliases,
    extract_domain,
    extract_house_numbers,
    extract_postal_code,
    standardize_street_terms,
    NormalizedAddress,
    NormalizedEntity,
)


def test_unicode_accents():
    # French accent test
    assert normalize_unicode("Thermal & Fils SASU") == "Thermal & Fils SASU"
    assert clean_string("Ptit Àmicale École") == "ptit amicale ecole"
    assert clean_string("Changodar Sérvices") == "changodar services"


def test_legal_suffixes():
    # US
    assert strip_legal_suffix("clemons silver eastern inc") == "clemons silver eastern"
    assert strip_legal_suffix("lyrelle wave llc") == "lyrelle wave"
    # India
    assert strip_legal_suffix("bs projects private limited") == "bs projects"
    assert strip_legal_suffix("reliance industries pvt ltd") == "reliance industries"
    # France
    assert strip_legal_suffix("znb club sarl") == "znb club"
    assert strip_legal_suffix("engages art pharmacie sci") == "engages art pharmacie"
    # Devanagari
    assert strip_legal_suffix("राम मार्केटिंग प्राइवेट लिमिटेड") == "राम मार्केटिंग"


def test_domain_extraction():
    assert extract_domain("wilfordhancock.com") == "wilfordhancock"
    assert extract_domain("lyrellewave.com") == "lyrellewave"
    assert extract_domain("www.abc-logistics.org") == "abc logistics"
    assert extract_domain("Regular Company Name") is None


def test_alias_extraction():
    aliases = extract_aliases("Halodelta aka Clemons Silver Eastern Inc")
    assert len(aliases) == 2
    assert "halodelta" in aliases
    assert "clemons silver eastern inc" in aliases


def test_address_house_numbers():
    # Leading zero removal
    assert extract_house_numbers("0189 Laurel Road, Arden, NC") == ["189"]
    # Fractional / multiple numbers
    assert "1619" in extract_house_numbers("1619 1/2 Julia Park Drive")
    # Indian slash format
    assert extract_house_numbers("Shop No.- 4, 59/101 Kanhaiya Plaza") == ["4", "59/101"]
    # French bis format
    assert "5" in extract_house_numbers("5 bis Rue Pierre Dignac")


def test_postal_codes():
    assert extract_postal_code("Pune, Maharashtra 411001", country="India") == "411001"
    assert extract_postal_code("Tyler, TX 75701", country="US") == "75701"
    assert extract_postal_code("75008 Paris, France", country="France") == "75008"


def test_street_abbreviations():
    res = standardize_street_terms("189 mg rd, 42 5th st, 10 r de lille")
    assert "road" in res
    assert "street" in res
    assert "rue" in res


def test_normalized_entity():
    rec = EntityRecord(
        entity_id="S1-505333132",
        business_name="Clemons Silver Eastern Inc",
        business_address="1619 Julia Park Drive, Spring, TX 77386",
        country="US",
    )
    ne = NormalizedEntity(rec)
    assert ne.entity_id == "S1-505333132"
    assert ne.name.core == "clemons silver eastern"
    assert ne.address.house_numbers == ["1619"]
    assert ne.address.postal_code == "77386"
    assert "julia" in ne.address.street_core


if __name__ == "__main__":
    print("Running normalization engine tests...")
    test_unicode_accents()
    test_legal_suffixes()
    test_domain_extraction()
    test_alias_extraction()
    test_address_house_numbers()
    test_postal_codes()
    test_street_abbreviations()
    test_normalized_entity()
    print("ALL NORMALIZATION TESTS PASSED!")
