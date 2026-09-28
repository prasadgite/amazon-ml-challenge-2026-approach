import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.normalization.normalizer import TextNormalizer


N = TextNormalizer()


def test_unicode_and_legal_suffix_normalization():
    r = N.normalize("1", "Café Société LLC", "0189 Rue de l'Église, 75001 Paris", "France")
    assert r.country == "fr"
    assert r.name_clean == "cafe societe llc"
    assert r.name_core == "cafe societe"
    assert r.house_numbers == ("189",)
    assert "75001" in r.postal_codes
    assert r.address_street_core.startswith("189 rue")


def test_alias_and_domain_are_separate_signals():
    r = N.normalize(
        "1",
        "Halodelta aka Clemons Silver Eastern Inc https://lyrellewave.com",
        "12 Main Street, Pune 411001",
        "India",
    )
    assert "clemons silver eastern inc" in r.name_aliases
    assert "lyrellewave.com" in r.domains
    assert "lyrellewave" in r.name_domain_names
    assert r.name_core.startswith("halodelta")


def test_house_number_conflicts_are_preserved():
    r = N.normalize("1", "Acme Ltd", "No. 0042-0043 Main Road", "US")
    assert r.house_numbers == ("42", "43")
    assert "rd" in r.address_tokens


def test_street_abbreviation_equivalence():
    a = N.normalize("1", "Acme", "12 Main Street", "US")
    b = N.normalize("2", "Acme", "12 Main St.", "US")
    assert a.address_street_core == b.address_street_core
    assert a.house_numbers == b.house_numbers


def test_non_latin_text_is_not_deleted():
    r = N.normalize("1", "株式会社 テスト", "東京都1丁目2番", "Japan")
    assert r.name_core
    assert r.address_clean
