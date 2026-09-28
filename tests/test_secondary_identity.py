from business_entity_resolution.blocking.secondary_identity import (
    extract_aliases,
    extract_domain,
    secondary_identity_keys,
)


def test_domain():

    assert (
        extract_domain(
            "Acme Holdings acme.com"
        )
        == "acme com"
    )


def test_dba():

    aliases = extract_aliases(
        "ABC Corporation DBA Super Store"
    )

    assert (
        "super store"
        in aliases
    )


def test_aka():

    aliases = extract_aliases(
        "ABC Ltd AKA Alpha Traders"
    )

    assert (
        "alpha traders"
        in aliases
    )


def test_secondary_keys():

    keys = secondary_identity_keys(
        "ABC Corp DBA Alpha Traders alpha.com"
    )

    assert (
        "alias:alpha traders"
        in keys
    )

    assert (
        "domain:alpha com"
        in keys
    )
