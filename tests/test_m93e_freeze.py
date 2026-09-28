from pathlib import Path


def test_m93e_production_feature_engine_is_unchanged():
    path = Path(
        "src/business_entity_resolution/features/pairwise.py"
    )

    assert path.exists()

    text = path.read_text(
        encoding="utf-8"
    )

    forbidden = (
        "soft_token_similarity",
        "soft_token_coverage",
        "embedded_domain_present",
        "embedded_domain_name_similarity",
    )

    for feature in forbidden:
        assert feature not in text
