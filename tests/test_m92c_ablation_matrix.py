from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_m92c_ablations import build_experiments


def _flags(config):
    return (
        config.enable_alias_blocking,
        config.enable_transliteration,
        config.enable_country_fallback,
        config.enable_smart_capping,
    )


def test_b0_has_no_recovery_channels():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B0"]) == (
        False,
        False,
        False,
        False,
    )


def test_b1_is_alias_only():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B1"]) == (
        True,
        False,
        False,
        False,
    )


def test_b2_is_transliteration_only():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B2"]) == (
        False,
        True,
        False,
        False,
    )


def test_b3_is_country_fallback_only():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B3"]) == (
        False,
        False,
        True,
        False,
    )


def test_b4_is_smart_capping_only():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B4"]) == (
        False,
        False,
        False,
        True,
    )


def test_b5_enables_all_channels():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B5"]) == (
        True,
        True,
        True,
        True,
    )


def test_b1_to_b4_are_not_duplicates():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    configurations = [
        _flags(experiments[name])
        for name in ("B1", "B2", "B3", "B4")
    ]

    assert len(set(configurations)) == 4


def test_b4_and_b5_are_not_duplicates():
    experiments = dict(
        (name, config)
        for name, _, config in build_experiments()
    )

    assert _flags(experiments["B4"]) != _flags(
        experiments["B5"]
    )
