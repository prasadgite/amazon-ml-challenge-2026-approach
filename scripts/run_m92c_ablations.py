#!/usr/bin/env python3
"""
M9.2-C Blocking Channel Ablation Runner.

Experiment design:

    B0 = frozen baseline

    B1 = secondary identity channel only
         - DBA / AKA / alias
         - domain identity

    B2 = cross-script transliteration only

    B3 = scoped country fallback only

    B4 = smart capped-bucket recovery only

    B5 = all channels combined

The experiments are intentionally NON-CUMULATIVE for B1-B4.
This allows causal attribution of each blocking channel.

Outputs:

    artifacts/diagnostics/m92c_blocking_ablations.json
    artifacts/diagnostics/m92c_blocking_ablations.txt

    artifacts/scratch/grounded_sample/
        blocking_b0.sqlite
        blocking_b1.sqlite
        blocking_b2.sqlite
        blocking_b3.sqlite
        blocking_b4.sqlite
        blocking_b5.sqlite
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Force temporary files onto the working drive.
# ---------------------------------------------------------------------------

SCRATCH_DIR = PROJECT_ROOT / "artifacts" / "scratch"
TEMP_DIR = SCRATCH_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)


from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.logging_config import setup_logger
from business_entity_resolution.normalization.normalizer import TextNormalizer
from business_entity_resolution.blocking.blocker import (
    BlockingConfig,
    BlockingEngine,
)
from business_entity_resolution.blocking.evaluator import (
    BlockingRecallEvaluator,
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(values: list[int | float], p: float) -> float:
    """Linear-interpolated percentile."""
    if not values:
        return 0.0

    values = sorted(values)

    k = (len(values) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return float(values[lower])

    return float(
        values[lower] * (upper - k)
        + values[upper] * (k - lower)
    )


def candidate_distribution(
    db_path: Path,
    total_s1_entities: int,
) -> dict[str, float]:
    """Compute candidate-count distribution including zero-candidate S1s."""

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT source1_id, COUNT(*)
            FROM candidates
            GROUP BY source1_id
            """
        ).fetchall()

    counts = [int(row[1]) for row in rows]

    missing = max(0, total_s1_entities - len(counts))

    all_counts = sorted([0] * missing + counts)

    if not all_counts:
        return {
            "s1_count": float(total_s1_entities),
            "s1_with_candidates": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "s1_count": float(total_s1_entities),
        "s1_with_candidates": float(len(counts)),
        "min": float(all_counts[0]),
        "mean": round(
            sum(all_counts) / len(all_counts),
            2,
        ),
        "median": percentile(all_counts, 0.50),
        "p75": percentile(all_counts, 0.75),
        "p90": percentile(all_counts, 0.90),
        "p95": percentile(all_counts, 0.95),
        "p99": percentile(all_counts, 0.99),
        "max": float(all_counts[-1]),
    }


# ---------------------------------------------------------------------------
# Ground-truth / candidate inspection
# ---------------------------------------------------------------------------

def load_ground_truth_pairs(
    gt_path: Path,
) -> set[tuple[str, str, str]]:
    """
    Load ground-truth positive pairs.

    Expected schema:

        source1_entity_id
        matched_entity_ids

    matched_entity_ids may contain comma/semicolon/pipe separated IDs.
    """

    import csv

    pairs: set[tuple[str, str, str]] = set()

    with gt_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:

        reader = csv.DictReader(handle, delimiter="\t")

        for row in reader:
            s1 = (
                row.get("source1_entity_id")
                or row.get("source1_id")
                or ""
            ).strip()

            raw_targets = (
                row.get("matched_entity_ids")
                or ""
            ).strip()

            if not s1 or not raw_targets:
                continue

            normalized = (
                raw_targets
                .replace(";", ",")
                .replace("|", ",")
            )

            for target_id in normalized.split(","):
                target_id = target_id.strip()

                if not target_id:
                    continue

                if target_id.startswith("S2-"):
                    target_source = "S2"
                elif target_id.startswith("S3-"):
                    target_source = "S3"
                else:
                    continue

                pairs.add(
                    (
                        s1,
                        target_source,
                        target_id,
                    )
                )

    return pairs


def load_candidate_pairs(
    db_path: Path,
) -> set[tuple[str, str, str]]:
    """Load generated candidate pairs from the blocking SQLite database."""

    pairs: set[tuple[str, str, str]] = set()

    with sqlite3.connect(str(db_path)) as conn:

        rows = conn.execute(
            """
            SELECT source1_id, target_source, target_id
            FROM candidates
            """
        ).fetchall()

    for s1, target_source, target_id in rows:
        pairs.add(
            (
                str(s1),
                str(target_source),
                str(target_id),
            )
        )

    return pairs


def recovery_sets(
    baseline_pairs: set[tuple[str, str, str]],
    variant_pairs: set[tuple[str, str, str]],
    ground_truth: set[tuple[str, str, str]],
) -> dict[str, int]:

    baseline_misses = ground_truth - baseline_pairs
    variant_misses = ground_truth - variant_pairs

    recovered = baseline_misses - variant_misses

    return {
        "baseline_misses": len(baseline_misses),
        "variant_misses": len(variant_misses),
        "newly_recovered_true_pairs": len(recovered),
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    description: str,
    config: BlockingConfig,
    sample_dir: Path,
    normalizer: TextNormalizer,
    logger: logging.Logger,
) -> dict[str, Any]:

    logger.info(
        "============================================================"
    )
    logger.info(
        "Starting %s: %s",
        name,
        description,
    )

    started = time.time()

    db_path = (
        sample_dir
        / f"blocking_{name.lower()}.sqlite"
    )

    if db_path.exists():
        db_path.unlink()

    s1_path = sample_dir / "train_source1.tsv"
    s2_path = sample_dir / "train_source2.tsv"
    s3_path = sample_dir / "train_source3.tsv"
    gt_path = sample_dir / "train_ground_truth.tsv"

    engine = BlockingEngine(
        db_path,
        config,
    )

    engine.initialize("train")

    target_count = engine.build_target_index(
        {
            "S2": s2_path,
            "S3": s3_path,
        },
        normalizer=normalizer,
    )

    stats = engine.generate_candidates(
        s1_path,
        "train",
        normalizer=normalizer,
    )

    evaluator = BlockingRecallEvaluator(
        db_path
    )

    report = evaluator.evaluate(
        gt_path
    )

    distribution = candidate_distribution(
        db_path,
        stats.s1_rows,
    )

    ground_truth = load_ground_truth_pairs(
        gt_path
    )

    candidates = load_candidate_pairs(
        db_path
    )

    covered = ground_truth & candidates
    missed = ground_truth - candidates

    elapsed = time.time() - started

    result = {
        "experiment": name,
        "description": description,

        "config": {
            "max_bucket_size": config.max_bucket_size,
            "enable_alias_blocking":
                config.enable_alias_blocking,
            "enable_transliteration":
                config.enable_transliteration,
            "enable_country_fallback":
                config.enable_country_fallback,
            "enable_smart_capping":
                config.enable_smart_capping,
        },

        "elapsed_seconds": round(
            elapsed,
            2,
        ),

        "target_entities": target_count,

        "ground_truth_pairs":
            len(ground_truth),

        "covered_ground_truth_pairs":
            len(covered),

        "missed_ground_truth_pairs":
            len(missed),

        "blocking_recall":
            round(
                len(covered) / len(ground_truth),
                6,
            ) if ground_truth else 1.0,

        "blocking_recall_pct":
            round(
                (
                    len(covered)
                    / len(ground_truth)
                    * 100.0
                ),
                4,
            ) if ground_truth else 100.0,

        "candidate_pairs":
            len(candidates),

        "candidate_reduction_ratio":
            round(
                report.candidate_reduction_ratio,
                6,
            ),

        "candidate_reduction_pct":
            round(
                report.candidate_reduction_ratio
                * 100.0,
                4,
            ),

        "candidates_per_s1":
            distribution,

        "strategy_recovered_counts":
            report.strategy_recovered_counts,

        "strategy_recall_breakdown":
            report.strategy_recall_breakdown,

        "db_path":
            str(db_path),

        # Keep actual IDs for causal analysis.
        "missed_pair_ids": [
            list(pair)
            for pair in sorted(missed)
        ],
    }

    logger.info(
        "[%s] Recall %.4f%% | covered=%d/%d | "
        "candidates=%d | median/S1=%.1f | "
        "P95/S1=%.1f | max/S1=%.0f | %.2fs",
        name,
        result["blocking_recall_pct"],
        result["covered_ground_truth_pairs"],
        result["ground_truth_pairs"],
        result["candidate_pairs"],
        distribution["median"],
        distribution["p95"],
        distribution["max"],
        elapsed,
    )

    return result


# ---------------------------------------------------------------------------
# Correct ablation matrix
# ---------------------------------------------------------------------------

def build_experiments() -> list[
    tuple[str, str, BlockingConfig]
]:
    """
    Build a TRUE isolated-channel ablation matrix.

    B1-B4 each enable exactly one new channel.
    B5 enables all channels.
    """

    return [

        (
            "B0",
            "Frozen baseline blocker",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=False,
                enable_transliteration=False,
                enable_country_fallback=False,
                enable_smart_capping=False,
            ),
        ),

        (
            "B1",
            "Secondary identity channel: "
            "DBA / AKA / alias / domain",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=True,
                enable_transliteration=False,
                enable_country_fallback=False,
                enable_smart_capping=False,
            ),
        ),

        (
            "B2",
            "Cross-script transliteration only",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=False,
                enable_transliteration=True,
                enable_country_fallback=False,
                enable_smart_capping=False,
            ),
        ),

        (
            "B3",
            "Scoped country fallback only",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=False,
                enable_transliteration=False,
                enable_country_fallback=True,
                enable_smart_capping=False,
            ),
        ),

        (
            "B4",
            "Smart capped-bucket recovery only",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=False,
                enable_transliteration=False,
                enable_country_fallback=False,
                enable_smart_capping=True,
            ),
        ),

        (
            "B5",
            "All blocking recovery channels combined",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=True,
                enable_transliteration=True,
                enable_country_fallback=True,
                enable_smart_capping=True,
                enable_secondary_identity=False,
            ),
        ),

        (
            "B6",
            "B5 + Secondary identity channel (DBA / AKA / embedded domain)",
            BlockingConfig(
                max_bucket_size=500,
                enable_alias_blocking=True,
                enable_transliteration=True,
                enable_country_fallback=True,
                enable_smart_capping=True,
                enable_secondary_identity=True,
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_analysis(
    results: list[dict[str, Any]],
    ground_truth: set[tuple[str, str, str]],
) -> dict[str, Any]:

    by_name = {
        result["experiment"]: result
        for result in results
    }

    b0 = by_name["B0"]

    baseline_pairs = load_candidate_pairs(
        Path(b0["db_path"])
    )

    analysis: dict[str, Any] = {
        "title":
            "M9.2-C Isolated Blocking Channel Ablation",

        "ground_truth_pairs":
            len(ground_truth),

        "experiments":
            results,

        "channel_effects": {},

        "combined_effect": {},
    }

    # ---------------------------------------------------------------
    # Individual channel effects
    # ---------------------------------------------------------------

    for channel in ["B1", "B2", "B3", "B4"]:
        if channel not in by_name or "B0" not in by_name:
            continue

        result = by_name[channel]

        variant_pairs = load_candidate_pairs(
            Path(result["db_path"])
        )

        recovery = recovery_sets(
            baseline_pairs,
            variant_pairs,
            ground_truth,
        )

        analysis["channel_effects"][channel] = {
            **recovery,

            "recall_delta_pp":
                round(
                    result["blocking_recall_pct"]
                    - b0["blocking_recall_pct"],
                    4,
                ),

            "candidate_delta":
                (
                    result["candidate_pairs"]
                    - b0["candidate_pairs"]
                ),

            "candidate_delta_pct":
                round(
                    (
                        result["candidate_pairs"]
                        / b0["candidate_pairs"]
                        - 1.0
                    )
                    * 100.0,
                    4,
                ),
        }

    # ---------------------------------------------------------------
    # Combined effect
    # ---------------------------------------------------------------

    if "B5" in by_name and "B0" in by_name:
        b5 = by_name["B5"]

        b5_pairs = load_candidate_pairs(
            Path(b5["db_path"])
        )

        combined_recovery = recovery_sets(
            baseline_pairs,
            b5_pairs,
            ground_truth,
        )

        individual_gain = sum(
            analysis["channel_effects"][channel][
                "newly_recovered_true_pairs"
            ]
            for channel in analysis["channel_effects"]
        )

        combined_gain = (
            b5["covered_ground_truth_pairs"]
            - b0["covered_ground_truth_pairs"]
        )

        analysis["combined_effect"] = {
            **combined_recovery,

            "recall_delta_pp":
                round(
                    b5["blocking_recall_pct"]
                    - b0["blocking_recall_pct"],
                    4,
                ),

            "candidate_delta":
                (
                    b5["candidate_pairs"]
                    - b0["candidate_pairs"]
                ),

            "candidate_delta_pct":
                round(
                    (
                        b5["candidate_pairs"]
                        / b0["candidate_pairs"]
                        - 1.0
                    )
                    * 100.0,
                    4,
                ),

            "sum_individual_recovered_pairs":
                individual_gain,

            "combined_recovered_pairs":
                combined_gain,

            "interaction_effect_pairs":
                combined_gain - individual_gain,
        }

    if "B6" in by_name and "B5" in by_name:
        b5 = by_name["B5"]
        b6 = by_name["B6"]
        b5_pairs = load_candidate_pairs(Path(b5["db_path"]))
        b6_pairs = load_candidate_pairs(Path(b6["db_path"]))
        b6_recovery = recovery_sets(
            b5_pairs,
            b6_pairs,
            ground_truth,
        )
        analysis["b6_effect"] = {
            **b6_recovery,
            "recall_delta_pp_vs_b5": round(
                b6["blocking_recall_pct"] - b5["blocking_recall_pct"], 4
            ),
            "candidate_delta_vs_b5": b6["candidate_pairs"] - b5["candidate_pairs"],
            "candidate_delta_pct_vs_b5": round(
                (b6["candidate_pairs"] / max(1, b5["candidate_pairs"]) - 1.0) * 100.0, 4
            ),
        }

    return analysis



# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    paths: ProjectPaths,
    analysis: dict[str, Any],
) -> None:

    diagnostics = paths.diagnostics_dir
    diagnostics.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        diagnostics
        / "m92c_blocking_ablations.json"
    )

    txt_path = (
        diagnostics
        / "m92c_blocking_ablations.txt"
    )

    json_path.write_text(
        json.dumps(
            analysis,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "=" * 100,
        "M9.2-C: ISOLATED BLOCKING CHANNEL ABLATION",
        "=" * 100,
        "",
        f"Ground-truth pairs: {analysis['ground_truth_pairs']:,}",
        "",
        "-" * 100,
        "EXPERIMENT RESULTS",
        "-" * 100,
        (
            f"{'Exp':<4} | "
            f"{'Recall':>9} | "
            f"{'Misses':>7} | "
            f"{'Candidates':>12} | "
            f"{'Median/S1':>11} | "
            f"{'P95/S1':>9} | "
            f"{'Max/S1':>9}"
        ),
        "-" * 100,
    ]

    for result in analysis["experiments"]:

        dist = result["candidates_per_s1"]

        lines.append(
            f"{result['experiment']:<4} | "
            f"{result['blocking_recall_pct']:>8.4f}% | "
            f"{result['missed_ground_truth_pairs']:>7} | "
            f"{result['candidate_pairs']:>12,} | "
            f"{dist['median']:>11.1f} | "
            f"{dist['p95']:>9.1f} | "
            f"{dist['max']:>9.0f}"
        )

    if analysis["channel_effects"]:
        lines.extend([
            "-" * 100,
            "",
            "-" * 100,
            "INDIVIDUAL CHANNEL EFFECTS",
            "-" * 100,
            (
                f"{'Channel':<8} | "
                f"{'Recovered':>10} | "
                f"{'Recall Δ pp':>12} | "
                f"{'Candidate Δ':>13} | "
                f"{'Candidate Δ %':>15}"
            ),
            "-" * 100,
        ])

        for channel, effect in analysis["channel_effects"].items():
            lines.append(
                f"{channel:<8} | "
                f"{effect['newly_recovered_true_pairs']:>10} | "
                f"{effect['recall_delta_pp']:>12.4f} | "
                f"{effect['candidate_delta']:>13,} | "
                f"{effect['candidate_delta_pct']:>14.4f}%"
            )

    combined = analysis.get("combined_effect", {})

    if "combined_recovered_pairs" in combined:
        lines.extend([
            "",
            "-" * 100,
            "COMBINED B5 EFFECT",
            "-" * 100,
            f"Recovered true pairs:       {combined['combined_recovered_pairs']}",
            f"Recall delta:               {combined['recall_delta_pp']:.4f} pp",
            f"Candidate delta:            {combined['candidate_delta']:,}",
            f"Candidate delta:            {combined['candidate_delta_pct']:.4f}%",
            f"Sum individual recoveries:  {combined['sum_individual_recovered_pairs']}",
            f"Interaction effect:         {combined['interaction_effect_pairs']}",
        ])

    if "b6_effect" in analysis:
        b6_eff = analysis["b6_effect"]
        lines.extend([
            "",
            "-" * 100,
            "B6 EFFECT (vs B5: + Secondary Identity Channel)",
            "-" * 100,
            f"Recovered true pairs vs B5: {b6_eff['newly_recovered_true_pairs']}",
            f"Recall delta vs B5:         {b6_eff['recall_delta_pp_vs_b5']:+.4f} pp",
            f"Candidate delta vs B5:      {b6_eff['candidate_delta_vs_b5']:+,}",
            f"Candidate delta % vs B5:    {b6_eff['candidate_delta_pct_vs_b5']:+.4f}%",
        ])

    lines.extend([
        "",
        "=" * 100,
    ])

    txt_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "M9.2-C isolated blocking-channel "
            "ablation runner"
        )
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names to run (e.g. B0,B5,B6 or all)",
    )

    args = parser.parse_args()

    paths = ProjectPaths()

    logger = setup_logger(
        "m92c_blocking_ablations",
        log_file=(
            paths.logs_dir
            / "m92c_blocking_ablations.log"
        ),
    )

    sample_dir = (
        paths.artifacts_dir
        / "scratch"
        / "grounded_sample"
    )

    required_files = [
        sample_dir / "train_source1.tsv",
        sample_dir / "train_source2.tsv",
        sample_dir / "train_source3.tsv",
        sample_dir / "train_ground_truth.tsv",
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required grounded evaluation files missing:\n"
            + "\n".join(missing)
        )

    normalizer = TextNormalizer()

    experiments = build_experiments()

    if args.experiments != "all":
        selected_names = {x.strip().upper() for x in args.experiments.split(",")}
        experiments = [e for e in experiments if e[0].upper() in selected_names]

    results: list[dict[str, Any]] = []

    for (
        experiment_name,
        description,
        config,
    ) in experiments:

        result = run_experiment(
            experiment_name,
            description,
            config,
            sample_dir,
            normalizer,
            logger,
        )

        results.append(result)

    ground_truth = load_ground_truth_pairs(
        sample_dir / "train_ground_truth.tsv"
    )

    analysis = build_analysis(
        results,
        ground_truth,
    )

    write_report(
        paths,
        analysis,
    )

    print()
    print("=" * 100)
    print("M9.2-C ABLATION COMPLETE")
    print("=" * 100)

    for result in results:

        print(
            f"{result['experiment']}: "
            f"recall={result['blocking_recall_pct']:.4f}% | "
            f"misses={result['missed_ground_truth_pairs']} | "
            f"candidates={result['candidate_pairs']:,}"
        )

    for channel, effect in analysis.get("channel_effects", {}).items():
        print(
            f"{channel}: "
            f"recovered="
            f"{effect['newly_recovered_true_pairs']} | "
            f"recall_delta="
            f"{effect['recall_delta_pp']:+.4f} pp | "
            f"candidate_delta="
            f"{effect['candidate_delta']:+,}"
        )

    if analysis.get("channel_effects"):
        print()

    combined = analysis.get("combined_effect", {})
    if "combined_recovered_pairs" in combined:
        print(
            "B5 combined: "
            f"recovered="
            f"{combined['combined_recovered_pairs']} | "
            f"recall_delta="
            f"{combined['recall_delta_pp']:+.4f} pp | "
            f"candidate_delta="
            f"{combined['candidate_delta']:+,}"
        )
        print()

    if "b6_effect" in analysis:
        b6_eff = analysis["b6_effect"]
        print(
            "B6 (+ secondary identity): "
            f"recovered_vs_b5="
            f"{b6_eff['newly_recovered_true_pairs']} | "
            f"recall_delta_vs_b5="
            f"{b6_eff['recall_delta_pp_vs_b5']:+.4f} pp | "
            f"candidate_delta_vs_b5="
            f"{b6_eff['candidate_delta_vs_b5']:+,}"
        )
        print()

    print()
    print(
        "Report:",
        paths.diagnostics_dir
        / "m92c_blocking_ablations.txt",
    )


if __name__ == "__main__":
    main()
