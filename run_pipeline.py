#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import os
import csv
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# Enforce all temporary storage on D: drive where space is ample
d_temp = Path(__file__).resolve().parent / "artifacts" / "scratch" / "temp"
d_temp.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(d_temp)
os.environ["TMP"] = str(d_temp)
os.environ["TMPDIR"] = str(d_temp)

try:
    csv.field_size_limit(2147483647)
except Exception:
    pass

from business_entity_resolution.bootstrap import check_environment, validate_schema
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.logging_config import configure_logging
from business_entity_resolution.normalization.normalizer import (
    TextNormalizer, iter_normalized_tsv, write_normalized_jsonl,
)
from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
from business_entity_resolution.blocking.evaluator import BlockingRecallEvaluator
from business_entity_resolution.features.pairwise import FeatureConfig, FeatureSchema, PairFeatureEngine
from business_entity_resolution.matching.model import MatchModel, MatchModelConfig
from business_entity_resolution.decision.calibration import DecisionConfig, DecisionEngine
from business_entity_resolution.consolidation.consolidator import ConsolidationConfig, ConsolidationEvaluator, EntityConsolidator
from business_entity_resolution.evaluation.final_evaluator import FinalEvaluationConfig, FinalEvaluator
from business_entity_resolution.profiling.profile import (
    DatasetProfiler,
    write_profile_json,
    write_profile_report,
)

LOGGER = logging.getLogger("ber.pipeline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Amazon ML 2026 Business Entity Resolution pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=("verify", "profile", "normalize", "block", "evaluate-blocking", "blocking", "features", "validate", "train-model", "predict", "model", "calibrate", "decision", "consolidate", "entity-resolution", "final-evaluation", "evaluate-final", "finalize-test", "m9-test", "all"),
        default="validate",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None, help="Path to configuration JSON file.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--exact-id-uniqueness",
        action="store_true",
        help="Track exact entity IDs in RAM to detect duplicate IDs. "
             "Leave off for lowest memory usage.",
    )
    parser.add_argument(
        "--max-bucket-size",
        type=int,
        default=500,
        help="Skip blocking keys whose target bucket exceeds this size.",
    )
    parser.add_argument(
        "--export-candidates",
        action="store_true",
        help="Also export candidate TSV files; disabled by default to avoid duplicating large SQLite artifacts.",
    )
    parser.add_argument(
        "--feature-db", type=Path, default=None,
        help="SQLite feature-store path; defaults to artifacts/indexes/<split>_features.sqlite.",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Generate pairwise features without training labels.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of normalized records per source to export for inspection.",
    )
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Model pickle path; defaults to artifacts/models/match_model.pkl.")
    parser.add_argument("--validation-fraction", type=float, default=0.20,
                        help="Stable S1-level validation fraction for M6.")
    parser.add_argument("--hard-negative-ratio", type=int, default=4,
                        help="Maximum hard-negative examples per positive training example.")
    parser.add_argument("--max-positive-rows", type=int, default=100000,
                        help="Maximum positive training rows retained in memory.")
    parser.add_argument("--max-negative-rows", type=int, default=400000,
                        help="Maximum negative candidates retained before hard-negative selection.")
    parser.add_argument("--max-validation-rows", type=int, default=200000,
                        help="Maximum validation rows retained for metrics.")
    parser.add_argument("--thresholds", type=str, default="0.5,0.7,0.8,0.9,0.95,0.97,0.98,0.99",
                        help="Comma-separated thresholds for validation metrics.")
    parser.add_argument("--calibration-method", choices=("platt", "isotonic", "none"), default="platt",
                        help="M7 probability calibration method.")
    parser.add_argument("--calibration-fraction", type=float, default=0.50,
                        help="Fraction of M6-held-out S1 groups used to fit the calibrator.")
    parser.add_argument("--target-precision", type=float, default=0.99,
                        help="Required validation precision for M7 threshold selection.")
    parser.add_argument("--min-recall-at-target", type=float, default=0.0,
                        help="Minimum recall constraint when selecting the precision-constrained threshold.")
    parser.add_argument("--threshold-grid-size", type=int, default=1001,
                        help="Number of threshold points in M7 threshold sweep.")
    parser.add_argument("--decision-path", type=Path, default=None,
                        help="M7 decision policy path; defaults to artifacts/models/decision_policy.pkl.")
    parser.add_argument("--no-country-veto", action="store_true",
                        help="Disable the conservative country-conflict veto in M7 decisions.")
    parser.add_argument("--m7-fp-limit", type=int, default=100,
                        help="Number of validation false positives to export for diagnostics.")
    parser.add_argument("--m8-min-edge-score", type=float, default=0.0,
                        help="Minimum calibrated M7 score required for an M8 cluster edge.")
    parser.add_argument("--m8-max-cluster-size", type=int, default=1000,
                        help="Flag clusters larger than this size for diagnostics.")
    parser.add_argument("--m8-no-explicit-negative-veto", action="store_true",
                        help="Allow transitive cluster merges despite strong explicit NON_MATCH evidence.")
    parser.add_argument("--m8-explicit-negative-threshold", type=float, default=0.99,
                        help="M7 calibrated score at or above which an explicit NON_MATCH blocks transitive merging.")
    parser.add_argument("--m8-no-country-veto", action="store_true",
                        help="Allow M8 edges with explicit country-conflict evidence.")
    parser.add_argument("--m9-folds", type=int, default=5,
                        help="Number of S1-grouped development OOF folds for M9.")
    parser.add_argument("--m9-holdout-fraction", type=float, default=0.15,
                        help="Deterministic S1-level holdout fraction never used for model or threshold selection.")
    parser.add_argument("--m9-calibration-fraction", type=float, default=0.25,
                        help="Nested calibration fraction inside each M9 training fold.")
    parser.add_argument("--m9-calibration-method", choices=("platt", "isotonic", "none"), default="platt",
                        help="Nested OOF probability calibration method.")
    parser.add_argument("--m9-threshold-grid-size", type=int, default=401,
                        help="M9 threshold grid size. Threshold evaluation uses one sorted score pass.")
    parser.add_argument("--m9-target-precision", type=float, default=0.99,
                        help="Precision constraint used only for M9 threshold selection.")
    parser.add_argument("--m9-min-recall", type=float, default=0.0,
                        help="Minimum macro recall constraint for M9 threshold selection.")
    parser.add_argument(
        "--m9-model",
        choices=("logistic", "hgb"),
        default="logistic",
        help=(
            "M9 matcher. Keep logistic for the baseline; "
            "use hgb for M9.1 nonlinear matcher."
        ),
    )
    parser.add_argument(
        "--m9-hgb-max-iter",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--m9-hgb-learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--m9-hgb-max-leaf-nodes",
        type=int,
        default=31,
    )
    parser.add_argument(
        "--m9-hgb-min-samples-leaf",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--m9-hgb-l2",
        type=float,
        default=1.0,
    )
    parser.add_argument("--m9-max-positive-rows", type=int, default=100000)
    parser.add_argument("--m9-max-negative-rows", type=int, default=400000)
    parser.add_argument("--m9-max-calibration-rows", type=int, default=100000)
    parser.add_argument("--m9-top-errors", type=int, default=100,
                        help="Maximum false-positive/false-negative examples stored in diagnostics.")

    # M9.2 evaluation cohort options
    parser.add_argument(
        "--evaluation-s1-source",
        choices=("feature", "file"),
        default="feature",
        help="Source of S1 evaluation cohort: 'feature' (default) derives from feature file, 'file' uses explicit file.",
    )
    parser.add_argument(
        "--evaluation-s1-file",
        type=Path,
        default=None,
        help="Path to S1 ID file when --evaluation-s1-source is 'file'.",
    )

    # M9.2 multi-channel blocking controls
    parser.add_argument("--m92-no-alias-blocking", action="store_true", help="Disable alias/DBA blocking keys.")
    parser.add_argument("--m92-no-transliteration", action="store_true", help="Disable script transliteration.")
    parser.add_argument("--m92-no-country-fallback", action="store_true", help="Disable country fallback.")
    parser.add_argument("--m92-no-smart-capping", action="store_true", help="Disable smart overflow capping.")
    return parser.parse_args()


def validate_dataset(config: PipelineConfig) -> int:
    env = check_environment(config)
    if env["errors"]:
        for error in env["errors"]:
            LOGGER.error(error)
        return 1

    schema = validate_schema(config)
    if not schema["valid"]:
        for filename in schema["invalid_files"]:
            LOGGER.error("Invalid TSV schema: %s", filename)
        return 1

    LOGGER.info("Environment & Dataset Verification")
    LOGGER.info("Dataset files and schemas validated.")
    return 0


def stage_verify(paths=None, log=None) -> bool:
    config = PipelineConfig.from_root(paths.root if paths else None)
    return validate_dataset(config) == 0


def stage_validate(paths=None, log=None) -> bool:
    if paths and not paths.matching_results.is_file():
        return False
    config = PipelineConfig.from_root(paths.root if paths else None)
    return validate_dataset(config) == 0


def run_stage_pairwise_features(paths=None, config=None, log=None) -> bool:
    if paths is not None:
        idx_dir = getattr(paths, "indexes_dir", paths.root / "artifacts" / "indexes")
        blocking_db = idx_dir / "train_blocking.sqlite"
        if not blocking_db.is_file():
            if log:
                log.error("Missing blocking database: %s", blocking_db)
            return False
    cfg = config or PipelineConfig.from_root(paths.root if paths else None)
    try:
        return run_pairwise_features(cfg, None, True) == 0
    except Exception as exc:
        if log:
            log.error("Pairwise features failed: %s", exc)
        return False



def run_profile(config: PipelineConfig, exact_id_uniqueness: bool) -> int:
    LOGGER.info("Starting Milestone 2 streaming profiler.")
    profiler = DatasetProfiler()

    result = profiler.profile_dataset(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        exact_id_uniqueness=exact_id_uniqueness,
    )

    json_path = config.artifact_dir / "diagnostics" / "dataset_profile.json"
    report_path = config.artifact_dir / "diagnostics" / "dataset_profile.txt"

    write_profile_json(result, json_path)
    write_profile_report(result, report_path)

    LOGGER.info("Profile JSON: %s", json_path)
    LOGGER.info("Profile report: %s", report_path)

    for split in ("train", "test"):
        total = sum(x["rows"] for x in result[split]["sources"].values())
        LOGGER.info("%s total records: %,d", split, total)

    gt = result["train"]["ground_truth"]
    LOGGER.info(
        "Train S1: %,d | zero-match: %,d | singleton: %,d | match links: %,d",
        gt["rows"], gt["zero_match_s1"], gt["singleton_s1"],
        gt["total_match_links"],
    )
    return 0


def run_normalization_sample(config: PipelineConfig, sample_size: int) -> int:
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0")

    normalizer = TextNormalizer()
    output_dir = config.artifact_dir / "diagnostics" / "normalization_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_dir, prefix in ((config.train_dir, "train"), (config.test_dir, "test")):
        for source_num in (1, 2, 3):
            source_path = split_dir / f"{prefix}_source{source_num}.tsv"
            output_path = output_dir / f"{prefix}_source{source_num}.jsonl"
            count = write_normalized_jsonl(
                iter_normalized_tsv(source_path, normalizer),
                output_path,
                limit=sample_size,
            )
            LOGGER.info("Normalized sample: %s -> %,d records", output_path, count)

    LOGGER.info("Milestone 3 normalization sample generation completed.")
    return 0


def run_blocking(config: PipelineConfig, max_bucket_size_or_args: Any = 500, export_candidates: bool = False) -> int:
    if hasattr(max_bucket_size_or_args, "max_bucket_size"):
        args = max_bucket_size_or_args
        max_bucket_size = args.max_bucket_size
        export_candidates = getattr(args, "export_candidates", False)
        enable_alias = not getattr(args, "m92_no_alias_blocking", False)
        enable_translit = not getattr(args, "m92_no_transliteration", False)
        enable_country = not getattr(args, "m92_no_country_fallback", False)
        enable_smart = not getattr(args, "m92_no_smart_capping", False)
    else:
        max_bucket_size = int(max_bucket_size_or_args)
        enable_alias = True
        enable_translit = True
        enable_country = True
        enable_smart = True

    if max_bucket_size <= 0:
        raise ValueError("--max-bucket-size must be > 0")

    blocking_config = BlockingConfig(
        max_bucket_size=max_bucket_size,
        enable_alias_blocking=enable_alias,
        enable_transliteration=enable_translit,
        enable_country_fallback=enable_country,
        enable_smart_capping=enable_smart,
    )
    normalizer = TextNormalizer()
    for split, split_dir in (("train", config.train_dir), ("test", config.test_dir)):
        db_path = config.artifact_dir / "indexes" / f"{split}_blocking.sqlite"
        engine = BlockingEngine(db_path, blocking_config)
        engine.initialize(split)
        target_count = engine.build_target_index({
            "S2": split_dir / f"{split}_source2.tsv",
            "S3": split_dir / f"{split}_source3.tsv",
        }, normalizer)
        stats = engine.generate_candidates(
            split_dir / f"{split}_source1.tsv", split, normalizer
        )

        report = stats.as_dict()
        report["bucket_statistics"] = engine.bucket_statistics()
        report["target_index_rows"] = target_count
        report_path = config.artifact_dir / "diagnostics" / f"{split}_blocking.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if export_candidates:
            candidate_path = config.artifact_dir / "features" / f"{split}_candidates.tsv"
            engine.write_candidates_tsv(candidate_path)
            LOGGER.info("Candidate TSV: %s", candidate_path)

        LOGGER.info(
            "%s blocking: S1=%,d targets=%,d candidates=%,d capped_keys=%,d reduction=%.6f",
            split, stats.s1_rows, stats.target_rows, stats.candidate_pairs,
            stats.capped_bucket_keys, report["candidate_reduction_ratio"],
        )

    return 0


def run_pairwise_features(config: PipelineConfig, feature_db: Path | None, include_labels: bool) -> int:
    """Build normalized feature stores and stream M4 candidates into feature TSVs."""
    normalizer = TextNormalizer()
    for split, split_dir in (("train", config.train_dir), ("test", config.test_dir)):
        blocking_db = config.artifact_dir / "indexes" / f"{split}_blocking.sqlite"
        if not blocking_db.exists():
            raise FileNotFoundError(f"Missing blocking database: {blocking_db}. Run --stage block first.")

        db_path = feature_db if feature_db and split == "train" else config.artifact_dir / "indexes" / f"{split}_features.sqlite"
        engine = PairFeatureEngine(db_path, FeatureConfig(include_labels=include_labels and split == "train"))
        engine.initialize()
        source_paths = {
            "S1": split_dir / f"{split}_source1.tsv",
            "S2": split_dir / f"{split}_source2.tsv",
            "S3": split_dir / f"{split}_source3.tsv",
        }
        record_count = engine.build_record_store(source_paths, normalizer)
        if include_labels and split == "train":
            label_count = engine.load_ground_truth(config.train_ground_truth)
        else:
            label_count = 0

        output_path = config.artifact_dir / "features" / f"{split}_pair_features.tsv"
        stats = engine.generate_features(blocking_db, output_path, include_labels=(include_labels and split == "train"))
        report = {
            "split": split,
            "record_store_rows": record_count,
            "ground_truth_positive_links": label_count,
            "output_path": str(output_path),
            "feature_columns": FeatureSchema.columns(include_labels and split == "train"),
            **stats,
        }
        report_path = config.artifact_dir / "diagnostics" / f"{split}_pair_features.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info(
            "%s pair features: pairs=%,d positives=%,d negatives=%,d output=%s",
            split, stats.get("pairs", 0), stats.get("positive", 0), stats.get("negative", 0), output_path,
        )
    return 0


def run_matching_model(config: PipelineConfig, model_path: Path | None, validation_fraction: float,
                       hard_negative_ratio: int, max_positive_rows: int, max_negative_rows: int,
                       max_validation_rows: int, thresholds: str) -> int:
    feature_path = config.artifact_dir / "features" / "train_pair_features.tsv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing training features: {feature_path}. Run --stage features first.")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be > 0 and < 0.5")
    if hard_negative_ratio <= 0:
        raise ValueError("--hard-negative-ratio must be > 0")
    model_path = model_path or config.artifact_dir / "models" / "match_model.pkl"
    model = MatchModel(MatchModelConfig(
        validation_fraction=validation_fraction,
        hard_negative_ratio=hard_negative_ratio,
        max_positive_rows=max_positive_rows,
        max_negative_rows=max_negative_rows,
        max_validation_rows=max_validation_rows,
    ))
    summary = model.fit(feature_path)
    model.save(model_path)
    coeff_path = config.artifact_dir / "diagnostics" / "match_model_coefficients.json"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    coeff_path.write_text(json.dumps(model.coefficient_report(), indent=2) + "\n", encoding="utf-8")
    threshold_values = [float(x.strip()) for x in thresholds.split(",") if x.strip()]
    threshold_report = model.evaluate_thresholds(feature_path, threshold_values)
    threshold_path = config.artifact_dir / "diagnostics" / "match_model_thresholds.json"
    threshold_path.write_text(json.dumps(threshold_report, indent=2) + "\n", encoding="utf-8")
    report = {**summary, "model_path": str(model_path), "coefficient_report": str(coeff_path),
              "threshold_report": str(threshold_path)}
    report_path = config.artifact_dir / "diagnostics" / "match_model.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("M6 model trained: train=%d validation=%d model=%s",
                summary["train_rows"], summary["validation_rows"], model_path)
    if summary.get("validation"):
        LOGGER.info("M6 validation: precision=%.6f recall=%.6f AP=%.6f",
                    summary["validation"]["precision"], summary["validation"]["recall"],
                    summary["validation"]["average_precision"] or 0.0)
    return 0

def run_decision_calibration(config: PipelineConfig, model_path: Path | None, decision_path: Path | None,
                             calibration_method: str, calibration_fraction: float, target_precision: float,
                             min_recall_at_target: float, threshold_grid_size: int, no_country_veto: bool,
                             fp_limit: int) -> int:
    model_path = model_path or config.artifact_dir / "models" / "match_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing M6 model: {model_path}. Run --stage train-model first.")
    feature_path = config.artifact_dir / "features" / "train_pair_features.tsv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing training features: {feature_path}. Run --stage features first.")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be > 0 and < 1")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError("--target-precision must be > 0 and <= 1")
    if not 0.0 <= min_recall_at_target <= 1.0:
        raise ValueError("--min-recall-at-target must be between 0 and 1")
    if threshold_grid_size < 11:
        raise ValueError("--threshold-grid-size must be >= 11")
    if fp_limit < 0:
        raise ValueError("--m7-fp-limit must be >= 0")

    model = MatchModel.load(model_path)
    engine = DecisionEngine(model, DecisionConfig(
        calibration_fraction=calibration_fraction,
        calibration_method=calibration_method,
        target_precision=target_precision,
        min_recall_at_target=min_recall_at_target,
        threshold_grid_size=threshold_grid_size,
        veto_country_conflict=not no_country_veto,
    ))
    report = engine.fit(feature_path)
    decision_path = decision_path or config.artifact_dir / "models" / "decision_policy.pkl"
    engine.save(decision_path)

    report["decision_path"] = str(decision_path)
    report_path = config.artifact_dir / "diagnostics" / "decision_calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fp = engine.false_positive_report(feature_path, fp_limit) if fp_limit else []
    fp_path = config.artifact_dir / "diagnostics" / "validation_false_positives.json"
    fp_path.write_text(json.dumps(fp, indent=2) + "\n", encoding="utf-8")

    selected = report["evaluation"]["selected_metrics"]
    LOGGER.info(
        "M7 decision: status=%s threshold=%.6f precision=%.6f recall=%.6f calibration=%s",
        report["evaluation"]["threshold_selection"]["status"],
        engine.threshold, selected["precision"], selected["recall"], calibration_method,
    )
    LOGGER.info("M7 policy: %s", decision_path)
    return 0



def run_consolidation(config: PipelineConfig, model_path: Path | None, decision_path: Path | None,
                      split: str, m8_min_edge_score: float, m8_max_cluster_size: int,
                      m8_no_country_veto: bool, m8_no_explicit_negative_veto: bool,
                      m8_explicit_negative_threshold: float) -> int:
    model_path = model_path or config.artifact_dir / "models" / "match_model.pkl"
    decision_path = decision_path or config.artifact_dir / "models" / "decision_policy.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing M6 model: {model_path}. Run --stage train-model first.")
    if not decision_path.exists():
        raise FileNotFoundError(f"Missing M7 policy: {decision_path}. Run --stage calibrate first.")
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    model = MatchModel.load(model_path)
    engine = DecisionEngine.load(decision_path, model)
    feature_path = config.artifact_dir / "features" / f"{split}_pair_features.tsv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing {split} features: {feature_path}. Run --stage features first.")

    pair_decisions = config.artifact_dir / "predictions" / f"{split}_pair_decisions.tsv"
    score_stats = engine.score_file(feature_path, pair_decisions)

    consolidator = EntityConsolidator(ConsolidationConfig(
        min_edge_score=m8_min_edge_score,
        reject_country_conflict=not m8_no_country_veto,
        reject_explicit_negative=not m8_no_explicit_negative_veto,
        explicit_negative_score_threshold=m8_explicit_negative_threshold,
        max_cluster_size=m8_max_cluster_size,
    ))
    assignments = config.artifact_dir / "predictions" / f"{split}_entity_assignments.tsv"
    resolution = config.artifact_dir / "predictions" / f"{split}_entity_resolution.tsv"
    edges = config.artifact_dir / "predictions" / f"{split}_accepted_edges.tsv"
    report_path = config.artifact_dir / "diagnostics" / f"{split}_entity_consolidation.json"
    report = consolidator.run(pair_decisions, assignments, edges, report_path)
    state = consolidator.build_clusters(pair_decisions)
    report["s1_resolution"] = consolidator.write_s1_resolution(state, resolution)
    report["score_stats"] = score_stats

    if split == "train":
        evaluator = ConsolidationEvaluator()
        evaluation = evaluator.evaluate(state, config.train_ground_truth)
        report["ground_truth_evaluation"] = evaluation
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOGGER.info(
            "M8 train: clusters=%,d direct_precision=%.6f cluster_precision=%.6f cluster_recall=%.6f inferred_fp=%d",
            report["clusters"],
            evaluation["direct_edge_metrics"]["precision"],
            evaluation["cluster_closure_metrics"]["precision"],
            evaluation["cluster_closure_metrics"]["recall"],
            evaluation["transitive_inference"]["incorrect_inferred_pairs"],
        )
    else:
        LOGGER.info("M8 test: clusters=%,d accepted_edges=%,d assignments=%s",
                    report["clusters"], report["accepted_edges"], assignments)
    return 0

def run_prediction(config: PipelineConfig, model_path: Path | None) -> int:
    model_path = model_path or config.artifact_dir / "models" / "match_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Run --stage train-model first.")
    model = MatchModel.load(model_path)
    feature_path = config.artifact_dir / "features" / "test_pair_features.tsv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing test features: {feature_path}. Run --stage features first.")
    output_path = config.artifact_dir / "predictions" / "test_pair_scores.tsv"
    stats = model.predict_file(feature_path, output_path)
    report_path = config.artifact_dir / "diagnostics" / "test_pair_scores.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("M6 test scoring: rows=%,d output=%s", stats["rows"], output_path)
    return 0


def run_blocking_evaluation(config: PipelineConfig) -> int:
    db_path = config.artifact_dir / "indexes" / "train_blocking.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(
            f"Missing blocking database: {db_path}. Run --stage block first."
        )
    evaluator = BlockingRecallEvaluator(db_path)
    report = evaluator.evaluate(config.train_ground_truth)
    output_path = config.artifact_dir / "diagnostics" / "train_blocking_recall.json"
    evaluator.write_report(report, output_path)
    LOGGER.info(
        "Train blocking recall=%.8f (%d/%d positive pairs), candidates=%,d, reduction=%.6f, candidate_positive_rate=%.8f",
        report.recall, report.covered_ground_truth_pairs, report.total_ground_truth_pairs,
        report.candidate_pairs, report.candidate_reduction_ratio,
        report.covered_ground_truth_pairs / report.candidate_pairs if report.candidate_pairs else 0.0,
    )
    return 0


def run_final_evaluation(config: PipelineConfig, args) -> int:
    feature_path = config.artifact_dir / "features" / "train_pair_features.tsv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing training features: {feature_path}. Run --stage features first.")
    if not config.train_ground_truth.exists():
        raise FileNotFoundError(f"Missing training ground truth: {config.train_ground_truth}")
    if not config.train_source1.exists():
        raise FileNotFoundError(f"Missing training Source 1 file: {config.train_source1}")
    if args.m9_folds < 2:
        raise ValueError("--m9-folds must be >= 2")
    if not 0.0 < args.m9_holdout_fraction < 0.5:
        raise ValueError("--m9-holdout-fraction must be > 0 and < 0.5")
    evaluator = FinalEvaluator(FinalEvaluationConfig(
        n_folds=args.m9_folds,
        holdout_fraction=args.m9_holdout_fraction,
        calibration_fraction=args.m9_calibration_fraction,
        calibration_method=args.m9_calibration_method,
        threshold_grid_size=args.m9_threshold_grid_size,
        target_precision=args.m9_target_precision,
        min_recall_at_target=args.m9_min_recall,

        model_type=args.m9_model,

        hgb_max_iter=args.m9_hgb_max_iter,
        hgb_learning_rate=args.m9_hgb_learning_rate,
        hgb_max_leaf_nodes=args.m9_hgb_max_leaf_nodes,
        hgb_min_samples_leaf=args.m9_hgb_min_samples_leaf,
        hgb_l2_regularization=args.m9_hgb_l2,

        hard_negative_ratio=args.hard_negative_ratio,
        max_positive_rows=args.m9_max_positive_rows,
        max_negative_rows=args.m9_max_negative_rows,
        max_calibration_rows=args.m9_max_calibration_rows,
        top_errors=args.m9_top_errors,
        m8_min_edge_score=args.m8_min_edge_score,
        m8_max_cluster_size=args.m8_max_cluster_size,
        m8_reject_country_conflict=not args.m8_no_country_veto,
        m8_reject_explicit_negative=not args.m8_no_explicit_negative_veto,
        m8_explicit_negative_threshold=args.m8_explicit_negative_threshold,
        evaluation_s1_source=getattr(args, "evaluation_s1_source", "feature"),
        evaluation_s1_file=str(args.evaluation_s1_file) if getattr(args, "evaluation_s1_file", None) else None,
    ))
    report = evaluator.run(
        feature_path=feature_path,
        ground_truth_path=config.train_ground_truth,
        source1_path=config.train_source1,
        artifact_dir=config.artifact_dir,
    )
    oof = report["oof"]["selected_metrics"]
    holdout = report["holdout"]["metrics"]
    LOGGER.info(
        "M9 model: %s",
        args.m9_model,
    )
    LOGGER.info(
        "M9 OOF: F0.5=%.6f precision=%.6f recall=%.6f threshold=%.6f",
        oof["macro_f05"], oof["macro_precision"], oof["macro_recall"], oof["threshold"],
    )
    LOGGER.info(
        "M9 holdout: F0.5=%.6f precision=%.6f recall=%.6f gap=%.4f pp status=%s",
        holdout["macro_f05"], holdout["macro_precision"], holdout["macro_recall"],
        report["generalization"]["absolute_gap_percentage_points"],
        report["anti_overfit_status"]["status"],
    )
    LOGGER.info("M9 report: %s", config.artifact_dir / "diagnostics" / "final_evaluation.json")
    return 0


def run_m9_test(config: PipelineConfig, args) -> int:
    """Score hidden/unlabeled test data using only the frozen M9 artifacts."""
    model_path = config.artifact_dir / "models" / "m9_frozen_match_model.pkl"
    policy_path = config.artifact_dir / "models" / "m9_frozen_decision_policy.pkl"
    feature_path = config.artifact_dir / "features" / "test_pair_features.tsv"
    for path in (model_path, policy_path, feature_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing M9 final artifact: {path}. Run --stage final-evaluation first.")

    model = MatchModel.load(model_path)
    engine = DecisionEngine.load(policy_path, model)
    prediction_path = config.artifact_dir / "predictions" / "m9_test_pair_decisions.tsv"
    score_stats = engine.score_file(feature_path, prediction_path)

    consolidator = EntityConsolidator(ConsolidationConfig(
        min_edge_score=args.m8_min_edge_score,
        reject_country_conflict=not args.m8_no_country_veto,
        reject_explicit_negative=not args.m8_no_explicit_negative_veto,
        explicit_negative_score_threshold=args.m8_explicit_negative_threshold,
        max_cluster_size=args.m8_max_cluster_size,
    ))
    assignments = config.artifact_dir / "predictions" / "m9_test_entity_assignments.tsv"
    edges = config.artifact_dir / "predictions" / "m9_test_accepted_edges.tsv"
    resolution = config.artifact_dir / "predictions" / "m9_test_entity_resolution.tsv"
    report_path = config.artifact_dir / "diagnostics" / "m9_test_entity_consolidation.json"
    report = consolidator.run(prediction_path, assignments, edges, report_path)
    state = consolidator.build_clusters(prediction_path)
    report["s1_resolution"] = consolidator.write_s1_resolution(state, resolution)
    report["score_stats"] = score_stats
    report["frozen_model"] = str(model_path)
    report["frozen_policy"] = str(policy_path)
    report["test_labels_used"] = False
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOGGER.info("M9 test scoring: rows=%,d matches=%,d", score_stats["rows"], score_stats["matches"])
    LOGGER.info("M9 test resolution: %s", resolution)
    return 0


def main() -> int:
    args = parse_args()
    if args.config is not None and not args.config.is_file():
        logging.basicConfig(level=logging.INFO)
        LOGGER.error("Configuration file not found: %s", args.config)
        return 1

    config = PipelineConfig.from_root(args.root)
    configure_logging(config.log_dir, args.verbose)

    LOGGER.info("Project root: %s", config.project_root)
    LOGGER.info("Dataset directory: %s", config.dataset_dir)

    rc = validate_dataset(config)
    if rc:
        return rc

    if args.stage == "verify":
        return 0

    if args.stage == "validate":
        matching_results = config.project_root / "output" / "matching_results.tsv"
        if not matching_results.is_file():
            LOGGER.error("Validation failed: matching_results.tsv not found at %s", matching_results)
            return 1
        return 0

    if args.stage == "profile":
        return run_profile(config, args.exact_id_uniqueness)

    if args.stage == "normalize":
        return run_normalization_sample(config, args.sample_size)

    if args.stage == "block":
        return run_blocking(config, args)

    if args.stage == "evaluate-blocking":
        return run_blocking_evaluation(config)

    if args.stage == "features":
        return run_pairwise_features(config, args.feature_db, not args.no_labels)

    if args.stage == "train-model":
        return run_matching_model(config, args.model_path, args.validation_fraction, args.hard_negative_ratio,
                                  args.max_positive_rows, args.max_negative_rows, args.max_validation_rows, args.thresholds)

    if args.stage == "predict":
        return run_prediction(config, args.model_path)

    if args.stage == "consolidate":
        return run_consolidation(config, args.model_path, args.decision_path, "test",
                                 args.m8_min_edge_score, args.m8_max_cluster_size, args.m8_no_country_veto,
                                 args.m8_no_explicit_negative_veto, args.m8_explicit_negative_threshold)

    if args.stage == "entity-resolution":
        rc = run_decision_calibration(config, args.model_path, args.decision_path, args.calibration_method,
                                      args.calibration_fraction, args.target_precision, args.min_recall_at_target,
                                      args.threshold_grid_size, args.no_country_veto, args.m7_fp_limit)
        if rc:
            return rc
        rc = run_consolidation(config, args.model_path, args.decision_path, "train",
                               args.m8_min_edge_score, args.m8_max_cluster_size, args.m8_no_country_veto,
                                 args.m8_no_explicit_negative_veto, args.m8_explicit_negative_threshold)
        if rc:
            return rc
        return run_consolidation(config, args.model_path, args.decision_path, "test",
                                 args.m8_min_edge_score, args.m8_max_cluster_size, args.m8_no_country_veto,
                                 args.m8_no_explicit_negative_veto, args.m8_explicit_negative_threshold)


    if args.stage in {"final-evaluation", "evaluate-final"}:
        return run_final_evaluation(config, args)

    if args.stage in {"finalize-test", "m9-test"}:
        return run_m9_test(config, args)

    if args.stage == "calibrate":
        return run_decision_calibration(config, args.model_path, args.decision_path, args.calibration_method,
                                        args.calibration_fraction, args.target_precision, args.min_recall_at_target,
                                        args.threshold_grid_size, args.no_country_veto, args.m7_fp_limit)

    if args.stage == "decision":
        rc = run_decision_calibration(config, args.model_path, args.decision_path, args.calibration_method,
                                      args.calibration_fraction, args.target_precision, args.min_recall_at_target,
                                      args.threshold_grid_size, args.no_country_veto, args.m7_fp_limit)
        if rc:
            return rc
        model_path = args.model_path or config.artifact_dir / "models" / "match_model.pkl"
        decision_path = args.decision_path or config.artifact_dir / "models" / "decision_policy.pkl"
        engine = DecisionEngine.load(decision_path, MatchModel.load(model_path))
        output_path = config.artifact_dir / "predictions" / "test_pair_decisions.tsv"
        stats = engine.score_file(config.artifact_dir / "features" / "test_pair_features.tsv", output_path)
        report_path = config.artifact_dir / "diagnostics" / "test_pair_decisions.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("M7 test decisions: rows=%,d matches=%,d output=%s", stats["rows"], stats["matches"], output_path)
        return 0

    if args.stage == "model":
        rc = run_matching_model(config, args.model_path, args.validation_fraction, args.hard_negative_ratio,
                                args.max_positive_rows, args.max_negative_rows, args.max_validation_rows, args.thresholds)
        if rc:
            return rc
        rc = run_decision_calibration(config, args.model_path, args.decision_path, args.calibration_method,
                                      args.calibration_fraction, args.target_precision, args.min_recall_at_target,
                                      args.threshold_grid_size, args.no_country_veto, args.m7_fp_limit)
        if rc:
            return rc
        rc = run_consolidation(config, args.model_path, args.decision_path, "train",
                               args.m8_min_edge_score, args.m8_max_cluster_size, args.m8_no_country_veto,
                                 args.m8_no_explicit_negative_veto, args.m8_explicit_negative_threshold)
        if rc:
            return rc
        return run_consolidation(config, args.model_path, args.decision_path, "test",
                                 args.m8_min_edge_score, args.m8_max_cluster_size, args.m8_no_country_veto,
                                 args.m8_no_explicit_negative_veto, args.m8_explicit_negative_threshold)

    if args.stage == "blocking":
        rc = run_blocking(config, args)
        if rc:
            return rc
        return run_blocking_evaluation(config)

    if args.stage == "all":
        rc = run_profile(config, args.exact_id_uniqueness)
        if rc:
            return rc
        rc = run_normalization_sample(config, args.sample_size)
        if rc:
            return rc
        rc = run_blocking(config, args)
        if rc:
            return rc
        rc = run_blocking_evaluation(config)
        if rc:
            return rc
        rc = run_pairwise_features(config, args.feature_db, not args.no_labels)
        if rc:
            return rc
        rc = run_matching_model(config, args.model_path, args.validation_fraction, args.hard_negative_ratio,
                                args.max_positive_rows, args.max_negative_rows, args.max_validation_rows, args.thresholds)
        if rc:
            return rc
        rc = run_final_evaluation(config, args)
        if rc:
            return rc
        return run_m9_test(config, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
