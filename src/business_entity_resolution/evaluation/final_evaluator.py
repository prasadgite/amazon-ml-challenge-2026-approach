from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import pickle
import statistics
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    csv.field_size_limit(2147483647)
except Exception:
    pass

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:  # pragma: no cover
    np = None
    LogisticRegression = None
    HistGradientBoostingClassifier = None
    average_precision_score = None
    brier_score_loss = None
    precision_score = None
    recall_score = None
    Pipeline = None
    StandardScaler = None
    IMPORT_ERROR = exc

from business_entity_resolution.decision.calibration import ProbabilityCalibrator, DecisionConfig, DecisionEngine
from business_entity_resolution.matching.model import FEATURE_COLUMNS, MatchModel, MatchModelConfig, _hardness
from business_entity_resolution.consolidation.consolidator import EntityConsolidator, ConsolidationConfig, ConsolidationEvaluator


@dataclass(frozen=True)
class FinalEvaluationConfig:
    """Configuration for leakage-safe M9 evaluation.

    The dataset is partitioned by S1 entity, never by pair row.  A deterministic
    holdout is reserved first and is never used for model/threshold selection.
    The remaining development population is evaluated with grouped OOF folds.
    """

    n_folds: int = 5
    holdout_fraction: float = 0.15
    calibration_fraction: float = 0.25
    calibration_method: str = "platt"
    threshold_grid_size: int = 401
    target_precision: float = 0.99
    min_recall_at_target: float = 0.0
    random_state: int = 42

    model_type: str = "logistic"

    # M9.1 HistGradientBoosting configuration.
    # These defaults are deliberately conservative because the feature matrix
    # can contain ~1.5M pair records.
    hgb_max_iter: int = 300
    hgb_learning_rate: float = 0.05
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 1.0

    hard_negative_ratio: int = 4
    max_positive_rows: int = 100_000
    max_negative_rows: int = 400_000
    max_calibration_rows: int = 100_000
    batch_size: int = 5000
    top_errors: int = 100
    m8_min_edge_score: float = 0.0
    m8_max_cluster_size: int = 1000
    m8_reject_country_conflict: bool = True
    m8_reject_explicit_negative: bool = True
    m8_explicit_negative_threshold: float = 0.99

    # M9.2 evaluation cohort control
    evaluation_s1_source: str = "feature"  # "feature" or "file"
    evaluation_s1_file: str | None = None

    # M9.3-E feature column customization
    feature_columns: tuple[str, ...] | None = None



class FinalEvaluator:
    """M9: generalization-safe evaluation and frozen policy generation.

    Key guarantees:
      * S1 entities are never split across train/validation/holdout.
      * The final holdout is never used for threshold selection.
      * OOF predictions are produced by models that did not train on those S1s.
      * Calibration is nested inside each OOF fold: the calibrator never sees
        the outer validation S1s.
      * Threshold selection is based on OOF S1-level macro F0.5.
    """

    def __init__(self, config: FinalEvaluationConfig | None = None):
        self.config = config or FinalEvaluationConfig()
        self._require_dependencies()

    @staticmethod
    def _require_dependencies() -> None:
        if (
            np is None
            or LogisticRegression is None
            or HistGradientBoostingClassifier is None
            or ProbabilityCalibrator is None
        ):
            raise RuntimeError("M9 requires scikit-learn") from IMPORT_ERROR

    @staticmethod
    def _read_rows(path: Path) -> Iterable[dict[str, str]]:
        def _clean_lines(fh):
            """Strip NUL bytes and stop at first NUL-only line."""
            for line in fh:
                cleaned = line.replace("\x00", "")
                if not cleaned or cleaned.isspace():
                    continue
                yield cleaned

        with path.open("r", encoding="utf-8", newline="", errors="replace") as handle:
            yield from csv.DictReader(_clean_lines(handle), delimiter="\t")

    @staticmethod
    def _hash(value: str, seed: int) -> float:
        raw = f"{seed}:{value}".encode("utf-8")
        digest = hashlib.blake2b(raw, digest_size=8).digest()
        return int.from_bytes(digest, "big") / float(2**64)

    def _role(self, s1_id: str) -> tuple[str, int | None]:
        h = self._hash(s1_id, self.config.random_state)
        if h < self.config.holdout_fraction:
            return "holdout", None
        development_h = (h - self.config.holdout_fraction) / (1.0 - self.config.holdout_fraction)
        fold = min(self.config.n_folds - 1, int(development_h * self.config.n_folds))
        return "development", fold

    def _matrix(self, rows: list[dict[str, str]]):
        columns = self.config.feature_columns or FEATURE_COLUMNS
        return np.asarray([[float(row.get(c, 0.0) or 0.0) for c in columns] for row in rows], dtype=float)

    @staticmethod
    def _safe_int(row: dict[str, str], key: str, default: int = 0) -> int:
        try:
            return int(float(row.get(key, default) or default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _fit_pipeline(self, rows: list[dict[str, str]]):
        if not rows:
            raise ValueError("No rows available to fit M9 model")

        positives = sum(int(r["label"]) for r in rows)
        negatives = len(rows) - positives

        if positives == 0 or negatives == 0:
            raise ValueError(
                "M9 training fold needs both classes: "
                f"positives={positives}, negatives={negatives}"
            )

        model_type = self.config.model_type.lower()

        if model_type == "logistic":
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=self.config.random_state,
                        solver="liblinear",
                    ),
                ),
            ])

        elif model_type == "hgb":
            # HistGradientBoosting is tree based, so feature scaling is unnecessary.
            #
            # These parameters are intentionally conservative for the ~1.5M-row
            # pairwise feature matrix. The objective of M9.1 is to test whether
            # nonlinear feature interactions recover recall without materially
            # damaging precision.
            pipeline = HistGradientBoostingClassifier(
                max_iter=self.config.hgb_max_iter,
                learning_rate=self.config.hgb_learning_rate,
                max_leaf_nodes=self.config.hgb_max_leaf_nodes,
                min_samples_leaf=self.config.hgb_min_samples_leaf,
                l2_regularization=self.config.hgb_l2_regularization,
                random_state=self.config.random_state,
                early_stopping=False,
            )

        else:
            raise ValueError(
                "model_type must be 'logistic' or 'hgb'"
            )

        pipeline.fit(
            self._matrix(rows),
            [int(r["label"]) for r in rows],
        )

        return pipeline

    def _select_training_rows(self, path: Path, excluded_fold: int, role: str = "development") -> tuple[list[dict], list[dict], dict]:
        import random
        rng = random.Random(self.config.random_state + excluded_fold * 997)
        fit_pos: list[dict] = []
        fit_neg: list[dict] = []
        calibration: list[dict] = []
        pos_seen = neg_seen = cal_seen = 0

        for row in self._read_rows(path):
            if not row.get("label"):
                continue
            row_role, fold = self._role(row["source1_id"])
            if row_role != role or fold == excluded_fold:
                continue
            # Nested calibration split. The model never trains on these rows.
            cal_bucket = self._hash(row["source1_id"], self.config.random_state + 7919 + excluded_fold)
            if cal_bucket < self.config.calibration_fraction:
                cal_seen += 1
                self._reservoir_add(calibration, row, self.config.max_calibration_rows, cal_seen, rng)
                continue
            label = int(row["label"])
            if label:
                pos_seen += 1
                self._reservoir_add(fit_pos, row, self.config.max_positive_rows, pos_seen, rng)
            else:
                neg_seen += 1
                self._reservoir_add(fit_neg, row, self.config.max_negative_rows, neg_seen, rng)

        fit_neg.sort(key=_hardness, reverse=True)
        target_neg = min(len(fit_neg), max(1, len(fit_pos) * self.config.hard_negative_ratio)) if fit_pos else 0
        if target_neg < len(fit_neg):
            hard = fit_neg[:target_neg]
            tail_n = min(max(0, target_neg // 10), len(fit_neg) - target_neg)
            if tail_n:
                hard.extend(rng.sample(fit_neg[target_neg:], tail_n))
            fit_neg = hard
        return fit_pos + fit_neg, calibration, {
            "positive_seen": pos_seen,
            "negative_seen": neg_seen,
            "calibration_seen": cal_seen,
            "fit_rows": len(fit_pos) + len(fit_neg),
            "fit_positive": len(fit_pos),
            "fit_negative": len(fit_neg),
            "calibration_rows": len(calibration),
        }

    @staticmethod
    def _reservoir_add(items: list[dict], row: dict, limit: int, seen: int, rng) -> None:
        if limit <= 0:
            return
        if len(items) < limit:
            items.append(row)
            return
        j = rng.randrange(seen)
        if j < limit:
            items[j] = row

    def _score_rows(self, pipeline: Pipeline, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=float)
        return pipeline.predict_proba(self._matrix(rows))[:, 1]

    def _fit_fold(self, feature_path: Path, fold: int, output_handle) -> dict:
        fit_rows, calibration_rows, train_stats = self._select_training_rows(feature_path, fold)
        pipeline = self._fit_pipeline(fit_rows)
        if not calibration_rows:
            raise ValueError(f"Fold {fold}: nested calibration population is empty")
        cal_raw = self._score_rows(pipeline, calibration_rows)
        cal_labels = np.asarray([int(r["label"]) for r in calibration_rows], dtype=int)
        calibrator = ProbabilityCalibrator(self.config.calibration_method)
        calibrator.fit(cal_raw, cal_labels)

        rows = 0
        s1_count: set[str] = set()
        positives = 0
        with feature_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            batch: list[dict[str, str]] = []
            for row in reader:
                if not row.get("label"):
                    continue
                role, row_fold = self._role(row["source1_id"])
                if role != "development" or row_fold != fold:
                    continue
                batch.append(row)
                if len(batch) >= self.config.batch_size:
                    rows += self._write_scored_batch(pipeline, calibrator, fold, batch, output_handle, s1_count)
                    positives += sum(int(x["label"]) for x in batch)
                    batch.clear()
            if batch:
                rows += self._write_scored_batch(pipeline, calibrator, fold, batch, output_handle, s1_count)
                positives += sum(int(x["label"]) for x in batch)

        return {
            "fold": fold,
            "train": train_stats,
            "validation_rows": rows,
            "validation_s1": len(s1_count),
            "validation_positive_rows": positives,
            "calibration_brier": float(brier_score_loss(cal_labels, calibrator.predict(cal_raw))) if len(np.unique(cal_labels)) > 1 else None,
        }

    def _write_scored_batch(self, pipeline, calibrator, fold: int, batch: list[dict[str, str]], output_handle, s1_count: set[str]) -> int:
        raw = self._score_rows(pipeline, batch)
        calibrated = calibrator.predict(raw)
        writer = output_handle
        for row, raw_score, score in zip(batch, raw, calibrated):
            s1_count.add(row["source1_id"])
            writer.writerow({
                "source1_id": row["source1_id"],
                "target_source": row["target_source"],
                "target_id": row["target_id"],
                "raw_score": f"{float(raw_score):.10f}",
                "calibrated_score": f"{float(score):.10f}",
                "label": row.get("label", ""),
                "country_conflict": row.get("country_conflict", "0"),
                "strong_evidence_count": row.get("strong_evidence_count", "0"),
                "name_jaro_winkler": row.get("name_jaro_winkler", "0"),
                "name_token_jaccard": row.get("name_token_jaccard", "0"),
                "address_token_jaccard": row.get("address_token_jaccard", "0"),
                "domain_exact": row.get("domain_exact", "0"),
                "alias_exact": row.get("alias_exact", "0"),
                "blocking_hit_count": row.get("blocking_hit_count", "0"),
            })
        return len(batch)

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _ground_truth(path: Path) -> dict[str, set[str]]:
        truth: dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                s1 = row.get("source1_entity_id", "").strip()
                values = {x.strip() for x in row.get("matched_entity_ids", "").split(",") if x.strip()}
                truth[s1] = values
        return truth

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _source1_ids(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            field = "entity_id"
            if field not in (reader.fieldnames or []):
                raise ValueError(f"Missing entity_id in {path}")
            return [row[field].strip() for row in reader if row.get(field)]

    def _resolve_s1_ids(self, source1_target: Path | list[str]) -> list[str]:
        if isinstance(source1_target, list):
            return source1_target
        return self._source1_ids(source1_target)

    def _get_evaluated_s1_ids(self, feature_path: Path, source1_path: Path) -> list[str]:
        source_mode = getattr(self.config, "evaluation_s1_source", "feature")
        if source_mode == "feature":
            seen: set[str] = set()
            order: list[str] = []
            for row in self._read_rows(feature_path):
                s1 = (row.get("source1_id") or "").strip()
                if s1 and s1 not in seen:
                    seen.add(s1)
                    order.append(s1)
            if order:
                return order
            return self._source1_ids(source1_path)
        elif source_mode == "file":
            custom_file = getattr(self.config, "evaluation_s1_file", None)
            target = Path(custom_file) if custom_file else source1_path
            return self._source1_ids(target)
        return self._source1_ids(source1_path)

    @staticmethod
    def _prediction_map(decision_path: Path, threshold: float) -> dict[str, set[str]]:
        result: dict[str, set[str]] = defaultdict(set)
        with decision_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if float(row.get("calibrated_score", 0.0) or 0.0) >= threshold:
                    result[row["source1_id"]].add(row["target_id"])
        return result

    @staticmethod
    def _f05(precision: float, recall: float) -> float:
        return (1.25 * precision * recall / (0.25 * precision + recall)) if (0.25 * precision + recall) else 0.0

    def _s1_metrics(self, prediction_path: Path, truth_path: Path, source1_target: Path | list[str], threshold: float, partition: str | None = None) -> dict:
        truth = self._ground_truth(truth_path)
        all_s1 = self._resolve_s1_ids(source1_target)
        pred = self._prediction_map(prediction_path, threshold)

        part_s1 = [s for s in all_s1 if partition is None or self._role(s)[0] == partition]
        total_s1 = len(part_s1)
        part_set = set(part_s1)
        total_singletons = sum(1 for s in part_s1 if not truth.get(s))
        total_with_truth = total_s1 - total_singletons
        total_truth_pairs = sum(len(truth.get(s, set())) for s in part_s1)

        active_f05_sum = 0.0
        active_p_sum = 0.0
        active_r_sum = 0.0
        pair_tp = pair_fp = 0
        singleton_fp = 0
        active_count = 0

        for s1, predicted in pred.items():
            if s1 not in part_set:
                continue
            active_count += 1
            actual = truth.get(s1, set())
            tp = len(actual & predicted)
            fp = len(predicted - actual)
            pair_tp += tp
            pair_fp += fp
            if not actual:
                singleton_fp += 1
                active_r_sum += 1.0
            else:
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / len(actual) if actual else 0.0
                active_p_sum += p
                active_r_sum += r
                active_f05_sum += self._f05(p, r)

        inactive_singletons = total_singletons - singleton_fp
        macro_f05 = (active_f05_sum + inactive_singletons) / total_s1 if total_s1 else 0.0
        macro_precision = (active_p_sum + inactive_singletons) / total_s1 if total_s1 else 0.0
        macro_recall = (active_r_sum + inactive_singletons) / total_s1 if total_s1 else 0.0
        pair_fn = total_truth_pairs - pair_tp

        return {
            "threshold": float(threshold),
            "macro_f05": macro_f05,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "pair_tp": pair_tp,
            "pair_fp": pair_fp,
            "pair_fn": pair_fn,
            "pair_precision": pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0,
            "pair_recall": pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else 0.0,
            "s1_entities": total_s1,
            "s1_with_truth_matches": total_with_truth,
            "s1_singletons": total_singletons,
            "singleton_false_positives": singleton_fp,
            "singleton_correct_empty_predictions": inactive_singletons,
            "predicted_s1_with_matches": active_count,
        }

    def _threshold_grid(self) -> list[float]:
        n = max(101, int(self.config.threshold_grid_size))
        return [i / (n - 1) for i in range(n)]

    def _select_threshold(self, prediction_path: Path, truth_path: Path, source1_target: Path | list[str]) -> tuple[float, dict, list[dict]]:
        """Select threshold using one score sort rather than N full-file scans."""
        truth = self._ground_truth(truth_path)
        dev_s1 = [x for x in self._resolve_s1_ids(source1_target) if self._role(x)[0] == "development"]
        actual_sizes = {s1: len(truth.get(s1, set())) for s1 in dev_s1}
        tp_by_s1: dict[str, int] = defaultdict(int)
        pred_by_s1: dict[str, int] = defaultdict(int)
        grid = sorted(self._threshold_grid(), reverse=True)
        metrics_by_threshold: dict[float, dict] = {}
        db_path = prediction_path.with_suffix(".threshold_index.sqlite")
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("CREATE TABLE scores (source1_id TEXT NOT NULL, target_id TEXT NOT NULL, score REAL NOT NULL)")
            with prediction_path.open("r", encoding="utf-8", newline="") as handle:
                rows = []
                for row in csv.DictReader(handle, delimiter="\t"):
                    if self._role(row["source1_id"])[0] != "development":
                        continue
                    rows.append((row["source1_id"], row["target_id"], float(row["calibrated_score"])))
                    if len(rows) >= 10000:
                        conn.executemany("INSERT INTO scores VALUES (?,?,?)", rows)
                        rows.clear()
                if rows:
                    conn.executemany("INSERT INTO scores VALUES (?,?,?)", rows)
            conn.execute("CREATE INDEX idx_scores_score ON scores(score DESC)")
            conn.commit()

            cursor = conn.execute("SELECT source1_id, target_id, score FROM scores ORDER BY score DESC")
            next_row = cursor.fetchone()
            total_dev = len(dev_s1)
            total_singletons = sum(1 for s in dev_s1 if actual_sizes[s] == 0)
            total_with_truth = total_dev - total_singletons
            total_truth_pairs = sum(actual_sizes.values())

            for threshold in grid:
                while next_row is not None and float(next_row[2]) >= threshold:
                    s1, target, score = next_row
                    pred_by_s1[s1] += 1
                    if target in truth.get(s1, set()):
                        tp_by_s1[s1] += 1
                    next_row = cursor.fetchone()

                active_f05_sum = 0.0
                active_p_sum = 0.0
                active_r_sum = 0.0
                pair_tp = 0
                pair_fp = 0
                singleton_fp = 0

                for s1, predicted in pred_by_s1.items():
                    actual = actual_sizes.get(s1, 0)
                    tp = tp_by_s1.get(s1, 0)
                    fp = predicted - tp
                    pair_tp += tp
                    pair_fp += fp
                    if actual == 0:
                        singleton_fp += 1
                        active_r_sum += 1.0
                    else:
                        p = tp / (tp + fp) if tp + fp else 0.0
                        r = tp / (tp + (actual - tp)) if actual else 0.0
                        active_p_sum += p
                        active_r_sum += r
                        active_f05_sum += self._f05(p, r)

                inactive_singletons = total_singletons - singleton_fp
                macro_f05 = (active_f05_sum + inactive_singletons) / total_dev if total_dev else 0.0
                macro_precision = (active_p_sum + inactive_singletons) / total_dev if total_dev else 0.0
                macro_recall = (active_r_sum + inactive_singletons) / total_dev if total_dev else 0.0
                pair_fn = total_truth_pairs - pair_tp

                metrics_by_threshold[threshold] = {
                    "threshold": float(threshold),
                    "macro_f05": macro_f05,
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "pair_tp": pair_tp,
                    "pair_fp": pair_fp,
                    "pair_fn": pair_fn,
                    "pair_precision": pair_tp / (pair_tp + pair_fp) if pair_tp + pair_fp else 0.0,
                    "pair_recall": pair_tp / (pair_tp + pair_fn) if pair_tp + pair_fn else 0.0,
                    "s1_entities": total_dev,
                    "s1_with_truth_matches": total_with_truth,
                    "s1_singletons": total_singletons,
                    "singleton_false_positives": singleton_fp,
                    "singleton_correct_empty_predictions": inactive_singletons,
                    "predicted_s1_with_matches": len(pred_by_s1),
                }
            grid_metrics = [metrics_by_threshold[t] for t in sorted(metrics_by_threshold)]
        finally:
            conn.close()
            try:
                db_path.unlink()
            except FileNotFoundError:
                pass

        eligible = [m for m in grid_metrics if m["macro_precision"] >= self.config.target_precision and m["macro_recall"] >= self.config.min_recall_at_target]
        if eligible:
            selected = max(eligible, key=lambda x: (x["macro_f05"], x["macro_precision"], -x["threshold"]))
            status = "target_precision_reached"
        else:
            selected = max(grid_metrics, key=lambda x: (x["macro_f05"], x["macro_precision"], -x["threshold"]))
            status = "target_precision_not_reached"
        return float(selected["threshold"]), {
            "status": status,
            "target_precision": self.config.target_precision,
            "min_recall_at_target": self.config.min_recall_at_target,
            "selected": selected,
            "eligible_count": len(eligible),
        }, grid_metrics

    def _fold_metrics(self, prediction_path: Path, truth_path: Path, source1_target: Path | list[str], threshold: float) -> dict[int, dict]:
        truth = self._ground_truth(truth_path)
        all_s1 = self._resolve_s1_ids(source1_target)
        fold_s1_counts: dict[int, int] = defaultdict(int)
        fold_singleton_counts: dict[int, int] = defaultdict(int)
        for s1 in all_s1:
            role, fold = self._role(s1)
            if role == "development" and fold is not None:
                fold_s1_counts[fold] += 1
                if not truth.get(s1):
                    fold_singleton_counts[fold] += 1

        pred: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if float(row.get("calibrated_score", 0.0) or 0.0) < threshold:
                    continue
                role, fold = self._role(row["source1_id"])
                if role == "development" and fold is not None:
                    pred[fold][row["source1_id"]].add(row["target_id"])

        result = {}
        for fold in sorted(fold_s1_counts):
            total_fold_s1 = fold_s1_counts[fold]
            total_fold_singletons = fold_singleton_counts[fold]
            active_f05_sum = 0.0
            fold_singleton_fp = 0
            for s1, predicted in pred[fold].items():
                actual = truth.get(s1, set())
                if not actual:
                    fold_singleton_fp += 1
                else:
                    tp = len(actual & predicted)
                    fp = len(predicted - actual)
                    p = tp / (tp + fp) if tp + fp else 0.0
                    r = tp / len(actual) if actual else 0.0
                    active_f05_sum += self._f05(p, r)
            inactive_singletons = total_fold_singletons - fold_singleton_fp
            macro_f05 = (active_f05_sum + inactive_singletons) / total_fold_s1 if total_fold_s1 else 0.0
            result[fold] = {
                "s1_entities": total_fold_s1,
                "macro_f05": macro_f05,
            }
        return result

    def _score_quality(self, prediction_path: Path) -> dict:
        labels = []
        scores = []
        buckets = Counter()
        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels.append(int(row.get("label", "0") or 0))
                score = float(row.get("calibrated_score", 0.0) or 0.0)
                scores.append(score)
                if score < 0.90:
                    key = "<0.90"
                elif score < 0.95:
                    key = "0.90-0.95"
                elif score < 0.99:
                    key = "0.95-0.99"
                elif score < 0.995:
                    key = "0.99-0.995"
                elif score < 0.999:
                    key = "0.995-0.999"
                else:
                    key = "0.999-1.0"
                buckets[key] += 1
        arr_y = np.asarray(labels, dtype=int)
        arr_s = np.asarray(scores, dtype=float)
        result = {
            "rows": len(labels),
            "positive_rows": int(arr_y.sum()) if len(arr_y) else 0,
            "score_buckets": dict(buckets),
        }
        if len(arr_y) and len(np.unique(arr_y)) > 1:
            result.update({
                "average_precision": float(average_precision_score(arr_y, arr_s)),
                "brier": float(brier_score_loss(arr_y, arr_s)),
            })
        return result

    def _error_taxonomy(self, prediction_path: Path, truth_path: Path, threshold: float) -> dict:
        truth = self._ground_truth(truth_path)
        fp_examples = []
        fn_examples = []
        counts = Counter()
        with prediction_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                score = float(row.get("calibrated_score", 0.0) or 0.0)
                actual = row["target_id"] in truth.get(row["source1_id"], set())
                if score >= threshold and not actual:
                    reason = self._classify_false_positive(row)
                    counts[reason] += 1
                    if len(fp_examples) < self.config.top_errors:
                        fp_examples.append({**row, "error_type": reason})
                elif score < threshold and actual:
                    reason = self._classify_false_negative(row)
                    counts[reason] += 1
                    if len(fn_examples) < self.config.top_errors:
                        fn_examples.append({**row, "error_type": reason})
        return {
            "counts": dict(counts),
            "false_positive_examples": sorted(fp_examples, key=lambda x: float(x["calibrated_score"]), reverse=True),
            "false_negative_examples": sorted(fn_examples, key=lambda x: float(x["calibrated_score"]), reverse=True),
        }

    def _classify_false_positive(self, row: dict[str, str]) -> str:
        if self._safe_int(row, "country_conflict"):
            return "country_conflict"
        if self._safe_float(row, "domain_exact") or self._safe_float(row, "alias_exact"):
            return "alias_or_domain_collision"
        if self._safe_float(row, "name_jaro_winkler") >= 0.95 and self._safe_float(row, "name_token_jaccard") >= 0.8:
            if self._safe_float(row, "address_token_jaccard") < 0.5:
                return "name_collision"
        if self._safe_float(row, "address_token_jaccard") >= 0.8 and self._safe_float(row, "name_jaro_winkler") < 0.8:
            return "address_collision"
        if self._safe_int(row, "strong_evidence_count") <= 0:
            return "weak_evidence"
        return "generic_pairwise_false_positive"

    def _classify_false_negative(self, row: dict[str, str]) -> str:
        if self._safe_int(row, "blocking_hit_count") <= 0:
            return "blocking_miss"
        if self._safe_int(row, "country_conflict"):
            return "country_conflict"
        if self._safe_float(row, "domain_exact") or self._safe_float(row, "alias_exact"):
            return "strong_identity_below_threshold"
        if self._safe_int(row, "strong_evidence_count") <= 0:
            return "weak_evidence"
        return "generic_pairwise_false_negative"

    def _write_decisions(self, scored_path: Path, output_path: Path, threshold: float) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = matches = 0
        with scored_path.open("r", encoding="utf-8", newline="") as source, output_path.open("w", encoding="utf-8", newline="") as dest:
            reader = csv.DictReader(source, delimiter="\t")
            writer = csv.DictWriter(dest, fieldnames=[
                "source1_id", "target_source", "target_id", "raw_score", "calibrated_score",
                "decision", "decision_reason", "label", "country_conflict", "strong_evidence_count",
            ], delimiter="\t")
            writer.writeheader()
            for row in reader:
                score = float(row["calibrated_score"])
                if score < threshold:
                    decision, reason = "NON_MATCH", "below_threshold"
                elif self.config.m8_reject_country_conflict and self._safe_int(row, "country_conflict"):
                    decision, reason = "NON_MATCH", "country_conflict_veto"
                else:
                    decision, reason = "MATCH", "threshold_pass"
                writer.writerow({**{k: row.get(k, "") for k in writer.fieldnames}, "decision": decision, "decision_reason": reason})
                rows += 1
                matches += int(decision == "MATCH")
        return {"rows": rows, "matches": matches, "output": str(output_path)}

    def _write_partition_truth(self, source1_target: Path | list[str], truth_path: Path, output_path: Path, partition: str) -> Path:
        truth = self._ground_truth(truth_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t")
            writer.writeheader()
            for s1 in self._resolve_s1_ids(source1_target):
                if self._role(s1)[0] != partition:
                    continue
                writer.writerow({"source1_entity_id": s1, "matched_entity_ids": ",".join(sorted(truth.get(s1, set())))})
        return output_path

    def _cluster_evaluation(self, decision_path: Path, truth_path: Path, work_dir: Path) -> dict:
        consolidator = EntityConsolidator(ConsolidationConfig(
            min_edge_score=self.config.m8_min_edge_score,
            reject_country_conflict=self.config.m8_reject_country_conflict,
            reject_explicit_negative=self.config.m8_reject_explicit_negative,
            explicit_negative_score_threshold=self.config.m8_explicit_negative_threshold,
            max_cluster_size=self.config.m8_max_cluster_size,
        ))
        assignments = work_dir / "entity_assignments.tsv"
        edges = work_dir / "accepted_edges.tsv"
        report = work_dir / "entity_consolidation.json"
        state = consolidator.build_clusters(decision_path)
        consolidator.write_assignments(state, assignments)
        consolidator.write_edges(state, edges)
        evaluation = ConsolidationEvaluator().evaluate(state, truth_path)
        return {
            "report": consolidator.report(state),
            "ground_truth_evaluation": evaluation,
        }

    def _save_frozen_artifacts(self, pipeline, calibrator, threshold: float, output_dir: Path) -> tuple[Path, Path]:
        """Save artifacts in the same format consumed by M6/M7."""
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "m9_frozen_match_model.pkl"
        policy_path = output_dir / "m9_frozen_decision_policy.pkl"

        model = MatchModel(MatchModelConfig(
            validation_fraction=0.20,
            hard_negative_ratio=self.config.hard_negative_ratio,
            max_positive_rows=self.config.max_positive_rows,
            max_negative_rows=self.config.max_negative_rows,
            max_validation_rows=1,
            random_state=self.config.random_state,
        ))
        model.pipeline = pipeline
        model.feature_columns = list(self.config.feature_columns or FEATURE_COLUMNS)
        model.training_summary = {
            "milestone": "M9",
            "selection_rule": "S1-grouped OOF macro-F0.5 with untouched holdout",
            "holdout_fraction": self.config.holdout_fraction,
            "calibration_method": self.config.calibration_method,
            "model_type": self.config.model_type,
        }
        model.save(model_path)

        policy = DecisionEngine(model, DecisionConfig(
            calibration_fraction=self.config.calibration_fraction,
            calibration_method=self.config.calibration_method,
            target_precision=self.config.target_precision,
            random_state=self.config.random_state,
            veto_country_conflict=self.config.m8_reject_country_conflict,
        ))
        policy.calibrator = calibrator
        policy.threshold = float(threshold)
        policy.threshold_selection = {
            "status": "selected_from_m9_oof",
            "threshold": float(threshold),
            "selection_rule": "maximize development OOF macro-F0.5 subject to target precision",
        }
        policy.save(policy_path)
        return model_path, policy_path

    def run(self, feature_path: Path, ground_truth_path: Path, source1_path: Path, artifact_dir: Path) -> dict:
        if not 2 <= self.config.n_folds <= 10:
            raise ValueError("n_folds must be between 2 and 10")
        if not 0.0 < self.config.holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be > 0 and < 0.5")
        if not 0.0 < self.config.calibration_fraction < 0.8:
            raise ValueError("calibration_fraction must be > 0 and < 0.8")
        if self.config.calibration_method not in {"platt", "isotonic", "none"}:
            raise ValueError("calibration_method must be platt, isotonic, or none")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir = artifact_dir / "predictions"
        diagnostic_dir = artifact_dir / "diagnostics"
        model_dir = artifact_dir / "models"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        oof_path = prediction_dir / "m9_oof_predictions.tsv"
        fieldnames = [
            "source1_id", "target_source", "target_id", "raw_score", "calibrated_score", "label",
            "country_conflict", "strong_evidence_count", "name_jaro_winkler", "name_token_jaccard",
            "address_token_jaccard", "domain_exact", "alias_exact", "blocking_hit_count",
        ]
        fold_reports = []
        with oof_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for fold in range(self.config.n_folds):
                fold_report = self._fit_fold(feature_path, fold, writer)
                fold_reports.append(fold_report)

        s1_population = self._get_evaluated_s1_ids(feature_path, source1_path)

        threshold, threshold_selection, threshold_grid = self._select_threshold(oof_path, ground_truth_path, s1_population)
        oof_metrics = threshold_selection["selected"]
        fold_metrics = self._fold_metrics(oof_path, ground_truth_path, s1_population, threshold)
        fold_values = [v["macro_f05"] for v in fold_metrics.values()]
        holdout_s1 = [x for x in s1_population if self._role(x)[0] == "holdout"]

        # Holdout is scored by a model trained only on the development population.
        # A nested calibration split is retained, and the holdout remains completely unseen.
        final_fit_rows, final_cal_rows, final_train_stats = self._select_training_rows(feature_path, -1, role="development")
        final_pipeline = self._fit_pipeline(final_fit_rows)
        if not final_cal_rows:
            raise ValueError("Final development calibration population is empty")
        final_cal_raw = self._score_rows(final_pipeline, final_cal_rows)
        final_cal_labels = np.asarray([int(r["label"]) for r in final_cal_rows], dtype=int)
        final_calibrator = ProbabilityCalibrator(self.config.calibration_method)
        final_calibrator.fit(final_cal_raw, final_cal_labels)

        holdout_path = prediction_dir / "m9_holdout_predictions.tsv"
        with feature_path.open("r", encoding="utf-8", newline="") as source, holdout_path.open("w", encoding="utf-8", newline="") as dest:
            reader = csv.DictReader(source, delimiter="\t")
            writer = csv.DictWriter(dest, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            batch = []
            for row in reader:
                if not row.get("label") or self._role(row["source1_id"])[0] != "holdout":
                    continue
                batch.append(row)
                if len(batch) >= self.config.batch_size:
                    self._write_scored_batch(final_pipeline, final_calibrator, -1, batch, writer, set())
                    batch.clear()
            if batch:
                self._write_scored_batch(final_pipeline, final_calibrator, -1, batch, writer, set())

        holdout_metrics = self._s1_metrics(holdout_path, ground_truth_path, s1_population, threshold, partition="holdout")
        quality = self._score_quality(oof_path)
        errors = self._error_taxonomy(oof_path, ground_truth_path, threshold)
        decision_path = prediction_dir / "m9_oof_decisions.tsv"
        self._write_decisions(oof_path, decision_path, threshold)
        oof_truth_path = self._write_partition_truth(s1_population, ground_truth_path, artifact_dir / "m9_cluster_oof" / "development_ground_truth.tsv", "development")
        oof_cluster = self._cluster_evaluation(decision_path, oof_truth_path, artifact_dir / "m9_cluster_oof")

        model_path, policy_path = self._save_frozen_artifacts(final_pipeline, final_calibrator, threshold, model_dir)

        gap = oof_metrics["macro_f05"] - holdout_metrics["macro_f05"]
        report = {
            "milestone": "M9",
            "purpose": "generalization-safe final evaluation and frozen policy selection",
            "config": asdict(self.config),
            "data_contract": {
                "feature_path": str(feature_path),
                "ground_truth_path": str(ground_truth_path),
                "source1_path": str(source1_path),
                "test_data_used": False,
                "test_labels_used": False,
            },
            "partitioning": {
                "method": "deterministic_blake2b_by_source1",
                "holdout_fraction": self.config.holdout_fraction,
                "n_folds": self.config.n_folds,
                "holdout_s1_entities": len(holdout_s1),
                "development_s1_entities": len(s1_population) - len(holdout_s1),
                "total_evaluated_s1": len(s1_population),
                "evaluation_s1_source": self.config.evaluation_s1_source,
                "no_s1_cross_partition": True,
            },
            "oof": {
                "selected_threshold": threshold,
                "threshold_selection": threshold_selection,
                "selected_metrics": oof_metrics,
                "fold_metrics": fold_metrics,
                "fold_mean_f05": statistics.mean(fold_values) if fold_values else 0.0,
                "fold_std_f05": statistics.pstdev(fold_values) if len(fold_values) > 1 else 0.0,
                "fold_min_f05": min(fold_values) if fold_values else 0.0,
                "fold_max_f05": max(fold_values) if fold_values else 0.0,
                "threshold_grid_size": len(threshold_grid),
            },
            "holdout": {
                "selected_threshold_from_oof": threshold,
                "metrics": holdout_metrics,
                "training_population": final_train_stats,
                "calibration_brier": float(brier_score_loss(final_cal_labels, final_calibrator.predict(final_cal_raw))) if len(np.unique(final_cal_labels)) > 1 else None,
            },
            "generalization": {
                "oof_f05": oof_metrics["macro_f05"],
                "holdout_f05": holdout_metrics["macro_f05"],
                "absolute_gap": gap,
                "absolute_gap_percentage_points": gap * 100.0,
                "holdout_within_0_5pp": abs(gap) <= 0.005,
                "target_f05_reached_on_oof": oof_metrics["macro_f05"] >= 0.99,
                "target_f05_reached_on_holdout": holdout_metrics["macro_f05"] >= 0.99,
            },
            "score_quality": quality,
            "error_taxonomy": errors,
            "m8_cluster_evaluation": oof_cluster,
            "artifacts": {
                "oof_predictions": str(oof_path),
                "holdout_predictions": str(holdout_path),
                "oof_decisions": str(decision_path),
                "frozen_model": str(model_path),
                "frozen_policy": str(policy_path),
            },
            "anti_overfit_status": {
                "status": "PASS" if abs(gap) <= 0.005 and holdout_metrics["macro_f05"] >= 0.99 else "INVESTIGATE",
                "reason": "Holdout is untouched during OOF threshold selection; a material OOF/holdout gap indicates possible overfitting.",
            },
        }

        report_path = diagnostic_dir / "final_evaluation.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        text_path = diagnostic_dir / "final_evaluation.txt"
        text_path.write_text(self._render_text_report(report), encoding="utf-8")
        return report

    @staticmethod
    def _render_text_report(report: dict) -> str:
        oof = report["oof"]["selected_metrics"]
        hold = report["holdout"]["metrics"]
        gen = report["generalization"]
        lines = [
            "MILESTONE 9 — FINAL GENERALIZATION EVALUATION",
            "=" * 62,
            "",
            "OOF DEVELOPMENT",
            f"Macro F0.5       : {oof['macro_f05'] * 100:.4f}%",
            f"Macro precision  : {oof['macro_precision'] * 100:.4f}%",
            f"Macro recall     : {oof['macro_recall'] * 100:.4f}%",
            f"Selected threshold: {oof['threshold']:.6f}",
            f"Fold mean F0.5   : {report['oof']['fold_mean_f05'] * 100:.4f}%",
            f"Fold std F0.5    : {report['oof']['fold_std_f05'] * 100:.4f} pp",
            f"Fold min F0.5    : {report['oof']['fold_min_f05'] * 100:.4f}%",
            "",
            "UNTOUCHED HOLDOUT",
            f"Macro F0.5       : {hold['macro_f05'] * 100:.4f}%",
            f"Macro precision  : {hold['macro_precision'] * 100:.4f}%",
            f"Macro recall     : {hold['macro_recall'] * 100:.4f}%",
            f"Generalization gap: {gen['absolute_gap_percentage_points']:.4f} pp",
            "",
            "ANTI-OVERFIT STATUS",
            f"Status            : {report['anti_overfit_status']['status']}",
            f"OOF >= 99% F0.5   : {gen['target_f05_reached_on_oof']}",
            f"Holdout >= 99%    : {gen['target_f05_reached_on_holdout']}",
            "",
            "M8 CLUSTER CLOSURE",
            f"Direct precision  : {report['m8_cluster_evaluation']['ground_truth_evaluation']['direct_edge_metrics']['precision'] * 100:.4f}%",
            f"Cluster precision : {report['m8_cluster_evaluation']['ground_truth_evaluation']['cluster_closure_metrics']['precision'] * 100:.4f}%",
            f"Cluster recall    : {report['m8_cluster_evaluation']['ground_truth_evaluation']['cluster_closure_metrics']['recall'] * 100:.4f}%",
            f"Transitive FP     : {report['m8_cluster_evaluation']['ground_truth_evaluation']['transitive_inference']['incorrect_inferred_pairs']}",
            "",
            "TEST DATA USED: NO",
            "TEST LABELS USED: NO",
            "",
        ]
        return "\n".join(lines)
