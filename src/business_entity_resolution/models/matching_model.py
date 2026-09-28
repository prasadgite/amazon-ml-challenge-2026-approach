from __future__ import annotations

import csv
import hashlib
import heapq
import json
import logging
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from business_entity_resolution.features.pairwise import FeatureSchema

LOGGER = logging.getLogger("ber.match_model")

# Identifier columns that should never be fed directly into the model as numeric features
IDENTIFIER_COLUMNS = ("source1_id", "target_source", "target_id", "label")


def is_validation_entity(source1_id: str, validation_fraction: float = 0.20) -> bool:
    """
    Deterministic S1 entity-level partition using BLAKE2b hash.
    Ensures all candidate pairs for a given source1_id stay strictly together
    in either train or validation, preventing entity leakage.
    """
    if validation_fraction <= 0.0:
        return False
    if validation_fraction >= 1.0:
        return True
    digest = hashlib.blake2b(source1_id.encode("utf-8"), digest_size=8).digest()
    norm_val = int.from_bytes(digest, "big") / (2**64 - 1)
    return norm_val < validation_fraction


def compute_negative_difficulty(row: Dict[str, Any]) -> float:
    """
    Computes a composite difficulty score for non-matching candidate pairs.
    Candidates with high name similarity, address similarity, combined evidence,
    and blocking hits represent difficult 'confusing' negatives.
    """
    try:
        jw = float(row.get("name_jaro_winkler", 0.0) or 0.0)
        addr_sim = float(row.get("address_edit_similarity", 0.0) or 0.0)
        name_tok = float(row.get("name_token_jaccard", 0.0) or 0.0)
        addr_tok = float(row.get("address_token_jaccard", 0.0) or 0.0)
        comb_ev = float(row.get("combined_evidence_count", 0.0) or 0.0)
        hits = float(row.get("blocking_hit_count", 0.0) or 0.0)
        str_ev = float(row.get("strong_evidence_count", 0.0) or 0.0)
    except (ValueError, TypeError):
        return 0.0

    return (
        (jw * 3.0)
        + (addr_sim * 2.5)
        + (name_tok * 2.0)
        + (addr_tok * 1.5)
        + (comb_ev * 0.5)
        + (str_ev * 1.0)
        + (hits * 0.2)
    )


def extract_features(row: Dict[str, Any], feature_names: List[str]) -> List[float]:
    """Extracts numeric feature vector from row dictionary using predefined feature ordering."""
    vec = []
    for col in feature_names:
        raw_val = row.get(col, 0.0)
        try:
            vec.append(float(raw_val) if raw_val not in ("", None) else 0.0)
        except (ValueError, TypeError):
            vec.append(0.0)
    return vec


@dataclass
class MatchModelConfig:
    validation_fraction: float = 0.20
    hard_negative_ratio: int = 4
    max_positive_rows: int = 100_000
    max_negative_rows: int = 400_000
    max_validation_rows: int = 200_000
    model_type: str = "logistic"  # 'logistic' or 'lightgbm'
    random_state: int = 42
    threshold_list: Tuple[float, ...] = (
        0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MatchModel:
    """Wrapper encapsulating preprocessor, classifier, serialization, and coefficient inspection."""

    def __init__(
        self,
        feature_names: List[str],
        model_type: str = "logistic",
        random_state: int = 42,
    ):
        self.feature_names = list(feature_names)
        self.model_type = model_type.lower()
        self.random_state = random_state

        if self.model_type == "logistic":
            self.scaler: Optional[StandardScaler] = StandardScaler()
            self.clf = LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=self.random_state,
            )
        elif self.model_type == "lightgbm":
            import lightgbm as lgb
            self.scaler = None
            self.clf = lgb.LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Expected 'logistic' or 'lightgbm'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MatchModel":
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled, y)
        else:
            self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X_trans = self.scaler.transform(X)
        else:
            X_trans = X
        # Return probability of positive class (label = 1)
        proba = self.clf.predict_proba(X_trans)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, 1]

    def get_coefficients(self) -> List[Dict[str, Any]]:
        """Returns feature weights, standard deviations, and odds ratios."""
        results = []
        if self.model_type == "logistic" and hasattr(self.clf, "coef_"):
            coefs = self.clf.coef_[0]
            stds = self.scaler.scale_ if self.scaler is not None else np.ones_like(coefs)
            for name, coef, std in zip(self.feature_names, coefs, stds):
                odds_mult = math.exp(float(coef)) if abs(coef) < 700 else float("inf")
                results.append({
                    "feature": name,
                    "coefficient": round(float(coef), 6),
                    "odds_multiplier": round(odds_mult, 6) if math.isfinite(odds_mult) else None,
                    "scale": round(float(std), 6),
                })
            results.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        elif hasattr(self.clf, "feature_importances_"):
            importances = self.clf.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                results.append({
                    "feature": name,
                    "importance": int(imp),
                })
            results.sort(key=lambda x: x["importance"], reverse=True)
        return results

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: Path) -> "MatchModel":
        with open(model_path, "rb") as f:
            return pickle.load(f)


class ModelTrainer:
    """Orchestrates streaming training, hard negative mining, validation, and threshold diagnostics."""

    def __init__(self, config: Optional[MatchModelConfig] = None):
        self.config = config or MatchModelConfig()

    def train_and_evaluate(
        self,
        features_tsv_path: Path,
        model_out_path: Path,
        diagnostics_dir: Path,
    ) -> Dict[str, Any]:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        model_out_path.parent.mkdir(parents=True, exist_ok=True)

        if not features_tsv_path.exists():
            raise FileNotFoundError(f"Feature matrix file not found: {features_tsv_path}")

        LOGGER.info(f"Reading pairwise features from: {features_tsv_path}")

        # 1. Inspect header and determine feature column list
        with open(features_tsv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            fieldnames = reader.fieldnames or []
            if "label" not in fieldnames:
                raise ValueError(f"Features file {features_tsv_path} does not contain 'label' column required for training.")

            feature_names = [col for col in fieldnames if col not in IDENTIFIER_COLUMNS]
            LOGGER.info(f"Using {len(feature_names)} features for matching model training.")

            # 2. Stream dataset into Train and Validation pools
            train_positives: List[List[float]] = []
            val_rows: List[Tuple[List[float], int]] = []

            # Stream hard-negative candidates using a min-heap to keep highest difficulty negatives
            target_negatives = max(100, self.config.max_positive_rows * self.config.hard_negative_ratio)
            target_negatives = min(target_negatives, self.config.max_negative_rows)
            neg_heap: List[Tuple[float, int, List[float]]] = []  # (difficulty, counter, feat_vec)
            neg_counter = 0

            total_read = 0
            for row in reader:
                total_read += 1
                s1_id = row.get("source1_id", "")
                lbl_str = row.get("label", "0")
                label = 1 if lbl_str in ("1", 1, True, "True") else 0
                feats = extract_features(row, feature_names)

                if is_validation_entity(s1_id, self.config.validation_fraction):
                    if len(val_rows) < self.config.max_validation_rows:
                        val_rows.append((feats, label))
                else:
                    if label == 1:
                        if len(train_positives) < self.config.max_positive_rows:
                            train_positives.append(feats)
                    else:
                        # Hard negative candidate
                        diff = compute_negative_difficulty(row)
                        neg_counter += 1
                        if len(neg_heap) < target_negatives:
                            heapq.heappush(neg_heap, (diff, neg_counter, feats))
                        elif diff > neg_heap[0][0]:
                            heapq.heappushpop(neg_heap, (diff, neg_counter, feats))

                if total_read % 100_000 == 0:
                    LOGGER.info(f"Parsed {total_read:,} candidate pairs...")

        num_pos = len(train_positives)
        # Select target negatives matching hard_negative_ratio
        desired_negs = min(len(neg_heap), num_pos * self.config.hard_negative_ratio)
        if len(neg_heap) > desired_negs and desired_negs > 0:
            # Get the top desired_negs highest difficulty negatives
            sorted_negs = heapq.nlargest(desired_negs, neg_heap)
            train_negatives = [item[2] for item in sorted_negs]
        else:
            train_negatives = [item[2] for item in neg_heap]

        num_neg = len(train_negatives)
        LOGGER.info(
            f"Training set: {num_pos:,} positives, {num_neg:,} hard negatives "
            f"(ratio {num_neg / max(1, num_pos):.2f})."
        )
        LOGGER.info(f"Validation set: {len(val_rows):,} candidate pairs.")

        if num_pos == 0:
            raise ValueError("No positive examples (label=1) found in training set. Cannot train model.")
        if num_neg == 0:
            raise ValueError("No negative examples found in training set. Cannot train model.")
        if len(val_rows) == 0:
            raise ValueError("Validation set is empty. Check validation_fraction.")

        # 3. Assemble training arrays
        X_train = np.array(train_positives + train_negatives, dtype=np.float32)
        y_train = np.array([1] * num_pos + [0] * num_neg, dtype=np.int32)

        # 4. Train model
        model = MatchModel(
            feature_names=feature_names,
            model_type=self.config.model_type,
            random_state=self.config.random_state,
        )
        LOGGER.info(f"Fitting {self.config.model_type} matching model...")
        model.fit(X_train, y_train)

        # 5. Evaluate on Validation set
        X_val = np.array([r[0] for r in val_rows], dtype=np.float32)
        y_val = np.array([r[1] for r in val_rows], dtype=np.int32)
        y_scores = model.predict_proba(X_val)

        val_pos_count = int(np.sum(y_val == 1))
        val_neg_count = int(np.sum(y_val == 0))
        val_pr_auc = float(average_precision_score(y_val, y_scores)) if val_pos_count > 0 else 0.0
        val_roc_auc = float(roc_auc_score(y_val, y_scores)) if val_pos_count > 0 and val_neg_count > 0 else 0.5

        # Metrics at default threshold 0.50
        y_pred_default = (y_scores >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_default, labels=[0, 1]).ravel()
        default_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        default_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        default_f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        # Score quantiles
        def quantiles(vals: np.ndarray) -> Dict[str, float]:
            if len(vals) == 0:
                return {}
            qs = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
            labels = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
            res = np.quantile(vals, qs)
            return {lbl: round(float(v), 5) for lbl, v in zip(labels, res)}

        pos_scores = y_scores[y_val == 1]
        neg_scores = y_scores[y_val == 0]

        summary_metrics = {
            "model_type": self.config.model_type,
            "training_samples": {
                "positive_pairs": num_pos,
                "hard_negative_pairs": num_neg,
                "total_train_pairs": num_pos + num_neg,
                "hard_negative_ratio": round(num_neg / max(1, num_pos), 2),
            },
            "validation_samples": {
                "positive_pairs": val_pos_count,
                "negative_pairs": val_neg_count,
                "total_val_pairs": len(val_rows),
            },
            "validation_performance": {
                "pr_auc": round(val_pr_auc, 6),
                "roc_auc": round(val_roc_auc, 6),
                "at_threshold_0_50": {
                    "precision": round(default_prec, 6),
                    "recall": round(default_rec, 6),
                    "f1": round(default_f1, 6),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                },
            },
            "score_quantiles": {
                "positives": quantiles(pos_scores),
                "negatives": quantiles(neg_scores),
            },
        }

        # 6. Threshold Sweep Diagnostics
        threshold_results = []
        for tau in self.config.threshold_list:
            t_tp = int(np.sum((y_scores >= tau) & (y_val == 1)))
            t_fp = int(np.sum((y_scores >= tau) & (y_val == 0)))
            t_fn = int(np.sum((y_scores < tau) & (y_val == 1)))
            t_tn = int(np.sum((y_scores < tau) & (y_val == 0)))

            prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 1.0
            rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
            # Competition official beta is 0.5: F_0.5 = 1.25 * P * R / (0.25 * P + R)
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            threshold_results.append({
                "threshold": round(float(tau), 4),
                "precision": round(float(prec), 6),
                "recall": round(float(rec), 6),
                "f0_5": round(float(f05), 6),
                "f1": round(float(f1), 6),
                "true_positives": t_tp,
                "false_positives": t_fp,
                "false_negatives": t_fn,
                "true_negatives": t_tn,
            })

        # 7. Feature coefficients / importances
        coefficients = model.get_coefficients()

        # 8. Save artifacts
        model.save(model_out_path)
        LOGGER.info(f"Model saved to: {model_out_path}")

        json_out = diagnostics_dir / "match_model.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        coef_out = diagnostics_dir / "match_model_coefficients.json"
        with open(coef_out, "w", encoding="utf-8") as f:
            json.dump(coefficients, f, indent=2)

        thresh_out = diagnostics_dir / "match_model_thresholds.json"
        with open(thresh_out, "w", encoding="utf-8") as f:
            json.dump(threshold_results, f, indent=2)

        LOGGER.info(
            f"Validation PR-AUC={val_pr_auc:.4f} | ROC-AUC={val_roc_auc:.4f} | "
            f"Prec@0.5={default_prec:.4f} | Rec@0.5={default_rec:.4f}"
        )
        return summary_metrics


class ModelPredictor:
    """Streams test feature matrix in batches and computes predicted match probabilities."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = MatchModel.load(self.model_path)

    def predict(
        self,
        features_tsv_path: Path,
        output_scores_tsv: Path,
        batch_size: int = 50_000,
    ) -> int:
        if not features_tsv_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_tsv_path}")

        output_scores_tsv.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Scoring test candidate pairs from {features_tsv_path} -> {output_scores_tsv}")

        feature_names = self.model.feature_names
        total_scored = 0

        with open(features_tsv_path, "r", encoding="utf-8", newline="") as fin, \
             open(output_scores_tsv, "w", encoding="utf-8", newline="") as fout:

            reader = csv.DictReader(fin, delimiter="\t")
            writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_id", "target_source", "target_id", "score", "label"])

            batch_meta: List[Tuple[str, str, str, str]] = []
            batch_feats: List[List[float]] = []

            for row in reader:
                s1 = row.get("source1_id", "")
                t_src = row.get("target_source", "")
                t_id = row.get("target_id", "")
                lbl = row.get("label", "")
                batch_meta.append((s1, t_src, t_id, lbl))
                batch_feats.append(extract_features(row, feature_names))

                if len(batch_feats) >= batch_size:
                    X_arr = np.array(batch_feats, dtype=np.float32)
                    scores = self.model.predict_proba(X_arr)
                    for (s1_id, ts, ti, l), sc in zip(batch_meta, scores):
                        writer.writerow([s1_id, ts, ti, f"{sc:.6f}", l])
                    total_scored += len(batch_feats)
                    LOGGER.info(f"Scored {total_scored:,} candidate pairs...")
                    batch_meta.clear()
                    batch_feats.clear()

            if batch_feats:
                X_arr = np.array(batch_feats, dtype=np.float32)
                scores = self.model.predict_proba(X_arr)
                for (s1_id, ts, ti, l), sc in zip(batch_meta, scores):
                    writer.writerow([s1_id, ts, ti, f"{sc:.6f}", l])
                total_scored += len(batch_feats)

        LOGGER.info(f"Scoring complete. Total candidate pairs scored: {total_scored:,}")
        return total_scored
