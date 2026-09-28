from __future__ import annotations

import csv
import numpy as np
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LOGGER = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover
    LogisticRegression = None
    IMPORT_ERROR = exc


@dataclass(frozen=True)
class MatchModelConfig:
    validation_fraction: float = 0.20
    hard_negative_ratio: int = 4
    max_positive_rows: int = 100_000
    max_negative_rows: int = 400_000
    max_validation_rows: int = 200_000
    random_state: int = 42
    positive_weight: float = 1.0
    progress_every: int = 100_000


FEATURE_COLUMNS = (
    "country_match", "country_conflict", "country_missing_either",
    "name_exact", "address_exact", "name_address_exact", "house_exact",
    "postal_exact", "domain_exact", "alias_exact", "name_nonempty_both",
    "address_nonempty_both", "domain_available_both", "alias_available_both",
    "name_address_both_available", "blocking_hit_count", "blocking_strategy_count",
    "name_jaro_winkler", "name_edit_similarity", "name_token_jaccard",
    "name_token_containment", "name_sorted_token_jaccard", "name_char3_jaccard",
    "name_char4_jaccard", "name_length_ratio", "address_edit_similarity",
    "address_token_jaccard", "address_token_containment", "address_char3_jaccard",
    "address_char4_jaccard", "address_length_ratio", "house_overlap",
    "postal_overlap", "domain_jaccard", "alias_jaccard", "combined_evidence_count",
    "strong_evidence_count", "missing_field_count",
)


def _stable_bucket(value: str) -> float:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _hardness(row: dict[str, str]) -> float:
    """Similarity-based difficulty score for negative sampling only.

    This is deliberately not the final match score. It identifies negatives that
    look deceptively similar and should therefore be over-represented in training.
    """
    vals = [
        float(row.get("name_jaro_winkler", 0)),
        float(row.get("name_token_jaccard", 0)),
        float(row.get("name_char4_jaccard", 0)),
        float(row.get("address_token_jaccard", 0)),
        float(row.get("address_char4_jaccard", 0)),
        float(row.get("domain_jaccard", 0)),
        float(row.get("alias_jaccard", 0)),
    ]
    evidence = float(row.get("combined_evidence_count", 0)) / 9.0
    return 0.65 * max(vals) + 0.25 * sum(vals) / len(vals) + 0.10 * evidence


def _reservoir_add(items: list[dict], row: dict, limit: int, seen: int, rng) -> None:
    if limit <= 0:
        return
    if len(items) < limit:
        items.append(row)
        return
    j = rng.randrange(seen)
    if j < limit:
        items[j] = row


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    try:
        csv.field_size_limit(2147483647)
    except Exception:
        pass
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def _split_row(row: dict[str, str], validation_fraction: float) -> str:
    return "validation" if _stable_bucket(row["source1_id"]) < validation_fraction else "train"


def _metrics(y_true, scores, threshold: float = 0.5) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    result = {
        "threshold": threshold,
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "confusion_matrix": cm,
        "positive_rows": int(sum(y_true)),
        "rows": int(len(y_true)),
    }
    if len(set(y_true)) > 1:
        result["average_precision"] = float(average_precision_score(y_true, scores))
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        result["average_precision"] = None
        result["roc_auc"] = None
    return result


class MatchModel:
    """Leakage-safe pairwise matcher using logistic regression as M6 baseline."""

    def __init__(self, config: MatchModelConfig | None = None):
        self.config = config or MatchModelConfig()
        self.pipeline = None
        self.feature_columns = list(FEATURE_COLUMNS)
        self.training_summary: dict = {}

    def _require_sklearn(self) -> None:
        if LogisticRegression is None:
            raise RuntimeError(
                "Milestone 6 requires scikit-learn. Install requirements.txt before running the matcher."
            ) from IMPORT_ERROR

    def _load_split(self, path: Path):
        import random
        self._require_sklearn()
        rng = random.Random(self.config.random_state)
        train_pos: list[dict] = []
        train_neg: list[dict] = []
        val_rows: list[dict] = []
        pos_seen = neg_seen = val_seen = 0
        for row in _read_rows(path):
            if not row.get("label"):
                continue
            label = int(row["label"])
            split = _split_row(row, self.config.validation_fraction)
            if split == "validation":
                val_seen += 1
                _reservoir_add(val_rows, row, self.config.max_validation_rows, val_seen, rng)
                continue
            if label:
                pos_seen += 1
                _reservoir_add(train_pos, row, self.config.max_positive_rows, pos_seen, rng)
            else:
                neg_seen += 1
                _reservoir_add(train_neg, row, self.config.max_negative_rows, neg_seen, rng)
        # Keep all selected positives and select hard negatives first. Random tie
        # sampling prevents file order from becoming a hidden source of bias.
        train_neg.sort(key=lambda r: _hardness(r), reverse=True)
        target_neg = min(len(train_neg), max(1, len(train_pos) * self.config.hard_negative_ratio))
        if target_neg < len(train_neg):
            hard = train_neg[:target_neg]
            # retain a small random tail so the model sees some ordinary negatives
            tail_n = min(max(0, target_neg // 10), len(train_neg) - target_neg)
            if tail_n:
                hard.extend(rng.sample(train_neg[target_neg:], tail_n))
            train_neg = hard
        return train_pos, train_neg, val_rows, {"positive_seen": pos_seen, "negative_seen": neg_seen, "validation_seen": val_seen}

    def _matrix(self, rows: list[dict]):
        import numpy as np
        return np.asarray([[float(row[c]) for c in self.feature_columns] for row in rows], dtype=float)

    def fit(self, feature_path: Path) -> dict:
        train_pos, train_neg, val_rows, seen = self._load_split(feature_path)
        train_rows = train_pos + train_neg
        if not train_pos or not train_neg:
            raise ValueError(
                f"Insufficient training classes: positives={len(train_pos)}, negatives={len(train_neg)}"
            )
        y_train = [1] * len(train_pos) + [0] * len(train_neg)
        X_train = self._matrix(train_rows)
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                random_state=self.config.random_state,
                solver="liblinear",
            )),
        ])
        self.pipeline.fit(X_train, y_train)

        summary = {
            "train_rows": len(train_rows),
            "train_positive": len(train_pos),
            "train_negative": len(train_neg),
            "validation_rows": len(val_rows),
            "source1_split": {
                "method": "stable_blake2b_hash",
                "validation_fraction": self.config.validation_fraction,
            },
            "hard_negative_ratio": self.config.hard_negative_ratio,
            **seen,
        }
        if val_rows:
            X_val = self._matrix(val_rows)
            y_val = [int(r["label"]) for r in val_rows]
            scores = self.pipeline.predict_proba(X_val)[:, 1]
            summary["validation"] = _metrics(y_val, scores)
            summary["validation_score_quantiles"] = {
                "p50": float(np.quantile(scores, 0.50)),
                "p90": float(np.quantile(scores, 0.90)),
                "p95": float(np.quantile(scores, 0.95)),
                "p99": float(np.quantile(scores, 0.99)),
            }
        self.training_summary = summary
        return summary

    def predict_file(self, feature_path: Path, output_path: Path) -> dict:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        import numpy as np
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        positives = 0
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["source1_id", "target_source", "target_id", "score", "label"],
                delimiter="\t",
            )
            writer.writeheader()
            batch = []
            for row in _read_rows(feature_path):
                batch.append(row)
                if len(batch) >= 5000:
                    rows, positives = self._predict_batch(batch, writer, rows, positives)
                    batch.clear()
            if batch:
                rows, positives = self._predict_batch(batch, writer, rows, positives)
        return {"rows": rows, "positive_labels": positives, "output": str(output_path)}

    def _predict_batch(self, rows, writer, total, positives):
        scores = self.pipeline.predict_proba(self._matrix(rows))[:, 1]
        for row, score in zip(rows, scores):
            writer.writerow({
                "source1_id": row["source1_id"],
                "target_source": row["target_source"],
                "target_id": row["target_id"],
                "score": f"{float(score):.10f}",
                "label": row.get("label", ""),
            })
            total += 1
            positives += int(row.get("label", "0") or 0)
        return total, positives

    def evaluate_thresholds(self, feature_path: Path, thresholds: Iterable[float] | None = None) -> list[dict]:
        """Evaluate precision/recall on the deterministic S1-held-out validation population."""
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        thresholds = list(thresholds or (0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99))
        import random
        rng = random.Random(self.config.random_state)
        rows = []
        seen = 0
        for r in _read_rows(feature_path):
            if not r.get("label") or _split_row(r, self.config.validation_fraction) != "validation":
                continue
            seen += 1
            _reservoir_add(rows, r, self.config.max_validation_rows, seen, rng)
        if not rows:
            return []
        scores = self.pipeline.predict_proba(self._matrix(rows))[:, 1]
        y = [int(r["label"]) for r in rows]
        results = []
        for threshold in thresholds:
            result = _metrics(y, scores, float(threshold))
            result["false_positives"] = int(result["confusion_matrix"][0][1])
            result["false_negatives"] = int(result["confusion_matrix"][1][0])
            results.append(result)
        return results

    def save(self, path: Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted model")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({
                "pipeline": self.pipeline,
                "feature_columns": self.feature_columns,
                "config": self.config,
                "training_summary": self.training_summary,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "MatchModel":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        model = cls(payload["config"])
        model.pipeline = payload["pipeline"]
        model.feature_columns = payload["feature_columns"]
        model.training_summary = payload.get("training_summary", {})
        return model

    def coefficient_report(self) -> list[dict]:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted")
        coefficients = self.pipeline.named_steps["model"].coef_[0]
        return [
            {"feature": feature, "coefficient": float(value), "odds_multiplier": float(__import__("math").exp(value))}
            for feature, value in sorted(zip(self.feature_columns, coefficients), key=lambda x: abs(x[1]), reverse=True)
        ]
