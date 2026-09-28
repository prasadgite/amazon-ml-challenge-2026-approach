from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        precision_score,
        recall_score,
    )
except ImportError as exc:  # pragma: no cover
    np = None
    IsotonicRegression = None
    LogisticRegression = None
    IMPORT_ERROR = exc


@dataclass(frozen=True)
class DecisionConfig:
    calibration_fraction: float = 0.50
    calibration_method: str = "platt"  # platt | isotonic | none
    target_precision: float = 0.99
    min_recall_at_target: float = 0.0
    threshold_grid_size: int = 1001
    random_state: int = 42
    veto_country_conflict: bool = True
    veto_missing_identity_evidence: bool = False
    min_positive_predictions: int = 1


def _bucket(value: str, seed: int = 42) -> float:
    raw = f"{seed}:{value}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _read_rows(path: Path) -> Iterable[dict[str, str]]:
    try:
        csv.field_size_limit(2147483647)
    except Exception:
        pass
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1.0 - p))


def _safe_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


class ProbabilityCalibrator:
    """Calibrates M6 scores without changing the underlying matcher."""

    def __init__(self, method: str = "platt"):
        if method not in {"platt", "isotonic", "none"}:
            raise ValueError("calibration method must be platt, isotonic, or none")
        self.method = method
        self.model = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        if self.method == "none":
            return
        if len(np.unique(labels)) < 2:
            raise ValueError("Calibration requires both positive and negative labels")
        if self.method == "platt":
            self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.model.fit(_logit(scores).reshape(-1, 1), labels)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(scores, labels)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return np.asarray(scores, dtype=float)
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted")
        if self.method == "platt":
            return self.model.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return np.asarray(self.model.predict(scores), dtype=float)


class DecisionEngine:
    """M7: calibration, threshold selection, diagnostics, and conservative policy."""

    def __init__(self, model, config: DecisionConfig | None = None):
        self.model = model
        self.config = config or DecisionConfig()
        self.calibrator = ProbabilityCalibrator(self.config.calibration_method)
        self.threshold: float | None = None
        self.threshold_selection: dict = {}

    def _require(self):
        if np is None or LogisticRegression is None:
            raise RuntimeError("Milestone 7 requires scikit-learn") from IMPORT_ERROR
        if self.model.pipeline is None:
            raise RuntimeError("M6 model must be fitted or loaded")

    def _validation_rows(self, feature_path: Path) -> list[dict[str, str]]:
        rows = []
        for row in _read_rows(feature_path):
            if not row.get("label"):
                continue
            # Must match M6's validation population exactly.
            from business_entity_resolution.matching.model import _split_row
            if _split_row(row, self.model.config.validation_fraction) == "validation":
                rows.append(row)
        return rows

    def _score_rows(self, rows: list[dict[str, str]]) -> np.ndarray:
        batch_size = 5000
        output = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            X = self.model._matrix(batch)
            output.extend(self.model.pipeline.predict_proba(X)[:, 1].tolist())
        return np.asarray(output, dtype=float)

    def _calibration_split(self, rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
        cal, evaluation = [], []
        for row in rows:
            if _bucket(row["source1_id"], self.config.random_state) < self.config.calibration_fraction:
                cal.append(row)
            else:
                evaluation.append(row)
        return cal, evaluation

    @staticmethod
    def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
        return {
            "threshold": float(threshold),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "predicted_positive": int(preds.sum()),
            "true_positive": int(((preds == 1) & (labels == 1)).sum()),
            "false_positive": int(((preds == 1) & (labels == 0)).sum()),
            "false_negative": int(((preds == 0) & (labels == 1)).sum()),
            "rows": int(len(labels)),
            "confusion_matrix": cm,
        }

    def _threshold_grid(self) -> np.ndarray:
        n = max(11, int(self.config.threshold_grid_size))
        return np.linspace(0.0, 1.0, n)

    def _select_threshold(self, labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict, list[dict]]:
        candidates = []
        for threshold in self._threshold_grid():
            metrics = self._metrics(labels, scores, float(threshold))
            if metrics["predicted_positive"] < self.config.min_positive_predictions:
                continue
            candidates.append(metrics)

        eligible = [
            x for x in candidates
            if x["precision"] >= self.config.target_precision
            and x["recall"] >= self.config.min_recall_at_target
        ]
        if eligible:
            # Maximize recall subject to the required precision; use the lowest
            # threshold among equal-recall choices to avoid unnecessary abstention.
            selected = max(eligible, key=lambda x: (x["recall"], -x["threshold"]))
            status = "target_precision_reached"
        else:
            # No operating point met the requested precision. Select the highest
            # precision point, then the highest recall among ties. This is a diagnostic
            # fallback, not a claim that the target was achieved.
            selected = max(candidates, key=lambda x: (x["precision"], x["recall"]))
            status = "target_precision_not_reached"

        selection = {
            "status": status,
            "target_precision": self.config.target_precision,
            "min_recall_at_target": self.config.min_recall_at_target,
            "selected": selected,
            "eligible_count": len(eligible),
            "grid_size": len(candidates),
        }
        return float(selected["threshold"]), selection, candidates

    def fit(self, feature_path: Path) -> dict:
        self._require()
        rows = self._validation_rows(feature_path)
        if len(rows) < 10:
            raise ValueError(f"Too few held-out validation rows for M7: {len(rows)}")
        calibration_rows, evaluation_rows = self._calibration_split(rows)
        if not calibration_rows or not evaluation_rows:
            raise ValueError("Calibration/evaluation split produced an empty partition")

        cal_scores_raw = self._score_rows(calibration_rows)
        cal_labels = np.asarray([int(r["label"]) for r in calibration_rows], dtype=int)
        eval_scores_raw = self._score_rows(evaluation_rows)
        eval_labels = np.asarray([int(r["label"]) for r in evaluation_rows], dtype=int)

        self.calibrator.fit(cal_scores_raw, cal_labels)
        cal_scores = self.calibrator.predict(cal_scores_raw)
        eval_scores = self.calibrator.predict(eval_scores_raw)
        self.threshold, selection, grid = self._select_threshold(eval_labels, eval_scores)
        self.threshold_selection = selection

        report = {
            "config": asdict(self.config),
            "source1_split": {
                "method": "stable_blake2b_hash",
                "m6_validation_fraction": self.model.config.validation_fraction,
                "m7_calibration_fraction": self.config.calibration_fraction,
                "rows": len(rows),
                "calibration_rows": len(calibration_rows),
                "evaluation_rows": len(evaluation_rows),
            },
            "calibration": {
                "method": self.config.calibration_method,
                "brier_raw": float(brier_score_loss(cal_labels, cal_scores_raw)),
                "brier_calibrated": float(brier_score_loss(cal_labels, cal_scores)),
            },
            "evaluation": {
                "average_precision": float(average_precision_score(eval_labels, eval_scores)),
                "brier": float(brier_score_loss(eval_labels, eval_scores)),
                "threshold_selection": selection,
                "selected_metrics": self._metrics(eval_labels, eval_scores, self.threshold),
                "score_quantiles": {
                    f"p{q}": float(np.quantile(eval_scores, q / 100.0))
                    for q in (1, 5, 25, 50, 75, 90, 95, 99, 99.9)
                },
            },
            "threshold_grid": grid,
        }
        return report

    def decision(self, row: dict[str, str], raw_score: float) -> tuple[str, float, str]:
        if self.threshold is None:
            raise RuntimeError("M7 threshold has not been fitted")
        calibrated = float(self.calibrator.predict(np.asarray([raw_score]))[0])
        if calibrated < self.threshold:
            return "NON_MATCH", calibrated, "below_threshold"

        if self.config.veto_country_conflict and _safe_float(row, "country_conflict") >= 1:
            return "NON_MATCH", calibrated, "country_conflict_veto"

        if self.config.veto_missing_identity_evidence and _safe_float(row, "strong_evidence_count") <= 0:
            return "NON_MATCH", calibrated, "missing_strong_evidence_veto"

        return "MATCH", calibrated, "threshold_pass"

    def score_file(self, feature_path: Path, output_path: Path) -> dict:
        if self.threshold is None:
            raise RuntimeError("Fit M7 before scoring")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = matches = vetoes = 0
        reasons: dict[str, int] = {}
        batch: list[dict[str, str]] = []

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "source1_id", "target_source", "target_id", "raw_score",
                "calibrated_score", "decision", "decision_reason", "label",
            ], delimiter="\t")
            writer.writeheader()

            def flush(items: list[dict[str, str]]):
                nonlocal rows, matches, vetoes
                if not items:
                    return
                raw_scores = self.model.pipeline.predict_proba(self.model._matrix(items))[:, 1]
                calibrated = self.calibrator.predict(raw_scores)
                for row, raw, score in zip(items, raw_scores, calibrated):
                    if float(score) < self.threshold:
                        decision, reason = "NON_MATCH", "below_threshold"
                    elif self.config.veto_country_conflict and _safe_float(row, "country_conflict") >= 1:
                        decision, reason = "NON_MATCH", "country_conflict_veto"
                    elif self.config.veto_missing_identity_evidence and _safe_float(row, "strong_evidence_count") <= 0:
                        decision, reason = "NON_MATCH", "missing_strong_evidence_veto"
                    else:
                        decision, reason = "MATCH", "threshold_pass"
                    writer.writerow({
                        "source1_id": row["source1_id"],
                        "target_source": row["target_source"],
                        "target_id": row["target_id"],
                        "raw_score": f"{float(raw):.10f}",
                        "calibrated_score": f"{float(score):.10f}",
                        "decision": decision,
                        "decision_reason": reason,
                        "label": row.get("label", ""),
                    })
                    rows += 1
                    matches += int(decision == "MATCH")
                    vetoes += int(reason.endswith("_veto"))
                    reasons[reason] = reasons.get(reason, 0) + 1

            for row in _read_rows(feature_path):
                batch.append(row)
                if len(batch) >= 5000:
                    flush(batch)
                    batch.clear()
            flush(batch)
        return {"rows": rows, "matches": matches, "vetoes": vetoes, "reasons": reasons, "output": str(output_path)}

    def false_positive_report(self, feature_path: Path, limit: int = 100) -> list[dict]:
        """Return highest-confidence validation false positives for error analysis."""
        if self.threshold is None:
            raise RuntimeError("Fit M7 before false-positive analysis")
        rows = []
        for row in self._validation_rows(feature_path):
            raw = float(self.model.pipeline.predict_proba(self.model._matrix([row]))[0, 1])
            score = float(self.calibrator.predict(np.asarray([raw]))[0])
            if int(row["label"]) == 0 and score >= self.threshold:
                rows.append({
                    "source1_id": row["source1_id"],
                    "target_source": row["target_source"],
                    "target_id": row["target_id"],
                    "raw_score": raw,
                    "calibrated_score": score,
                    "name_jaro_winkler": _safe_float(row, "name_jaro_winkler"),
                    "name_token_jaccard": _safe_float(row, "name_token_jaccard"),
                    "address_token_jaccard": _safe_float(row, "address_token_jaccard"),
                    "domain_exact": _safe_float(row, "domain_exact"),
                    "alias_exact": _safe_float(row, "alias_exact"),
                    "country_conflict": _safe_float(row, "country_conflict"),
                    "blocking_hit_count": _safe_float(row, "blocking_hit_count"),
                })
        rows.sort(key=lambda x: x["calibrated_score"], reverse=True)
        return rows[:limit]

    def save(self, path: Path) -> None:
        if self.threshold is None:
            raise RuntimeError("Cannot save an unfitted M7 decision engine")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({
                "config": self.config,
                "calibrator": self.calibrator,
                "threshold": self.threshold,
                "threshold_selection": self.threshold_selection,
            }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, model):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        engine = cls(model, payload["config"])
        engine.calibrator = payload["calibrator"]
        engine.threshold = payload["threshold"]
        engine.threshold_selection = payload.get("threshold_selection", {})
        return engine
