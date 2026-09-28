from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from business_entity_resolution.evaluation.final_evaluator import (
    FinalEvaluationConfig,
    FinalEvaluator,
)
from business_entity_resolution.features.evidence_augmentation import (
    EXTRA_FEATURES,
)
from business_entity_resolution.matching.model import FEATURE_COLUMNS


class AugmentedFinalEvaluator(FinalEvaluator):
    """
    M9 evaluator with experiment-only additional features.

    Production M9 remains unchanged.

    The only behavioral change is the feature matrix used by the model.
    """

    def __init__(
        self,
        config: FinalEvaluationConfig | None = None,
        extra_columns: tuple[str, ...] = EXTRA_FEATURES,
    ):
        super().__init__(config)

        self.extra_columns = tuple(
            extra_columns
        )

        self.feature_columns = (
            tuple(FEATURE_COLUMNS)
            + self.extra_columns
        )

    def _matrix(
        self,
        rows: list[dict[str, str]],
    ):
        return np.asarray(
            [
                [
                    float(
                        row.get(column, 0.0)
                        or 0.0
                    )
                    for column in self.feature_columns
                ]
                for row in rows
            ],
            dtype=float,
        )

    def _save_frozen_artifacts(
        self,
        pipeline,
        calibrator,
        threshold: float,
        output_dir: Path,
    ):
        model_path, policy_path = (
            super()._save_frozen_artifacts(
                pipeline,
                calibrator,
                threshold,
                output_dir,
            )
        )

        with model_path.open(
            "rb"
        ) as handle:
            payload = pickle.load(handle)

        if isinstance(payload, dict):
            payload["feature_columns"] = list(
                self.feature_columns
            )
            if "training_summary" in payload and isinstance(payload["training_summary"], dict):
                payload["training_summary"][
                    "extra_feature_columns"
                ] = list(
                    self.extra_columns
                )
        else:
            payload.feature_columns = list(
                self.feature_columns
            )
            if hasattr(payload, "training_summary") and isinstance(payload.training_summary, dict):
                payload.training_summary[
                    "extra_feature_columns"
                ] = list(
                    self.extra_columns
                )

        with model_path.open(
            "wb"
        ) as handle:
            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return model_path, policy_path
