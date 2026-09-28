"""Matching model and hard-negative mining module for Milestone 6."""

from business_entity_resolution.models.matching_model import (
    MatchModelConfig,
    MatchModel,
    ModelTrainer,
    ModelPredictor,
    is_validation_entity,
    compute_negative_difficulty,
)

__all__ = [
    "MatchModelConfig",
    "MatchModel",
    "ModelTrainer",
    "ModelPredictor",
    "is_validation_entity",
    "compute_negative_difficulty",
]
