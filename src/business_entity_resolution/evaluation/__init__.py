"""
Industrial Entity Resolution: Evaluation & Data Control Subsystem.
"""

from .contracts import ExperimentContract, StageResultContract
from .population import EvaluationPopulation, PopulationMismatchError
from .splits import DatasetSplits, PartitionConfig
from .metrics import EvaluationMetrics, S1Score, calculate_s1_f05, compute_cohort_metrics
from .leakage import LeakageAuditReport, LeakageViolationError, audit_evaluation_leakage
from .final_evaluator import FinalEvaluationConfig, FinalEvaluator

__all__ = [
    "ExperimentContract",
    "StageResultContract",
    "EvaluationPopulation",
    "PopulationMismatchError",
    "DatasetSplits",
    "PartitionConfig",
    "EvaluationMetrics",
    "S1Score",
    "calculate_s1_f05",
    "compute_cohort_metrics",
    "LeakageAuditReport",
    "LeakageViolationError",
    "audit_evaluation_leakage",
    "FinalEvaluationConfig",
    "FinalEvaluator",
]
