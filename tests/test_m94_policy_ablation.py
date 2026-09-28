from __future__ import annotations

from pathlib import Path
import pytest

from scripts.run_m94_policy_ablation import compute_s1_metrics


def test_compute_s1_metrics_perfect_precision_and_recall():
    all_s1 = {"S1-1", "S1-2", "S1-3"}
    ground_truth = {("S1-1", "S2-1"), ("S1-2", "S3-1")}
    decisions = {("S1-1", "S2-1"), ("S1-2", "S3-1")}

    metrics = compute_s1_metrics(decisions, ground_truth, all_s1)
    assert metrics["macro_f05"] == pytest.approx(1.0, 1e-4)
    assert metrics["macro_precision"] == pytest.approx(1.0, 1e-4)
    assert metrics["macro_recall"] == pytest.approx(1.0, 1e-4)
    assert metrics["singleton_fps"] == 0
    assert metrics["total_singletons"] == 1  # S1-3 is a true singleton


def test_compute_s1_metrics_penalizes_singleton_fp():
    all_s1 = {"S1-1", "S1-2", "S1-3"}
    ground_truth = {("S1-1", "S2-1")}
    # S1-3 is singleton, but erroneously matched to S2-99
    decisions = {("S1-1", "S2-1"), ("S1-3", "S2-99")}

    metrics = compute_s1_metrics(decisions, ground_truth, all_s1)
    assert metrics["singleton_fps"] == 1
    assert metrics["macro_precision"] < 1.0
    assert metrics["macro_f05"] < 1.0
