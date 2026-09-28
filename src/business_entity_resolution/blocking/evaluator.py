"""
Blocking Recall and Candidate Efficiency Evaluation Experiment.
Supports SQLite-backed multi-pass indexing, bucket capping, reduction ratio,
candidate-positive rate, strategy provenance, and candidate TSV export.
"""

from typing import Dict, List, Any, Optional
import csv
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.io.record import EntityRecord
from business_entity_resolution.io.tsv_loader import load_ground_truth
from business_entity_resolution.normalization.normalizer import TextNormalizer, iter_normalized_tsv
from business_entity_resolution.blocking.sqlite_index import SQLiteBlockingIndex
from business_entity_resolution.blocking.keys import ALL_BLOCKING_STRATEGIES

logger = logging.getLogger("BlockingEvaluator")


class BlockingEvaluator:
    """Evaluates candidate recall, reduction ratio, and efficiency against ground truth."""

    def __init__(self, paths: ProjectPaths, config: PipelineConfig, log: Optional[logging.Logger] = None):
        self.paths = paths
        self.config = config
        self.logger = log or logger
        self.normalizer = TextNormalizer()

    def build_index(
        self,
        split: str = "train",
        max_rows: int = 0,
        max_bucket_size: int = 1000,
        export_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds the SQLite blocking index for a split (train or test),
        computes bucket statistics, and optionally exports candidate pairs.
        """
        self.paths.ensure_directories()
        db_file = self.paths.indexes_dir / f"{split}_blocking.sqlite"
        self.logger.info(f"Building SQLite blocking index for {split} split at {db_file}...")
        start_time = time.time()

        if db_file.exists():
            db_file.unlink()

        index = SQLiteBlockingIndex(db_file)

        src2_path = self.paths.train_source2 if split == "train" else self.paths.test_source2
        src3_path = self.paths.train_source3 if split == "train" else self.paths.test_source3
        src1_path = self.paths.train_source1 if split == "train" else self.paths.test_source1

        # Index Source 2 and Source 3
        for s_path in [src2_path, src3_path]:
            self.logger.info(f"  Indexing target records from {s_path.name}...")
            postings_count = 0
            batch: List[Any] = []
            for rec in iter_normalized_tsv(s_path, max_rows=max_rows, normalizer=self.normalizer):
                batch.append(rec)
                if len(batch) >= 10_000:
                    postings_count += index.bulk_insert_targets(batch)
                    batch.clear()
            if batch:
                postings_count += index.bulk_insert_targets(batch)
            self.logger.info(f"    Indexed {s_path.name} ({postings_count:,} postings added).")

        # Bucket statistics
        bucket_stats = index.get_bucket_statistics(max_bucket_size=max_bucket_size)
        diag_path = self.paths.diagnostics_dir / f"{split}_blocking.json"
        index.save_diagnostics(diag_path, max_bucket_size=max_bucket_size)
        self.logger.info(f"Bucket statistics saved to {diag_path.name}: {bucket_stats}")

        # Optional candidate export
        if export_candidates:
            cand_tsv = self.paths.features_dir / f"{split}_candidates.tsv"
            self.logger.info(f"Exporting candidate pairs to {cand_tsv}...")
            cand_count = 0
            with open(cand_tsv, "w", encoding="utf-8") as out_f:
                out_f.write("source1_entity_id\tcandidate_entity_ids\n")
                for s1_rec in iter_normalized_tsv(src1_path, max_rows=max_rows, normalizer=self.normalizer):
                    cands = index.query_candidates(s1_rec, max_bucket_size=max_bucket_size)
                    sorted_target_ids = sorted(cands.keys())
                    cand_str = ",".join(sorted_target_ids)
                    out_f.write(f"{s1_rec.entity_id}\t{cand_str}\n")
                    cand_count += len(sorted_target_ids)
            self.logger.info(f"Candidate export complete: {cand_count:,} candidate pairs written.")

        elapsed = round(time.time() - start_time, 2)
        summary = {
            "split": split,
            "elapsed_seconds": elapsed,
            "db_path": str(db_file),
            "bucket_statistics": bucket_stats,
        }
        return summary

    def run_evaluation(
        self,
        sample_s1: int = 2_000,
        background_distractor_count: int = 25_000,
        max_bucket_size: int = 1000,
        export_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs empirical recall evaluation on a representative slice of train data.
        Builds SQLite index over true targets + distractors, evaluates recall,
        reduction ratio, candidate-positive rate, and provenance.
        """
        self.logger.info(f"Starting Blocking Evaluation (sample_s1={sample_s1:,}, background_targets={background_distractor_count:,}, max_bucket_size={max_bucket_size})...")
        start_time = time.time()
        self.paths.ensure_directories()

        # 1. Load ground truth mapping
        self.logger.info("Loading ground truth mapping...")
        s1_to_targets, target_to_s1 = load_ground_truth(self.paths.train_ground_truth, max_rows=sample_s1)
        active_s1_ids = set(s1_to_targets.keys())

        needed_s2_targets = {t for t in target_to_s1 if t.startswith("S2-")}
        needed_s3_targets = {t for t in target_to_s1 if t.startswith("S3-")}
        total_true_pairs = sum(len(tgts) for tgts in s1_to_targets.values())

        self.logger.info(f"Loaded {len(active_s1_ids):,} S1 entities with {total_true_pairs:,} true match pairs.")
        self.logger.info(f"Targets needed: {len(needed_s2_targets):,} from S2, {len(needed_s3_targets):,} from S3.")

        # 2. Build SQLite Index over Target records
        db_file = self.paths.indexes_dir / "train_blocking.sqlite"
        if db_file.exists():
            db_file.unlink()

        index = SQLiteBlockingIndex(db_file)
        true_targets_indexed = 0
        distractors_indexed = 0
        distractor_budget = background_distractor_count // 2

        sources = [
            (self.paths.train_source2, needed_s2_targets),
            (self.paths.train_source3, needed_s3_targets),
        ]

        for src_path, needed_set in sources:
            found_needed = 0
            found_distractors = 0
            batch_records: List[Any] = []
            with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                f.readline()  # Skip header
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    tid = parts[0].strip() if len(parts) > 0 else ""
                    if not tid:
                        continue

                    is_needed = tid in needed_set
                    is_distractor = found_distractors < distractor_budget

                    if is_needed or is_distractor:
                        rec = EntityRecord(
                            entity_id=tid,
                            business_name=parts[1].strip() if len(parts) > 1 else "",
                            business_address=parts[2].strip() if len(parts) > 2 else "",
                            country=parts[3].strip() if len(parts) > 3 else "",
                        )
                        norm_rec = self.normalizer.normalize(rec)
                        batch_records.append(norm_rec)
                        if len(batch_records) >= 5000:
                            index.bulk_insert_targets(batch_records)
                            batch_records.clear()

                        if is_needed:
                            found_needed += 1
                            true_targets_indexed += 1
                        else:
                            found_distractors += 1
                            distractors_indexed += 1

                    if found_needed >= len(needed_set) and found_distractors >= distractor_budget:
                        break

            if batch_records:
                index.bulk_insert_targets(batch_records)
                batch_records.clear()

            self.logger.info(f"  {src_path.name}: Indexed {found_needed}/{len(needed_set)} true targets + {found_distractors:,} distractors.")

        total_targets_indexed = true_targets_indexed + distractors_indexed
        bucket_stats = index.get_bucket_statistics(max_bucket_size=max_bucket_size)
        self.logger.info(f"Indexing complete! Indexed {total_targets_indexed:,} targets across {bucket_stats['total_unique_keys']:,} unique keys.")

        # Save train_blocking.json
        index.save_diagnostics(self.paths.diagnostics_dir / "train_blocking.json", max_bucket_size=max_bucket_size)

        # 3. Stream and normalize query S1 entities
        self.logger.info("Streaming and querying S1 entities against SQLite index...")
        s1_eval_results: Dict[str, Dict[str, Any]] = {}
        s1_stream_count = 0

        with open(self.paths.train_source1, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0].strip() if len(parts) > 0 else ""
                if s1_id not in active_s1_ids:
                    continue

                rec = EntityRecord(
                    entity_id=s1_id,
                    business_name=parts[1].strip() if len(parts) > 1 else "",
                    business_address=parts[2].strip() if len(parts) > 2 else "",
                    country=parts[3].strip() if len(parts) > 3 else "",
                )
                norm_rec = self.normalizer.normalize(rec)
                cands = index.query_candidates(norm_rec, max_bucket_size=max_bucket_size)
                s1_eval_results[s1_id] = cands
                s1_stream_count += 1
                if s1_stream_count >= len(active_s1_ids):
                    break

        # 4. Compute Recall, Reduction Ratio, and Precision (Candidate-Positive Rate)
        true_pairs_found = 0
        total_candidates_generated = 0
        candidate_counts: List[int] = []
        zero_candidate_s1 = 0
        singletons_s1 = 0
        matched_s1 = 0

        # Provenance tracking: recall per strategy and per hit-count threshold
        strategy_hits: Dict[str, int] = {s: 0 for s in ALL_BLOCKING_STRATEGIES}
        hit_counts_recall: Dict[str, int] = {"hits>=1": 0, "hits>=2": 0, "hits>=3": 0}

        for s1_id, true_targets in s1_to_targets.items():
            if not true_targets:
                singletons_s1 += 1
            else:
                matched_s1 += 1

            cands = s1_eval_results.get(s1_id, {})
            cand_count = len(cands)
            candidate_counts.append(cand_count)
            total_candidates_generated += cand_count

            if cand_count == 0:
                zero_candidate_s1 += 1

            for tgt in true_targets:
                if tgt in cands:
                    true_pairs_found += 1
                    hits = cands[tgt]["hits"]
                    if hits >= 1:
                        hit_counts_recall["hits>=1"] += 1
                    if hits >= 2:
                        hit_counts_recall["hits>=2"] += 1
                    if hits >= 3:
                        hit_counts_recall["hits>=3"] += 1

                    for strat in cands[tgt]["strategies"]:
                        if strat in strategy_hits:
                            strategy_hits[strat] += 1

        recall = round((true_pairs_found / total_true_pairs * 100), 2) if total_true_pairs > 0 else 100.0
        candidate_positive_rate = round((true_pairs_found / total_candidates_generated * 100), 4) if total_candidates_generated > 0 else 0.0

        # Reduction ratio: 1 - (candidates / (S1 * targets))
        total_possible_pairs = len(active_s1_ids) * total_targets_indexed
        reduction_ratio = round(1.0 - (total_candidates_generated / total_possible_pairs), 6) if total_possible_pairs > 0 else 1.0

        candidate_counts.sort()
        n_s1 = len(candidate_counts)
        mean_cands = round(total_candidates_generated / n_s1, 2) if n_s1 else 0
        p50_cands = candidate_counts[int(0.50 * n_s1)] if n_s1 else 0
        p95_cands = candidate_counts[min(int(0.95 * n_s1), n_s1 - 1)] if n_s1 else 0
        p99_cands = candidate_counts[min(int(0.99 * n_s1), n_s1 - 1)] if n_s1 else 0
        max_cands = candidate_counts[-1] if n_s1 else 0

        self.logger.info("=== EVALUATION RESULTS ===")
        self.logger.info(f"Recall: {recall}% ({true_pairs_found}/{total_true_pairs} true pairs retrieved)")
        self.logger.info(f"Reduction Ratio: {reduction_ratio * 100:.4f}% | Candidate-Positive Rate: {candidate_positive_rate}%")
        self.logger.info(f"Candidate Stats per S1: mean={mean_cands}, p50={p50_cands}, p95={p95_cands}, p99={p99_cands}, max={max_cands}")

        # Optional candidate export
        if export_candidates:
            cand_tsv = self.paths.features_dir / "train_candidates.tsv"
            self.logger.info(f"Exporting candidate pairs to {cand_tsv}...")
            with open(cand_tsv, "w", encoding="utf-8") as out_f:
                out_f.write("source1_entity_id\tcandidate_entity_ids\n")
                for s1_id, cands in s1_eval_results.items():
                    c_str = ",".join(sorted(cands.keys()))
                    out_f.write(f"{s1_id}\t{c_str}\n")

        elapsed = round(time.time() - start_time, 2)
        recall_report = {
            "elapsed_seconds": elapsed,
            "total_s1_evaluated": n_s1,
            "singletons_s1": singletons_s1,
            "matched_s1": matched_s1,
            "total_true_pairs": total_true_pairs,
            "true_pairs_retrieved": true_pairs_found,
            "recall_pct": recall,
            "reduction_ratio": reduction_ratio,
            "candidate_positive_rate_pct": candidate_positive_rate,
            "total_candidates_generated": total_candidates_generated,
            "mean_candidates_per_s1": mean_cands,
            "p50_candidates": p50_cands,
            "p95_candidates": p95_cands,
            "p99_candidates": p99_cands,
            "max_candidates": max_cands,
            "zero_candidate_s1": zero_candidate_s1,
            "bucket_statistics": bucket_stats,
            "strategy_recall_breakdown": {
                s: {"true_pairs": cnt, "recall_pct": round(cnt / total_true_pairs * 100, 2) if total_true_pairs else 0}
                for s, cnt in strategy_hits.items()
            },
            "hit_threshold_recall": {
                h: {"true_pairs": cnt, "recall_pct": round(cnt / total_true_pairs * 100, 2) if total_true_pairs else 0}
                for h, cnt in hit_counts_recall.items()
            },
        }

        # Save artifacts/diagnostics/train_blocking_recall.json
        recall_json = self.paths.diagnostics_dir / "train_blocking_recall.json"
        with open(recall_json, "w", encoding="utf-8") as f:
            json.dump(recall_report, f, indent=2)

        # Also save legacy diagnostic report
        self._save_summary_text(recall_report)

        self.logger.info(f"Blocking recall evaluation saved to {recall_json}")
        return recall_report

    def _save_summary_text(self, report: Dict[str, Any]) -> None:
        txt_path = self.paths.diagnostics_dir / "blocking_evaluation.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" AMAZON ML CHALLENGE 2026: BLOCKING RECALL & CANDIDATE EFFICIENCY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluated S1 Entities:     {report['total_s1_evaluated']:,}\n")
            f.write(f"Ground Truth Recall:       {report['recall_pct']}%\n")
            f.write(f"Candidate Reduction Ratio: {report['reduction_ratio'] * 100:.4f}%\n")
            f.write(f"Candidate-Positive Rate:   {report['candidate_positive_rate_pct']}%\n")
            f.write(f"Mean Candidates per S1:    {report['mean_candidates_per_s1']}\n")
            f.write(f"Median Candidates (p50):   {report['p50_candidates']}\n")
            f.write(f"P95 / P99 Candidates:      {report['p95_candidates']} / {report['p99_candidates']}\n")
            f.write(f"Max Candidates on Single:  {report['max_candidates']}\n\n")

            f.write("1. MULTI-PASS STRATEGY RECALL BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for strat, d in report["strategy_recall_breakdown"].items():
                f.write(f"   {strat:<22} : {d['recall_pct']:>6.2f}% ({d['true_pairs']:,} true pairs)\n")
            f.write("\n")

            f.write("2. HIT THRESHOLD RECALL BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            for hit_lbl, d in report["hit_threshold_recall"].items():
                f.write(f"   {hit_lbl:<22} : {d['recall_pct']:>6.2f}% ({d['true_pairs']:,} true pairs)\n")
            f.write("\n")
            f.write("=" * 80 + "\n")


@dataclass
class RecallReport:
    total_ground_truth_pairs: int = 0
    covered_ground_truth_pairs: int = 0
    missed_ground_truth_pairs: int = 0
    ground_truth_s1_rows: int = 0
    covered_s1_rows: int = 0
    zero_match_s1_rows: int = 0
    candidate_pairs: int = 0
    candidate_s1_rows: int = 0
    total_s1_rows: int = 0
    target_rows: int = 0
    strategy_recovered_counts: dict = None
    strategy_recall_breakdown: dict = None

    def __post_init__(self) -> None:
        if self.strategy_recovered_counts is None:
            self.strategy_recovered_counts = {}
        if self.strategy_recall_breakdown is None:
            self.strategy_recall_breakdown = {}

    @property
    def recall(self) -> float:
        return (
            self.covered_ground_truth_pairs / self.total_ground_truth_pairs
            if self.total_ground_truth_pairs else 1.0
        )

    @property
    def candidate_reduction_ratio(self) -> float:
        denominator = self.total_s1_rows * self.target_rows
        return 1.0 - self.candidate_pairs / denominator if denominator else 0.0

    def as_dict(self) -> dict:
        return {
            "total_ground_truth_pairs": self.total_ground_truth_pairs,
            "covered_ground_truth_pairs": self.covered_ground_truth_pairs,
            "missed_ground_truth_pairs": self.missed_ground_truth_pairs,
            "blocking_recall": round(self.recall, 8),
            "ground_truth_s1_rows": self.ground_truth_s1_rows,
            "covered_s1_rows": self.covered_s1_rows,
            "zero_match_s1_rows": self.zero_match_s1_rows,
            "candidate_pairs": self.candidate_pairs,
            "candidate_s1_rows": self.candidate_s1_rows,
            "total_s1_rows": self.total_s1_rows,
            "target_rows": self.target_rows,
            "candidate_positive_rate": round(
                self.covered_ground_truth_pairs / self.candidate_pairs, 8
            ) if self.candidate_pairs else 0.0,
            "candidate_reduction_ratio": round(self.candidate_reduction_ratio, 8),
            "strategy_recovered_counts": self.strategy_recovered_counts,
            "strategy_recall_breakdown": self.strategy_recall_breakdown,
        }


class BlockingRecallEvaluator:
    """Evaluates whether every labeled positive pair survived blocking."""

    def __init__(self, candidate_db: Path) -> None:
        self.candidate_db = Path(candidate_db)

    @staticmethod
    def _ground_truth(path: Path):
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            expected = {"source1_entity_id", "matched_entity_ids"}
            if set(reader.fieldnames or ()) != expected:
                raise ValueError(f"Unexpected ground-truth schema in {path}: {reader.fieldnames}")
            for row in reader:
                source1 = (row.get("source1_entity_id") or "").strip()
                matches = tuple(dict.fromkeys(
                    x.strip() for x in (row.get("matched_entity_ids") or "").split(",") if x.strip()
                ))
                yield source1, matches

    def evaluate(self, ground_truth_path: Path) -> RecallReport:
        from collections import Counter
        report = RecallReport()
        strat_counter: Counter = Counter()
        conn = sqlite3.connect(self.candidate_db)
        try:
            report.candidate_pairs = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            report.candidate_s1_rows = conn.execute(
                "SELECT COUNT(DISTINCT source1_id) FROM candidates"
            ).fetchone()[0]
            report.target_rows = conn.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
            split_row = conn.execute("SELECT value FROM meta WHERE key='split'").fetchone()
            report.total_s1_rows = conn.execute(
                "SELECT COUNT(*) FROM candidates"
            ).fetchone()[0] if split_row is None else report.total_s1_rows
            # Prefer the exact S1 population recorded by the blocking run.
            s1_meta = conn.execute("SELECT value FROM meta WHERE key='s1_rows'").fetchone()
            if s1_meta is not None:
                report.total_s1_rows = int(s1_meta[0])

            has_candidate_hits = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='candidate_hits'"
            ).fetchone() is not None

            for source1_id, matches in self._ground_truth(ground_truth_path):
                report.ground_truth_s1_rows += 1
                if not matches:
                    report.zero_match_s1_rows += 1
                    continue

                for target_id in matches:
                    report.total_ground_truth_pairs += 1
                    target_row = conn.execute(
                        "SELECT source FROM targets WHERE entity_id=? LIMIT 1",
                        (target_id,),
                    ).fetchone()
                    if target_row is None:
                        continue
                    target_source = target_row[0]
                    found = conn.execute(
                        "SELECT 1 FROM candidates WHERE source1_id=? AND target_source=? AND target_id=?",
                        (source1_id, target_source, target_id),
                    ).fetchone()
                    if found:
                        report.covered_ground_truth_pairs += 1
                        if has_candidate_hits:
                            strats = [r[0] for r in conn.execute(
                                "SELECT strategy FROM candidate_hits WHERE source1_id=? AND target_source=? AND target_id=?",
                                (source1_id, target_source, target_id),
                            )]
                            for s in strats:
                                strat_counter[s] += 1

                if all(
                    conn.execute(
                        "SELECT 1 FROM candidates WHERE source1_id=? AND target_source=? AND target_id=?",
                        (source1_id, (conn.execute("SELECT source FROM targets WHERE entity_id=? LIMIT 1", (target_id,)).fetchone() or ("UNKNOWN",))[0], target_id),
                    ).fetchone()
                    for target_id in matches
                ):
                    report.covered_s1_rows += 1
        finally:
            conn.close()

        report.missed_ground_truth_pairs = report.total_ground_truth_pairs - report.covered_ground_truth_pairs
        report.strategy_recovered_counts = dict(strat_counter.most_common())
        report.strategy_recall_breakdown = {
            strat: {
                "true_pairs": count,
                "recall_pct": round(count / report.total_ground_truth_pairs * 100.0, 2) if report.total_ground_truth_pairs else 0.0,
            }
            for strat, count in strat_counter.most_common()
        }
        return report

    @staticmethod
    def write_report(report: RecallReport, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

