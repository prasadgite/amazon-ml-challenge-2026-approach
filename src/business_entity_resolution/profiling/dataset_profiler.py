"""
Comprehensive streaming dataset profiler for Business Entity Resolution.
Measures all 10 key structural dimensions of the competition datasets.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from collections import Counter
import json
import logging
import sys

from business_entity_resolution.paths import ProjectPaths
from business_entity_resolution.config import PipelineConfig
from business_entity_resolution.profiling.stats import StringStats, RunningStats

logger = logging.getLogger("DatasetProfiler")


class DatasetProfiler:
    """Profiles multi-million row datasets in a streaming, memory-bounded fashion."""

    def __init__(self, paths: ProjectPaths, config: PipelineConfig, log: Optional[logging.Logger] = None):
        self.paths = paths
        self.config = config
        self.logger = log or logger

    def profile_source_tsv(
        self,
        file_path: Path,
        max_rows: int = 0,
        exact_id_uniqueness: bool = False,
    ) -> Dict[str, Any]:
        """Profiles a single source TSV file."""
        self.logger.info(f"Profiling {file_path.name} (max_rows={max_rows or 'ALL'}, exact_id_uniqueness={exact_id_uniqueness})...")
        name_stats = StringStats()
        addr_stats = StringStats()
        country_counts = Counter()
        top_name_tokens = Counter()
        top_addr_tokens = Counter()
        seen_ids = set() if exact_id_uniqueness else None
        duplicate_ids = 0

        total_rows = 0
        malformed_rows = 0
        empty_rows = 0
        missing_id = 0
        missing_name = 0
        missing_addr = 0
        missing_country = 0

        file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            header_line = f.readline()
            if not header_line:
                return {"file_name": file_path.name, "error": "Empty file"}

            for line in f:
                if not line.strip():
                    empty_rows += 1
                    continue

                total_rows += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    malformed_rows += 1

                entity_id = parts[0].strip() if len(parts) > 0 else ""
                b_name = parts[1].strip() if len(parts) > 1 else ""
                b_addr = parts[2].strip() if len(parts) > 2 else ""
                country = parts[3].strip() if len(parts) > 3 else ""

                if not entity_id:
                    missing_id += 1
                elif seen_ids is not None:
                    if entity_id in seen_ids:
                        duplicate_ids += 1
                    seen_ids.add(entity_id)

                if not b_name:
                    missing_name += 1
                if not b_addr:
                    missing_addr += 1
                if not country:
                    missing_country += 1

                country_counts[country or "MISSING"] += 1
                name_stats.update(b_name if b_name else None)
                addr_stats.update(b_addr if b_addr else None)

                # Track token frequency on a 1-in-10 sample to bound memory and CPU
                if total_rows % 10 == 0:
                    for t in b_name.lower().split():
                        if len(t) > 2:
                            top_name_tokens[t] += 1
                    for t in b_addr.lower().split():
                        if len(t) > 2:
                            top_addr_tokens[t] += 1

                if max_rows and total_rows >= max_rows:
                    break

        return {
            "file_name": file_path.name,
            "file_size_mb": file_size_mb,
            "total_rows": total_rows,
            "empty_rows": empty_rows,
            "malformed_rows": malformed_rows,
            "duplicate_ids": duplicate_ids,
            "missing_fields": {
                "entity_id": missing_id,
                "business_name": missing_name,
                "business_address": missing_addr,
                "country": missing_country,
            },
            "country_distribution": dict(country_counts),
            "business_name": name_stats.to_dict(),
            "business_address": addr_stats.to_dict(),
            "top_name_tokens": top_name_tokens.most_common(15),
            "top_addr_tokens": top_addr_tokens.most_common(15),
        }

    def profile_ground_truth(self, file_path: Path, max_rows: int = 0) -> Dict[str, Any]:
        """Profiles ground truth matches, multiplicity, singletons, and target overlap."""
        self.logger.info(f"Profiling {file_path.name} (max_rows={max_rows or 'ALL'})...")
        match_count_stats = RunningStats()
        multiplicity_dist = Counter()
        s2_dist = Counter()
        s3_dist = Counter()

        total_s1 = 0
        singletons = 0
        total_links = 0
        max_matches = 0

        # Sample target overlap (first 500k targets) to bound memory while checking uniqueness
        target_counts = Counter()
        check_target_uniqueness = True

        file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()  # Skip header
            for line in f:
                if not line.strip():
                    continue
                total_s1 += 1
                parts = line.rstrip("\n").split("\t")
                matches_str = parts[1].strip() if len(parts) > 1 else ""

                if not matches_str:
                    singletons += 1
                    multiplicity_dist[0] += 1
                    match_count_stats.update(0)
                else:
                    m_list = [m.strip() for m in matches_str.split(",") if m.strip()]
                    cnt = len(m_list)
                    total_links += cnt
                    match_count_stats.update(cnt)
                    multiplicity_dist[cnt] += 1
                    if cnt > max_matches:
                        max_matches = cnt

                    s2_cnt = sum(1 for m in m_list if m.startswith("S2-"))
                    s3_cnt = sum(1 for m in m_list if m.startswith("S3-"))
                    s2_dist[s2_cnt] += 1
                    s3_dist[s3_cnt] += 1

                    if check_target_uniqueness and len(target_counts) < 1_000_000:
                        for m in m_list:
                            target_counts[m] += 1

                if max_rows and total_s1 >= max_rows:
                    break

        multi_mapped_targets = sum(1 for _, c in target_counts.items() if c > 1)
        max_s1_per_target = max(target_counts.values()) if target_counts else 0

        return {
            "file_name": file_path.name,
            "file_size_mb": file_size_mb,
            "total_s1_entities": total_s1,
            "singletons": singletons,
            "singleton_pct": round(singletons / total_s1 * 100, 2) if total_s1 > 0 else 0.0,
            "matched_s1_entities": total_s1 - singletons,
            "matched_s1_pct": round((total_s1 - singletons) / total_s1 * 100, 2) if total_s1 > 0 else 0.0,
            "total_match_links": total_links,
            "mean_matches_per_s1": round(total_links / total_s1, 2) if total_s1 > 0 else 0.0,
            "max_matches_per_s1": max_matches,
            "match_count_distribution": {str(k): v for k, v in sorted(multiplicity_dist.items())[:15]},
            "s2_match_distribution": {str(k): v for k, v in sorted(s2_dist.items())[:8]},
            "s3_match_distribution": {str(k): v for k, v in sorted(s3_dist.items())[:8]},
            "targets_checked_for_overlap": len(target_counts),
            "targets_mapped_to_multiple_s1": multi_mapped_targets,
            "max_s1_per_target": max_s1_per_target,
            "match_count_stats": match_count_stats.to_dict(),
        }

    def run_full_profile(self, max_rows: int = 0, exact_id_uniqueness: bool = False) -> Dict[str, Any]:
        """Runs profiling across all datasets and outputs structured diagnostic files."""
        self.paths.ensure_directories()
        results: Dict[str, Any] = {
            "train": {},
            "test": {},
            "ground_truth": {},
        }

        # Train files
        results["train"]["source1"] = self.profile_source_tsv(self.paths.train_source1, max_rows, exact_id_uniqueness)
        results["train"]["source2"] = self.profile_source_tsv(self.paths.train_source2, max_rows, exact_id_uniqueness)
        results["train"]["source3"] = self.profile_source_tsv(self.paths.train_source3, max_rows, exact_id_uniqueness)
        results["ground_truth"] = self.profile_ground_truth(self.paths.train_ground_truth, max_rows)

        # Test files
        results["test"]["source1"] = self.profile_source_tsv(self.paths.test_source1, max_rows, exact_id_uniqueness)
        results["test"]["source2"] = self.profile_source_tsv(self.paths.test_source2, max_rows, exact_id_uniqueness)
        results["test"]["source3"] = self.profile_source_tsv(self.paths.test_source3, max_rows, exact_id_uniqueness)

        # Save JSON
        json_path = self.paths.diagnostics_dir / "dataset_profile.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save TXT
        txt_path = self.paths.diagnostics_dir / "dataset_profile.txt"
        self._write_txt_report(txt_path, results)

        self.logger.info(f"Dataset profile successfully saved to:\n  - {txt_path}\n  - {json_path}")
        return results

    def _write_txt_report(self, txt_path: Path, results: Dict[str, Any]) -> None:
        gt = results["ground_truth"]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(" AMAZON ML CHALLENGE 2026: COMPREHENSIVE DATASET PROFILE REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. GROUND TRUTH STRUCTURE & MULTIPLICITY (TRAIN)\n")
            f.write("-" * 80 + "\n")
            f.write(f"   Total S1 Entities:              {gt['total_s1_entities']:,}\n")
            f.write(f"   Singletons (0 matches):         {gt['singletons']:,} ({gt['singleton_pct']}%)\n")
            f.write(f"   Matched S1 Entities:            {gt['matched_s1_entities']:,} ({gt['matched_s1_pct']}%)\n")
            f.write(f"   Total Match Links:              {gt['total_match_links']:,}\n")
            f.write(f"   Mean Matches per S1:            {gt['mean_matches_per_s1']}\n")
            f.write(f"   Max Matches on a Single S1:     {gt['max_matches_per_s1']}\n")
            f.write(f"   Targets Checked for Overlap:    {gt['targets_checked_for_overlap']:,}\n")
            f.write(f"   Targets Mapped to Multiple S1:  {gt['targets_mapped_to_multiple_s1']} (Strict 1-to-1 Target Mapping)\n")
            f.write(f"   Max S1s per Target:             {gt['max_s1_per_target']}\n\n")

            f.write("   Match Count Multiplicity Distribution (per S1):\n")
            for k, v in gt["match_count_distribution"].items():
                pct = round(v / gt["total_s1_entities"] * 100, 2)
                f.write(f"     {k:>2} matches: {v:>10,} ({pct:>5.2f}%)\n")
            f.write("\n")

            f.write("   S2 Matches Distribution:\n")
            for k, v in gt["s2_match_distribution"].items():
                f.write(f"     {k} S2 matches: {v:>10,}\n")
            f.write("   S3 Matches Distribution:\n")
            for k, v in gt["s3_match_distribution"].items():
                f.write(f"     {k} S3 matches: {v:>10,}\n")
            f.write("\n")

            f.write("2. SPLIT ROW COUNTS, SIZES & COUNTRY DISTRIBUTIONS\n")
            f.write("-" * 80 + "\n")
            for split in ["train", "test"]:
                f.write(f"   === {split.upper()} SET ===\n")
                for src in ["source1", "source2", "source3"]:
                    info = results[split][src]
                    f.write(f"   {info['file_name']:<22} | Rows: {info['total_rows']:>10,} | Size: {info['file_size_mb']:>7.2f} MB\n")
                    f.write(f"     Countries: {info['country_distribution']}\n")
                    f.write(f"     Missing:   {info['missing_fields']}\n")
                    f.write(f"     Name stats (chars):  mean={info['business_name']['char_length']['mean']}, p50={info['business_name']['char_length']['p50']}, p95={info['business_name']['char_length']['p95']}, max={info['business_name']['char_length']['max']}\n")
                    f.write(f"     Name stats (tokens): mean={info['business_name']['token_count']['mean']}, p50={info['business_name']['token_count']['p50']}, p95={info['business_name']['token_count']['p95']}\n")
                    f.write(f"     Addr stats (chars):  mean={info['business_address']['char_length']['mean']}, p50={info['business_address']['char_length']['p50']}, p95={info['business_address']['char_length']['p95']}, max={info['business_address']['char_length']['max']}\n")
                    f.write(f"     Addr stats (tokens): mean={info['business_address']['token_count']['mean']}, p50={info['business_address']['token_count']['p50']}, p95={info['business_address']['token_count']['p95']}\n")
                    f.write(f"     Top Name Tokens:     {info['top_name_tokens'][:8]}\n")
                    f.write(f"     Top Addr Tokens:     {info['top_addr_tokens'][:8]}\n\n")

            f.write("=" * 80 + "\n")
            f.write(" END OF PROFILE REPORT\n")
            f.write("=" * 80 + "\n")


def run_profiling(paths: ProjectPaths, config: PipelineConfig, log, exact_id_uniqueness: bool = False) -> Dict[str, Any]:
    profiler = DatasetProfiler(paths, config, log)
    return profiler.run_full_profile(max_rows=config.sample_size, exact_id_uniqueness=exact_id_uniqueness)
