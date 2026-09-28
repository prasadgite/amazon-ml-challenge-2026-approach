from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import REQUIRED_GROUND_TRUTH_COLUMNS, REQUIRED_SOURCE_COLUMNS

LOGGER = logging.getLogger(__name__)


@dataclass
class NumericSummary:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: int | None = None
    maximum: int | None = None

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self) -> dict:
        if not self.count:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
        }


@dataclass
class SourceProfile:
    path: str
    rows: int = 0
    malformed_rows: int = 0
    empty_rows: int = 0
    missing: Counter = field(default_factory=Counter)
    country_counts: Counter = field(default_factory=Counter)
    name_lengths: NumericSummary = field(default_factory=NumericSummary)
    address_lengths: NumericSummary = field(default_factory=NumericSummary)
    name_token_counts: NumericSummary = field(default_factory=NumericSummary)
    address_token_counts: NumericSummary = field(default_factory=NumericSummary)
    entity_ids_seen: int = 0
    entity_ids_unique: int | None = None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "rows": self.rows,
            "malformed_rows": self.malformed_rows,
            "empty_rows": self.empty_rows,
            "missing": dict(self.missing),
            "missing_percent": {
                k: round(v / self.rows * 100, 4) if self.rows else 0.0
                for k, v in self.missing.items()
            },
            "country_counts": dict(
                sorted(self.country_counts.items(), key=lambda x: (-x[1], x[0]))
            ),
            "country_percent": {
                k: round(v / self.rows * 100, 4) if self.rows else 0.0
                for k, v in self.country_counts.items()
            },
            "name_length": self.name_lengths.as_dict(),
            "address_length": self.address_lengths.as_dict(),
            "name_token_count": self.name_token_counts.as_dict(),
            "address_token_count": self.address_token_counts.as_dict(),
            "entity_ids_seen": self.entity_ids_seen,
            "entity_ids_unique": self.entity_ids_unique,
            "duplicate_entity_ids": (
                self.entity_ids_seen - self.entity_ids_unique
                if self.entity_ids_unique is not None
                else None
            ),
        }


@dataclass
class GroundTruthProfile:
    path: str
    rows: int = 0
    malformed_rows: int = 0
    zero_match_s1: int = 0
    singleton_s1: int = 0
    matched_s1: int = 0
    total_match_links: int = 0
    match_count_distribution: Counter = field(default_factory=Counter)
    s2_match_count_distribution: Counter = field(default_factory=Counter)
    s3_match_count_distribution: Counter = field(default_factory=Counter)
    target_to_s1: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "rows": self.rows,
            "malformed_rows": self.malformed_rows,
            "zero_match_s1": self.zero_match_s1,
            "singleton_s1": self.singleton_s1,
            "matched_s1": self.matched_s1,
            "zero_match_percent": round(self.zero_match_s1 / self.rows * 100, 4) if self.rows else 0.0,
            "singleton_percent": round(self.singleton_s1 / self.rows * 100, 4) if self.rows else 0.0,
            "total_match_links": self.total_match_links,
            "mean_matches_per_s1": round(self.total_match_links / self.rows, 4) if self.rows else 0.0,
            "max_matches_per_s1": max(self.match_count_distribution, default=0),
            "match_count_distribution": {
                str(k): v for k, v in sorted(self.match_count_distribution.items())
            },
            "s2_match_count_distribution": {
                str(k): v for k, v in sorted(self.s2_match_count_distribution.items())
            },
            "s3_match_count_distribution": {
                str(k): v for k, v in sorted(self.s3_match_count_distribution.items())
            },
            "target_mapped_to_multiple_s1": sum(
                1 for count in self.target_to_s1.values() if count > 1
            ),
            "max_s1_per_target": max(self.target_to_s1.values(), default=0),
        }


class DatasetProfiler:
    """Streaming profiler; it never loads an entire TSV into memory."""

    def __init__(self, log_every: int = 500_000) -> None:
        self.log_every = log_every

    @staticmethod
    def _open_tsv(path: Path):
        handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
        return handle, csv.reader(handle, delimiter="\t")

    @staticmethod
    def _validate_header(header: list[str], expected: tuple[str, ...], path: Path) -> None:
        if tuple(header) != expected:
            raise ValueError(
                f"Invalid header for {path.name}: expected {list(expected)}, got {header}"
            )

    @staticmethod
    def _token_count(value: str) -> int:
        return len(value.split()) if value else 0

    def profile_source(self, path: Path, exact_id_uniqueness: bool = False) -> SourceProfile:
        handle, reader = self._open_tsv(path)
        profile = SourceProfile(path=str(path))
        ids: set[str] | None = set() if exact_id_uniqueness else None

        try:
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Empty TSV file: {path}")
            self._validate_header(header, REQUIRED_SOURCE_COLUMNS, path)

            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    profile.empty_rows += 1
                    continue

                if len(row) != 4:
                    profile.malformed_rows += 1
                    continue

                entity_id, name, address, country = (cell.strip() for cell in row)
                profile.rows += 1
                profile.entity_ids_seen += 1

                if ids is not None and entity_id:
                    ids.add(entity_id)

                if not entity_id:
                    profile.missing["entity_id"] += 1
                if not name:
                    profile.missing["business_name"] += 1
                if not address:
                    profile.missing["business_address"] += 1
                if not country:
                    profile.missing["country"] += 1
                elif country:
                    profile.country_counts[country] += 1

                profile.name_lengths.add(len(name))
                profile.address_lengths.add(len(address))
                profile.name_token_counts.add(self._token_count(name))
                profile.address_token_counts.add(self._token_count(address))

                if profile.rows % self.log_every == 0:
                    LOGGER.info("%s: %,d rows", path.name, profile.rows)
        finally:
            handle.close()

        if ids is not None:
            profile.entity_ids_unique = len(ids)

        return profile

    def profile_ground_truth(self, path: Path) -> GroundTruthProfile:
        handle, reader = self._open_tsv(path)
        profile = GroundTruthProfile(path=str(path))

        try:
            header = next(reader, None)
            if header is None:
                raise ValueError(f"Empty TSV file: {path}")
            self._validate_header(header, REQUIRED_GROUND_TRUTH_COLUMNS, path)

            for row in reader:
                if not row or all(not cell.strip() for cell in row):
                    continue
                if len(row) != 2:
                    profile.malformed_rows += 1
                    continue

                matches = list(dict.fromkeys(
                    x.strip() for x in row[1].split(",") if x.strip()
                ))
                profile.rows += 1

                n = len(matches)
                profile.total_match_links += n
                profile.match_count_distribution[n] += 1

                if n == 0:
                    profile.zero_match_s1 += 1
                else:
                    profile.matched_s1 += 1
                if n == 1:
                    profile.singleton_s1 += 1

                s2 = sum(x.startswith("S2") for x in matches)
                s3 = sum(x.startswith("S3") for x in matches)
                profile.s2_match_count_distribution[s2] += 1
                profile.s3_match_count_distribution[s3] += 1

                for target_id in matches:
                    profile.target_to_s1[target_id] += 1

                if profile.rows % self.log_every == 0:
                    LOGGER.info("%s: %,d S1 ground-truth rows", path.name, profile.rows)
        finally:
            handle.close()

        return profile

    def profile_split(
        self,
        split_dir: Path,
        prefix: str,
        include_ground_truth: bool,
        exact_id_uniqueness: bool,
    ) -> dict:
        sources = {}
        for source_num in (1, 2, 3):
            path = split_dir / f"{prefix}_source{source_num}.tsv"
            if not path.exists():
                raise FileNotFoundError(path)
            sources[f"S{source_num}"] = self.profile_source(
                path, exact_id_uniqueness
            ).as_dict()

        ground_truth = None
        if include_ground_truth:
            gt_path = split_dir / f"{prefix}_ground_truth.tsv"
            ground_truth = self.profile_ground_truth(gt_path).as_dict()

        return {"sources": sources, "ground_truth": ground_truth}

    def profile_dataset(
        self,
        train_dir: Path,
        test_dir: Path,
        exact_id_uniqueness: bool = False,
    ) -> dict:
        return {
            "profiler_version": "0.2.0",
            "train": self.profile_split(
                train_dir, "train", True, exact_id_uniqueness
            ),
            "test": self.profile_split(
                test_dir, "test", False, exact_id_uniqueness
            ),
        }


def write_profile_json(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_profile_report(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "AMAZON ML 2026 — DATASET PROFILE",
        "=" * 80,
        "",
    ]

    for split_name in ("train", "test"):
        split = profile[split_name]
        lines += [split_name.upper(), "-" * 80]

        total = sum(v["rows"] for v in split["sources"].values())
        lines.append(f"Total source records: {total:,}")
        lines.append("")

        for source, info in split["sources"].items():
            lines.append(f"{source}: {Path(info['path']).name}")
            lines.append(f"  rows: {info['rows']:,}")
            lines.append(f"  malformed rows: {info['malformed_rows']:,}")
            lines.append(f"  empty rows: {info['empty_rows']:,}")
            lines.append(
                "  missingness: " + ", ".join(
                    f"{k}={v:.3f}%"
                    for k, v in info["missing_percent"].items()
                )
            )
            lines.append(
                "  countries: " + ", ".join(
                    f"{k}={v:,}"
                    for k, v in list(info["country_counts"].items())[:20]
                )
            )
            for key in ("name_length", "address_length", "name_token_count", "address_token_count"):
                stats = info[key]
                lines.append(
                    f"  {key}: min={stats.get('min')}, mean={stats.get('mean')}, "
                    f"max={stats.get('max')}, std={stats.get('std')}"
                )
            if info["entity_ids_unique"] is not None:
                lines.append(f"  duplicate entity IDs: {info['duplicate_entity_ids']:,}")
            lines.append("")

        if split["ground_truth"]:
            gt = split["ground_truth"]
            lines += [
                "GROUND TRUTH",
                f"  S1 rows: {gt['rows']:,}",
                f"  zero-match S1: {gt['zero_match_s1']:,} ({gt['zero_match_percent']:.3f}%)",
                f"  singleton S1: {gt['singleton_s1']:,} ({gt['singleton_percent']:.3f}%)",
                f"  matched S1: {gt['matched_s1']:,}",
                f"  total match links: {gt['total_match_links']:,}",
                f"  mean matches/S1: {gt['mean_matches_per_s1']:.4f}",
                f"  max matches/S1: {gt['max_matches_per_s1']}",
                f"  targets mapped to >1 S1: {gt['target_mapped_to_multiple_s1']:,}",
                f"  max S1 per target: {gt['max_s1_per_target']}",
                "  match distribution: " + ", ".join(
                    f"{k}={v:,}" for k, v in gt["match_count_distribution"].items()
                ),
                "",
            ]

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
