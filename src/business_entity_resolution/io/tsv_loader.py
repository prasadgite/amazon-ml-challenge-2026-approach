"""
High-throughput streaming TSV data loaders.
"""

from pathlib import Path
from typing import Generator, List, Dict, Optional, Tuple, Set
import logging

from business_entity_resolution.io.record import EntityRecord

logger = logging.getLogger("TSVLoader")


def stream_records(
    file_path: Path,
    country: Optional[str] = None,
    max_rows: int = 0,
) -> Generator[EntityRecord, None, None]:
    """
    Streams EntityRecord objects line-by-line from a TSV file.
    Optionally filters by country at stream time without memory buffering.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {file_path}")

    row_count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # Header check
        header_line = f.readline()
        if not header_line:
            return

        for line in f:
            if not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")
            entity_id = parts[0].strip() if len(parts) > 0 else ""
            b_name = parts[1].strip() if len(parts) > 1 else ""
            b_addr = parts[2].strip() if len(parts) > 2 else ""
            rec_country = parts[3].strip() if len(parts) > 3 else ""

            if country and rec_country != country:
                continue

            row_count += 1
            yield EntityRecord(
                entity_id=entity_id,
                business_name=b_name,
                business_address=b_addr,
                country=rec_country,
            )

            if max_rows and row_count >= max_rows:
                break


def stream_chunks(
    file_path: Path,
    chunk_size: int = 100_000,
    country: Optional[str] = None,
    max_rows: int = 0,
) -> Generator[List[EntityRecord], None, None]:
    """
    Streams records in fixed-size batches (chunks) for batch operations.
    """
    chunk: List[EntityRecord] = []
    total_yielded = 0

    for rec in stream_records(file_path, country=country, max_rows=max_rows):
        chunk.append(rec)
        total_yielded += 1

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

        if max_rows and total_yielded >= max_rows:
            break

    if chunk:
        yield chunk


def load_ground_truth(
    gt_path: Path,
    max_rows: int = 0,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Loads ground truth mappings:
    Returns:
        s1_to_targets: {s1_id: [target_id, ...]}
        target_to_s1:  {target_id: s1_id}
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    s1_to_targets: Dict[str, List[str]] = {}
    target_to_s1: Dict[str, str] = {}
    row_count = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # Skip header
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            s1_id, sep, matches_str = line.partition("\t")
            s1_id = s1_id.strip()
            matches_str = matches_str.strip()

            if matches_str:
                targets = [m.strip() for m in matches_str.split(",") if m.strip()]
                s1_to_targets[s1_id] = targets
                for t in targets:
                    target_to_s1[t] = s1_id
            else:
                s1_to_targets[s1_id] = []

            if max_rows and row_count >= max_rows:
                break

    return s1_to_targets, target_to_s1


def stream_ground_truth_pairs(
    gt_path: Path,
    max_rows: int = 0,
) -> Generator[Tuple[str, str], None, None]:
    """
    Streams (s1_id, target_id) positive match pairs line-by-line.
    """
    row_count = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            if not line.strip():
                continue
            row_count += 1
            s1_id, sep, matches_str = line.partition("\t")
            s1_id = s1_id.strip()
            matches_str = matches_str.strip()

            if matches_str:
                for target_id in matches_str.split(","):
                    target_id = target_id.strip()
                    if target_id:
                        yield (s1_id, target_id)

            if max_rows and row_count >= max_rows:
                break
