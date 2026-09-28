#!/usr/bin/env python3
"""
Business Entity Resolution: End-to-End Submission Reproducer.
Frozen Pipeline: B5 Blocker + E0 Features + Logistic Regression + Platt + P0 Policy (Threshold 0.8625) + M8.

Regenerates matching_results.tsv and candidate_pairs.tsv from test data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    csv.field_size_limit(2147483647)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR / "src"))

import numpy as np

# RapidFuzz SIMD acceleration
import rapidfuzz.distance.Levenshtein as rf_lev
import business_entity_resolution.features.pairwise as pw
pw._levenshtein = rf_lev.distance

from business_entity_resolution.blocking.blocker import BlockingConfig, BlockingEngine
from business_entity_resolution.features.pairwise import FeatureSchema, PairFeatureEngine
from business_entity_resolution.normalization import normalizer
from business_entity_resolution.normalization.normalizer import TextNormalizer, _char_ngrams

# Precompute legal suffixes
_PRECOMPUTED_SUFFIXES = {}
for _c, _s_list in normalizer.LEGAL_SUFFIXES.items():
    _PRECOMPUTED_SUFFIXES[_c] = [
        tuple(normalizer._tokens(normalizer._normalize_unicode(s))) for s in _s_list
    ]

def _fast_strip_legal_suffixes(tokens: tuple[str, ...], country: str) -> tuple[str, ...]:
    suffixes = _PRECOMPUTED_SUFFIXES.get(country, _PRECOMPUTED_SUFFIXES["default"])
    current = list(tokens)
    changed = True
    while changed and current:
        changed = False
        for suffix in suffixes:
            s_len = len(suffix)
            if len(current) >= s_len and tuple(current[-s_len:]) == suffix:
                current = current[:-s_len]
                changed = True
                break
    return tuple(current)

normalizer._strip_legal_suffixes = _fast_strip_legal_suffixes

_TRANSLATE_TABLE = str.maketrans({
    chr(i): (chr(i).lower() if (65 <= i <= 90 or 97 <= i <= 122 or 48 <= i <= 57) else " ")
    for i in range(128)
})
_orig_norm_unicode = normalizer._normalize_unicode

def _fast_normalize_unicode(value: str) -> str:
    if value.isascii():
        return " ".join(value.translate(_TRANSLATE_TABLE).split())
    return _orig_norm_unicode(value)

normalizer._normalize_unicode = _fast_normalize_unicode

FROZEN_P0_THRESHOLD = 0.8625
FEATURE_COLUMNS = FeatureSchema.columns(False)[3:]


def load_model_and_calibrator(artifacts_dir: Path):
    manifest_path = artifacts_dir / "final" / "model_manifest.json"
    if not manifest_path.exists():
        manifest_path = artifacts_dir / "model_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_path = artifacts_dir / "models" / manifest["model"]["artifact"]
    calibrator_path = artifacts_dir / "models" / manifest["calibration"]["artifact"]

    with model_path.open("rb") as f:
        raw_model = pickle.load(f)
    with calibrator_path.open("rb") as f:
        raw_calib = pickle.load(f)

    model = raw_model.get("pipeline", raw_model) if isinstance(raw_model, dict) else raw_model
    calibrator = raw_calib.get("calibrator", raw_calib) if isinstance(raw_calib, dict) else raw_calib

    scaler = model.named_steps["scale"]
    lr = model.named_steps["model"]
    w_eff = lr.coef_[0] / scaler.scale_
    b_eff = lr.intercept_[0] - float(np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))

    platt_a = float(calibrator.model.coef_[0, 0])
    platt_b = float(calibrator.model.intercept_[0])
    target_z_cal = float(np.log(FROZEN_P0_THRESHOLD / (1.0 - FROZEN_P0_THRESHOLD)))
    target_z_lr = (target_z_cal - platt_b) / platt_a

    return model, calibrator, w_eff, b_eff, target_z_lr


def run_pipeline(test_dir: Path, output_dir: Path, artifacts_dir: Path) -> int:
    print("=" * 72)
    print("BUSINESS ENTITY RESOLUTION: REPRODUCIBLE TEST SCORING RUNNER")
    print("=" * 72)

    model, calibrator, w_eff, b_eff, target_z_lr = load_model_and_calibrator(artifacts_dir)

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    print("Step 1: Reading Source 1 records...")
    t0 = time.time()
    s1_order: list[str] = []
    s1_by_country: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    with s1_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["entity_id"]
            c = row.get("country", "")
            s1_order.append(eid)
            s1_by_country[c].append((eid, row.get("business_name", ""), row.get("business_address", ""), c))
    print(f"Loaded {len(s1_order):,} Source 1 records across {len(s1_by_country)} countries in {time.time() - t0:.1f}s.")

    all_matches: dict[str, str] = {}
    all_candidates: dict[str, str] = {}

    block_cfg = BlockingConfig(
        max_bucket_size=50,
        enable_alias_blocking=True,
        enable_transliteration=True,
        enable_country_fallback=True,
        enable_smart_capping=True,
        enable_secondary_identity=False,
    )
    blocker = BlockingEngine(Path("dummy"), block_cfg)
    norm = TextNormalizer()

    countries = list(s1_by_country.keys())
    for country in countries:
        c_records = s1_by_country[country]
        print(f"\n--- Processing {country}: {len(c_records):,} records ---")
        t_cstart = time.time()

        print(f"  Indexing {country} targets from Source 2 and Source 3...")
        t_idx = time.time()
        compact_targets: dict[str, tuple] = {}
        index: dict[str, list[str]] = defaultdict(list)
        t_count = 0

        for path in (s2_path, s3_path):
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if row.get("country") != country:
                        continue
                    t_count += 1
                    eid = row["entity_id"]
                    rec = norm.normalize(eid, row.get("business_name", ""), row.get("business_address", ""), country)
                    compact_targets[eid] = (
                        rec.name_core,
                        rec.name_tokens,
                        rec.name_sorted_tokens,
                        rec.address_street_core,
                        rec.address_street_tokens,
                        rec.house_numbers,
                        rec.postal_codes,
                        rec.domains,
                        rec.name_aliases,
                        rec.country,
                    )
                    for strat, key in blocker._block_keys(rec):
                        if "char4" in strat or "street_prefix" in strat:
                            continue
                        lst = index[key]
                        if len(lst) <= 50:
                            lst.append(eid)
                    for strat, key in blocker._specific_keys_for_record(rec):
                        lst = index[key]
                        if len(lst) <= 50:
                            lst.append(eid)

        print(f"  Indexed {t_count:,} targets across {len(index):,} keys in {time.time() - t_idx:.1f}s.")

        print(f"  Querying and scoring {len(c_records):,} Source 1 records...")
        t_score = time.time()
        c_cands = 0
        c_matches = 0

        for idx, (s1_id, b_name, b_addr, _) in enumerate(c_records, 1):
            rec = norm.normalize(s1_id, b_name, b_addr, country)
            s1_dict = {
                "source": "S1",
                "entity_id": rec.entity_id,
                "country": rec.country,
                "name_core": rec.name_core,
                "name_tokens": rec.name_tokens,
                "name_sorted_tokens": rec.name_sorted_tokens,
                "name_aliases": rec.name_aliases,
                "domains": rec.domains,
                "name_char3": rec.name_char3,
                "name_char4": rec.name_char4,
                "address_street_core": rec.address_street_core,
                "address_street_tokens": rec.address_street_tokens,
                "house_numbers": rec.house_numbers,
                "postal_codes": rec.postal_codes,
                "address_char3": rec.address_char3,
                "address_char4": rec.address_char4,
            }

            s1_candidates = set()
            for strat, key in blocker._block_keys(rec):
                if "char4" in strat or "street_prefix" in strat:
                    continue
                bucket = index.get(key)
                if bucket is not None and len(bucket) <= 50:
                    for tid in bucket:
                        s1_candidates.add(tid)
                elif bucket is not None:
                    for spec_strat, spec_key in blocker._specific_keys_for_record(rec):
                        spec_bucket = index.get(spec_key)
                        if spec_bucket is not None and len(spec_bucket) <= 50:
                            for tid in spec_bucket:
                                s1_candidates.add(tid)

            cand_list = sorted(s1_candidates)
            c_cands += len(cand_list)
            cand_str = ",".join(cand_list)
            all_candidates[s1_id] = cand_str

            if not cand_list:
                all_matches[s1_id] = ""
                continue

            feat_matrix = []
            target_ids = []
            for tid in cand_list:
                compact = compact_targets.get(tid)
                if not compact:
                    continue
                t_dict = {
                    "source": "S2" if tid.startswith("S2") else "S3",
                    "entity_id": tid,
                    "country": compact[9],
                    "name_core": compact[0],
                    "name_tokens": compact[1],
                    "name_sorted_tokens": compact[2],
                    "address_street_core": compact[3],
                    "address_street_tokens": compact[4],
                    "house_numbers": compact[5],
                    "postal_codes": compact[6],
                    "domains": compact[7],
                    "name_aliases": compact[8],
                    "name_char3": _char_ngrams(compact[0], 3),
                    "name_char4": _char_ngrams(compact[0], 4),
                    "address_char3": _char_ngrams(compact[3], 3),
                    "address_char4": _char_ngrams(compact[3], 4),
                }
                feats = PairFeatureEngine._similarity_features(s1_dict, t_dict, hit_count=1, strategy_count=1)
                feat_matrix.append([float(feats.get(c, 0.0) or 0.0) for c in FEATURE_COLUMNS])
                target_ids.append(tid)

            if not feat_matrix:
                all_matches[s1_id] = ""
                continue

            X = np.asarray(feat_matrix, dtype=float)
            z_lr = np.dot(X, w_eff) + b_eff
            passed = z_lr >= target_z_lr

            boundary = np.abs(z_lr - target_z_lr) < 0.05
            if np.any(boundary):
                raw_prob = model.predict_proba(X[boundary])[:, 1]
                cal_prob = calibrator.predict(raw_prob)
                passed[boundary] = cal_prob >= FROZEN_P0_THRESHOLD

            matches = [tid for tid, is_match in zip(target_ids, passed) if is_match]
            matches_sorted = sorted(set(matches))
            if matches_sorted:
                c_matches += len(matches_sorted)
                all_matches[s1_id] = ",".join(matches_sorted)
            else:
                all_matches[s1_id] = ""

            if idx % 50_000 == 0 or idx == len(c_records):
                elapsed = time.time() - t_score
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(c_records) - idx) / rate if rate > 0 else 0
                print(f"    Processed {idx:,}/{len(c_records):,} in {elapsed:.1f}s ({rate:.0f} rec/s, ETA: {eta:.0f}s)...")

        print(f"  {country} completed in {time.time() - t_cstart:.1f}s.")
        del compact_targets
        del index

    output_dir.mkdir(parents=True, exist_ok=True)
    match_out = output_dir / "matching_results.tsv"
    cand_out = output_dir / "candidate_pairs.tsv"

    print(f"\nWriting final submission files to {output_dir}...")
    t_write = time.time()
    with match_out.open("w", encoding="utf-8", newline="") as fm, \
         cand_out.open("w", encoding="utf-8", newline="") as fc:

        wm = csv.writer(fm, delimiter="\t", lineterminator="\n")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        matched_entities = 0
        total_accepted = 0
        for s1_id in s1_order:
            m_str = all_matches.get(s1_id, "")
            c_str = all_candidates.get(s1_id, "")
            wm.writerow([s1_id, m_str])
            wc.writerow([s1_id, c_str])
            if m_str:
                matched_entities += 1
                total_accepted += len(m_str.split(","))

    print(f"Successfully wrote {len(s1_order):,} rows in {time.time() - t_write:.1f}s.")
    print(f"Matched S1 entities: {matched_entities:,} / {len(s1_order):,}")
    print(f"Total accepted match edges: {total_accepted:,}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Reproduce Business Entity Resolution matching and candidate results.")
    parser.add_argument("--test-dir", type=Path, default=SCRIPT_DIR.parents[1] / "dataset/student_resource/dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR.parents[1] / "output", help="Path to output directory")
    parser.add_argument("--artifacts-dir", type=Path, default=SCRIPT_DIR / "artifacts", help="Path to model artifacts")
    args = parser.parse_args()

    return run_pipeline(args.test_dir, args.output_dir, args.artifacts_dir)


if __name__ == "__main__":
    raise SystemExit(main())
