# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team EntityRes  
**Team Members:** Prasad & Team  
**Submission Date:** September 28, 2026  

---

## 1. Executive Summary

We present a high-precision, low-latency, and memory-bounded Entity Resolution (ER) system designed for the Amazon ML Challenge 2026. Our solution unifies business records across three noisy independent sources into a single deduplicated identity space using a three-stage architecture: (1) **B5 Multi-Channel Inverted Index Blocking** with deterministic smart bucket capping ($\le 50$) achieving **99.98% candidate recall** while reducing candidate pair space by $>99.999%$; (2) **E0 Pairwise Feature Engineering** comprising 38 phonetic, n-gram, edit distance, and structural address features accelerated via **AVX2 SIMD Levenshtein kernels** ($>29,000$ pairs/sec); and (3) a **Platt-Calibrated Logistic Match Classifier** operated under an optimal **P0 Global Decision Policy (threshold $0.8625$)** explicitly tuned to maximize the precision-heavy Macro $F_{0.5}$ metric while guaranteeing zero false positive singleton leakage.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and holdout sets, we identified several critical failure modes:
1. **Severe Legal Suffix & Form Noise:** Business entities frequently omit or swap legal suffixes (`Pvt Ltd`, `LLC`, `SARL`, `Inc.`, `Co`).
2. **Address Transliteration & Multilingual Structural Shift:** Addresses in India contain complex landmark prefixes (`Opp SBI ATM`, `Near Metro Pillar`), French addresses use directional street syntax (`Rue`, `Boulevard`, `Chemin`), and US records exhibit variable suite/unit formatting.
3. **Open-Set Test Domain (`France`):** While training data strictly covers `US` and `India`, test data introduces `France`, requiring open-domain language normalization without hardcoded country priors.
4. **Precision-Weighted Optimization ($F_{0.5}$):** The evaluation metric penalizes false positives twice as heavily as false negatives ($\beta = 0.5$). Merging two distinct businesses (false positive) causes catastrophic precision drops, demanding a conservative decision boundary.

### 2.2 Solution Strategy
We adopted a modular, reproducible four-phase pipeline:
* **Approach Type:** Hybrid Multi-Pass Deterministic Blocking + Pairwise Supervised Linear Model + Platt Probability Calibration + Graph Transitivity Consolidation.
* **Core Innovation:**
  - **Zero-Diff Precomputed Legal Suffix Stripping & SIMD String Kernels:** Reduced text normalization latency from $8\text{k}$ to $>22\text{k}$ rec/s and pairwise edit distance evaluation by $684\times$ using RapidFuzz SIMD kernels with 0 numerical deviation.
  - **B5 Deterministic Inverted Index with Smart Capping:** Prevents combinatorial explosion on high-frequency tokens (`street`, `solutions`, `services`) by falling back to specific token intersection keys, bounding memory under $3.5$ GB across $10\text{M}$ target entities.
  - **Platt Calibrated Precision Thresholding ($0.8625$):** Experimentally proven via holdout cross-validation to maximize Macro $F_{0.5}$ and eliminate singleton overmerging.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M}$ search space ($\approx 1.73 \times 10^{13}$ pairwise comparisons) to a feasible candidate pool:

- **Blocking Keys Used (B5 Multi-Pass Strategy):**
  1. `exact_name_country`: Exact normalized core business name + country code.
  2. `prefix4_name_country`: 4-character core name prefix + country code.
  3. `sorted_tokens_name_country`: Alphabetically sorted core name tokens.
  4. `alias_blocking`: Brand/DBA alias tokens extracted from company legal forms.
  5. `transliteration_token`: Non-ASCII normalized token representations.
  6. `house_postal_country`: Exact house number + postal code intersection.
  7. `street_name_country`: Address street core + country code.
  8. `specific_fallback`: When a primary block exceeds 50 entities, smart-capping dynamically queries secondary specificity keys (e.g. name prefix + postal code) to keep buckets $\le 50$.
- **Candidate Pairs Generated:**
  - Full test dataset candidate count: **$16,734,266$ candidate pairs** ($\sim 9.6$ candidates per Source 1 entity).
  - Reduction Ratio: **$> 99.9998\%$**.
- **Recall Assurance:**
  - Evaluated on $50,000$ ground-truth verified pairs: B5 captured **$49,987 / 50,000$ true matches**, delivering **99.98% blocking recall**.

---

## 4. Matching Model

### Features Used (E0 Schema — 38 Pairwise Features):
- **Name Features:**
  - Levenshtein normalized similarity (SIMD accelerated).
  - Jaro & Jaro-Winkler similarity with prefix bonus.
  - Token Jaccard similarity, Token Overlap, and Token Inclusion.
  - Character 3-gram and 4-gram Jaccard containment.
  - Soundex & Metaphone phonetic equality.
- **Address & Structural Features:**
  - Street core Levenshtein and Jaro-Winkler similarities.
  - House number exact match, subset match, and overlap ratio.
  - Postal code exact match and prefix-3 match.
  - Address token Jaccard and containment metrics.
- **Meta Features:**
  - Blocking hit count and strategy channel diversity count.
  - Domain suffix match indicator.

### Model Architecture:
- **Model Type:** Scikit-Learn Logistic Regression with Standard Scaler feature normalization.
- **Calibration:** Platt Scaling (univariate logistic calibration on cross-validation log-odds).
- **Threshold Selection:**
  - Optimized on validation holdout sets with macro $F_{0.5}$ objective.
  - Authoritative Policy: **P0 Policy with global calibrated probability threshold $\tau = 0.8625$**.
  - High threshold strictly enforces high precision, preventing singleton false merges.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **$0.861$** (Holdout cross-validation across representative evaluation splits).
- **Blocking Recall:** **$99.98\%$** ($13$ misses out of $50,000$ test pairs).
- **Singleton False Positives:** **$0$** on verified singleton validation partitions.
- **Common False Positives (Avoided by Threshold 0.8625):**
  - Franchise chains sharing corporate names across different branches/cities.
  - Shared commercial building addresses (`DLF Cyber City`, `Banjara Hills`) housing different unrelated entities.
- **Common False Negatives (Residual Gaps):**
  - Severe brand abbreviations (e.g., `SBI` vs. `State Bank of India`) where no postal code was present in either record.
  - Missing address fields in legacy records where both house number and postal code were unrecorded.

---

## 6. Conclusion

Our solution achieves state-of-the-art Entity Resolution performance by marrying deterministic multi-pass candidate blocking with precision-calibrated linear inference. By combining SIMD C++ string distance kernels, smart bucket capping, and strict probability thresholding, our pipeline scored all 1.73M test records across 10M targets in bounded memory with 0 validator issues, ensuring robust generalization to the unseen test set.

---

## Appendix

### A. Code Artefacts
All runnable code is located in the submission package under `code/business_entity_resolution/`:
* `src/business_entity_resolution/`: Core production modules (normalization, blocking, features, models, decision).
* `artifacts/`: Pre-trained frozen Logistic Regression model (`m9_frozen_match_model.pkl`) and Platt Calibrator (`m9_frozen_decision_policy.pkl`).
* `requirements.txt`: Pinned environment specifications.
* `reproduce_submission.py`: Standalone CLI script to regenerate `matching_results.tsv` and `candidate_pairs.tsv` from raw test source files.

**Single-Command Reproduction:**
```bash
python code/business_entity_resolution/reproduce_submission.py \
    --test-dir dataset/test \
    --output-dir output
```

### B. Additional Results
* Test Set Processing Latency: **$<15$ minutes** for 1.73M S1 entities against 9.97M target entities.
* Official Validator Status: **Exit Code 0 (PASS)** on `validate_submission.py --check-ids` with 0 invalid IDs across 9.97M target entities.
