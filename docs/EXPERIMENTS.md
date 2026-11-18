# TrialSense — Experimental Record

**Project:** TrialSense — Evidence-Grounded Clinical Trial Retrieval and Eligibility Screening  
**Team:** Bi-Gram — Khevna Vadaliya and Darshil Prajapati  
**Status:** Working research prototype; figures verified from local runs.

## Research question

Can explicit patient–trial eligibility constraints improve the ranking of clinical trials retrieved from narrative patient descriptions, while producing traceable evidence for detected conflicts?

TrialSense is a research prototype. Its screening output **does not determine patient eligibility**, medical suitability, or whether a trial is recruiting.

## Data and evaluation scope

- **Dataset:** TREC Clinical Trials 2022, 50 patient topics, relevance judgments (`qrels2022.txt`).
- **Corpus indexed:** 79,749 trial documents from the **first** `ClinicalTrials.2021-04-27` archive part; the full five-part corpus is **not** indexed.
- **Trial representation:** Parsed XML → JSONL (~519 MB) → SQLite FTS5 for BM25 (~104 MB).
- **Labels:** 0 = non-relevant; 1 = excluded; 2 = eligible. Only grade 2 receives positive credit in our *binary* ranking metrics.
- **Important:** Unjudged retrieved trials do not receive positive credit for these metrics, but **must not** be interpreted as ineligible.
- **Evaluation limitation:** All 50 topics were used during development; results are **exploratory**, not held-out, and are **not official TREC benchmark scores**.

## Implemented components

1. Trial XML parsing and persistent, disk-based SQLite FTS5 retrieval.
2. Baseline BM25 V1, heuristic clinical-query V2, and reciprocal-rank fusion V3.
3. Patient age/sex descriptor extraction with context-aware handling of mentions such as “male partners.”
4. Demographic conflict checks against trial minimum/maximum age and sex eligibility fields.
5. Explicit testosterone measurement extraction in supported `ng/dL` patterns.
6. A limited numerical threshold parser, including connected `and >` / `and <` testosterone comparisons.
7. Combined screening with `CONFLICT`, `NO_DETECTED_CONFLICT`, and `INSUFFICIENT_EVIDENCE` outputs, including source excerpts.
8. V4 stable reranking: retrieve BM25 top **30** by default, move explicitly conflicting candidates behind other candidates, and preserve BM25 order within each group. **No official judgments are used to rank candidates**.
9. Index-based document lookup using a 3.78 MB SQLite byte-offset index, avoiding full scans of the ~519 MB JSONL file for interactive use.
10. Evaluation scripts, ablations, audit utilities, CLI demos, and automated tests.

**Interpretation:** `NO_DETECTED_CONFLICT` only means *the supported checks found no conflict*. It never means the trial is clinically eligible.

## Comparison of retrieval strategies

**Part 1 corpus, 50 development topics, grade-2 binary relevance, top-10 evaluation.** V4 uses BM25 candidate depth K=30.

| Method | Precision@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 V1 | 0.2040 | 0.1738 | 0.2636 | 0.4117 |
| Clinical V2 | 0.1840 | 0.1635 | 0.2432 | 0.3886 |
| RRF Fusion V3 | 0.1840 | 0.1622 | 0.2541 | 0.4124 |
| **Eligibility-aware V4 (K30)** | **0.2240** | **0.1902** | **0.2874** | **0.4287** |

Relative V4 improvement over V1: **+9.8% Precision@10**, **+9.4% Recall@10**, **+9.0% nDCG@10**, **+4.1% MRR@10**. Improvements are measured on the reused development set, not unseen data.

### Paired bootstrap analysis: V4-K30 minus V1

5,000 resamples of **paired topic-level scores**, seed 42.

| Metric | Mean difference | Exploratory 95% percentile bootstrap CI |
|---|---:|---|
| Precision@10 | +0.0200 | [+0.0060, +0.0360] |
| Recall@10 | +0.0164 | [+0.0040, +0.0309] |
| nDCG@10 | +0.0238 | [+0.0104, +0.0386] |
| MRR@10 | +0.0170 | [+0.0046, +0.0311] |

All four intervals are above zero under this resampling procedure. Reusing topics during system design and partial-corpus/pooling effects remain important limitations.

## Ablation: which screening component drives the gain?

1,500 topic–candidate positions (50 × 30).

| Reranker | Precision@10 | Recall@10 | nDCG@10 | MRR@10 | Conflicts detected |
|---|---:|---:|---:|---:|---:|
| V1, no reranking | 0.2040 | 0.1738 | 0.2636 | 0.4117 | — |
| Demographic only | 0.2240 | 0.1902 | 0.2874 | 0.4287 | 379 |
| Numerical only | 0.2040 | 0.1738 | 0.2637 | 0.4117 | 2 |
| Combined V4 | 0.2240 | 0.1902 | 0.2874 | 0.4287 | 379 |

**Finding:** The ranking improvement is attributable to **demographic screening** in this experiment. Numerical screening supplies interpretable comparisons, but does not improve top-10 scores beyond demographics. The two numerical conflicts overlap with demographic conflicts.

## Screening audit

For the **top 10 BM25 trials per patient** (500 patient–trial pairs, 467 unique trials):

| Screening state | Count | Percentage |
|---|---:|---:|
| Conflict | 102 | 20.4% |
| No detected conflict | 2 | 0.4% |
| Insufficient evidence | 396 | 79.2% |

Judgments: 102 eligible, 67 excluded, 225 non-relevant, 106 unjudged; **78.8% judged coverage**. Explicit conflicts were detected for 21/67 excluded-labeled pairs. Two of the 102 eligible-labeled pairs were flagged. These figures are *diagnostics*, not clinical sensitivity/specificity.

At K30, three eligible-labeled candidates had age conflicts:

| Topic | Trial | Patient age | Trial minimum | TREC label |
|---|---|---:|---:|---|
| 22 | NCT00718341 | 15 | 18 | Eligible |
| 34 | NCT00110318 | 17 | 18 | Eligible |
| 34 | NCT00497913 | 17 | 18 | Eligible |

These age mismatches are supported by the structured trial records; the reason for their disagreement with the relevance labels is unverified. We do not modify official judgments.

### Demonstration cases

- **Topic 1:** V1 has 2 eligible-labeled trials in the top 10; V4-K30 has 3. Explicit age and testosterone conflicts are both detected for `NCT00240981`.
- **Topic 22:** V1 has 3 eligible-labeled trials; V4-K30 has 2 because an age-conflicting trial with a TREC eligible label is demoted.
- **Topic 34:** Comparing V4-K30 and V4-K50 shows one additional eligible-labeled result in the top 10 at K50; this is the **only topic** with changed aggregate retrieval metrics between those depths.

## Candidate-depth sensitivity

| Configuration | Precision@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| V1 baseline | 0.2040 | 0.1738 | 0.2636 | 0.4117 |
| V4 K10 | 0.2040 | 0.1738 | 0.2723 | 0.4239 |
| **V4 K30 — default** | **0.2240** | **0.1902** | **0.2874** | **0.4287** |
| V4 K50 | 0.2260 | 0.1969 | 0.2905 | 0.4294 |

With K10, the same ten trials are reordered, so Precision@10 and Recall@10 cannot change; rank-sensitive metrics can improve. K50 improves one of 50 topics over K30 and leaves 49 unchanged.

### Direct paired bootstrap: K50 minus K30

5,000 paired topic bootstrap resamples, seed 42.

| Metric | Mean difference | 95% bootstrap CI |
|---|---:|---|
| Precision@10 | +0.0020 | [+0.0000, +0.0060] |
| Recall@10 | +0.0067 | [+0.0000, +0.0200] |
| nDCG@10 | +0.0031 | [+0.0000, +0.0093] |
| MRR@10 | +0.0007 | [+0.0000, +0.0020] |

Intervals have lower endpoints at zero (rounded to four decimals). This does **not** establish a robust additional gain for K50. K30 remains the practical default, not a statistically proven optimum.

## Document-access performance

An offset index enables random access into the original trial JSONL. This was also developed to keep memory use low on a Windows laptop.

| Five-run local benchmark, same 30 records | Full JSONL scan | SQLite offset lookup |
|---|---:|---:|
| Mean time | 3,337.87 ms | 3.31 ms |
| Median time | 3,336.86 ms | 3.30 ms |
| Mean relative speedup | — | **1,008.77×** |
| Median relative speedup | — | **1,009.88×** |

**Record equality was verified.** These are local warm-cache measurements of **document loading only**; they exclude BM25 retrieval and screening. A sample V4 Topic 1 run measured 317.32 ms retrieval + 3.68 ms indexed loading + 5.71 ms screening/reranking = 326.72 ms for those measured stages, not the entire UI response.

An initial Windows `WinError 32` occurred while renaming an open SQLite database. It was fixed by explicitly closing the SQLite connection before replacement; regression tests then passed.

## Verification and commands

The last reported complete run: **72 passed** (`python -m pytest tests/ -q`). Git checkpoint at `7daa559`; three new candidate-depth CSV files and the comparison script may require a subsequent commit.

```powershell
python -m pytest tests/ -q
python -m scripts.demo --topic 1 --top-k 3
python -m scripts.demo_v4 --topic 1 --candidate-k 30 --top-k 10
python -m scripts.demo_v4 --topic 22 --candidate-k 30 --top-k 10
python -m scripts.demo_v4 --topic 34 --candidate-k 50 --top-k 10
python -m src.evaluation.ablation_study --candidate-k 30
python -m src.evaluation.statistical_analysis --iterations 5000 --seed 42
python -m src.evaluation.compare_candidate_depths --iterations 5000 --seed 42
python -m scripts.benchmark_document_loading --topic 1 --candidate-k 30 --runs 5
```

The offset index file and raw/processed clinical data are local generated assets; they are not Git-tracked.

## Unresolved limitations and next steps

- Full five-part corpus not yet evaluated.
- Development topics reused; a genuinely independent test set is needed for generalization claims.
- Judgment pooling produces unjudged candidates, creating evaluation uncertainty.
- Clinical NLP handles only a small subset of inclusion/exclusion rules, especially numerical rules.
- The observed V4 improvement comes from demographics; stronger clinical reasoning needs to be developed and ablated.
- Complex Boolean logic, temporal conditions, negations, treatment history, and unit normalization remain incomplete.
- Three documented TREC-label disagreements merit further adjudication, but we cannot relabel the benchmark.
- No live ClinicalTrials.gov recruitment verification, patient safety validation, or clinician-reviewed eligibility assessment.

**Milestone conclusion:** A functioning, auditable retrieval-and-screening prototype exists, with encouraging partial-corpus exploratory retrieval gains. The primary future research contribution should be broader and rigorously evaluated clinical eligibility reasoning, not simply more rules without validation.
