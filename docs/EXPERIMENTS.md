
# TrialSense — Experimental Results

## Project

TrialSense: Evidence-Grounded Clinical Trial Retrieval
and Eligibility Screening

Team: Bi-Gram
Members: Khevna Vadaliya and Darshil Prajapati

## Research Objective

Investigate whether explicit patient–trial eligibility
constraints can improve clinical trial retrieval and
provide interpretable explanations for candidate trials.

TrialSense is a research prototype, not a clinical
decision-support tool for real-world enrollment.

## Dataset

- Benchmark: TREC Clinical Trials 2022
- Patient topics: 50
- Trial corpus: ClinicalTrials 2021-04-27, Part 1
- Indexed trial documents: 79,749
- Full five-part corpus: Not yet implemented

TREC labels:
- 0: Non-relevant
- 1: Excluded
- 2: Eligible

Only grade 2 receives positive relevance credit
in the current binary retrieval evaluation.

Unjudged trials do not receive positive credit;
they are not assumed to be clinically ineligible.

## Implemented Pipeline

1. Parse clinical trial XML records into JSONL.
2. Build disk-based SQLite FTS5 BM25 index.
3. Retrieve candidate trials from patient descriptions.
4. Extract patient age and sex descriptors.
5. Compare structured demographic requirements.
6. Extract supported testosterone measurements.
7. Compare supported numerical inclusion thresholds.
8. Produce evidence-grounded conflict explanations.
9. Demote conflicting candidates through stable reranking.
10. Display rankings and reference judgments separately.

The system does not determine full clinical eligibility.

## Retrieval Experiments

Evaluation: 50 development topics, Part 1 corpus,
top-10 metrics, binary grade-2 relevance.

| Method | P@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 V1 | 0.2040 | 0.1738 | 0.2636 | 0.4117 |
| Clinical V2 | 0.1840 | 0.1635 | 0.2432 | 0.3886 |
| Fusion V3 | 0.1840 | 0.1622 | 0.2541 | 0.4124 |
| Eligibility-aware V4 | 0.2240 | 0.1902 | 0.2874 | 0.4287 |

V4 retrieves the top 30 BM25 candidates and uses
explicit eligibility-conflict checks to rerank them.

Conflicting candidates are demoted, not deleted.
BM25 order is preserved within each group.

TREC labels are never used as reranking inputs.

## V4 Ablation Study

| Method | P@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Baseline | 0.2040 | 0.1738 | 0.2636 | 0.4117 |
| Demographic-only | 0.2240 | 0.1902 | 0.2874 | 0.4287 |
| Numerical-only | 0.2040 | 0.1738 | 0.2637 | 0.4117 |
| Combined | 0.2240 | 0.1902 | 0.2874 | 0.4287 |

Across 1,500 candidate positions:

- Demographic conflicts: 379
- Numerical conflicts: 2
- Combined conflicts: 379

Finding: Demographic screening drove the observed
V4 top-10 retrieval improvements.

Numerical screening produced interpretable comparisons,
but it did not add a top-10 retrieval improvement
beyond demographic screening in this experiment.

## Screening Diagnostics

Top 10 BM25 candidates from each of 50 topics:

- Patient–trial pairs: 500
- Unique clinical trials: 467
- Conflict: 102 (20.40%)
- No detected conflict: 2 (0.40%)
- Insufficient evidence: 396 (79.20%)
- Judgment coverage: 78.80%

Of 67 TREC-excluded pairs, 21 were flagged
with a supported explicit conflict.

Of 102 TREC-eligible pairs, 2 were flagged
with a supported explicit conflict.

These are diagnostic comparisons, not full
clinical eligibility precision or recall.

## Error Analysis

Within the top-30 candidate pool, three
TREC-eligible pairs had detected conflicts:

- Topic 22 / NCT00718341:
  Patient age 15; trial minimum age 18.
- Topic 34 / NCT00110318:
  Patient age 17; trial minimum age 18.
- Topic 34 / NCT00497913:
  Patient age 17; trial minimum age 18.

The registered age limits support the rule-based
conflict findings. The reasons for disagreement
with TREC judgments remain unverified.

Topic 22 is an observed retrieval regression:
V1 returns three eligible-labeled trials
in the top 10; V4 returns two.

## Paired Bootstrap Analysis

Method: Paired resampling of 50 topic-level scores.
Iterations: 5,000.
Random seed: 42.

| Metric | V4 - V1 | 95% bootstrap interval |
|---|---:|---|
| P@10 | +0.0200 | [+0.0060, +0.0360] |
| Recall@10 | +0.0164 | [+0.0040, +0.0309] |
| nDCG@10 | +0.0238 | [+0.0104, +0.0386] |
| MRR@10 | +0.0170 | [+0.0046, +0.0311] |

All reported intervals exclude zero.

These are exploratory resampling intervals
on reused development topics. They do not
establish generalization to unseen patients.

## Document Loading Optimization

Original implementation:
Sequential scan of approximately 519 MB JSONL.

Optimized implementation:
SQLite byte-offset index with direct random access.

Offset index:
- 79,749 records
- 3.78 MB index file

Five-run local benchmark:
- Documents per run: 30
- Original scan mean: 3337.87 ms
- Original scan median: 3336.86 ms
- Indexed loader mean: 3.31 ms
- Indexed loader median: 3.30 ms
- Mean speedup: 1008.77x
- Median speedup: 1009.88x
- Returned records: Identical

This benchmark measures document loading only.
It excludes BM25 retrieval and eligibility screening.
Measurements reflect local warm-cache conditions.

## Reproducibility

Python environment: 3.10.11 on Windows.

Run all tests:

    python -m pytest tests/ -q

Last verified result: 72 passed.

Run the primary evidence-grounded demo:

    python -m scripts.demo --topic 1 --top-k 3

Run the optimized V1-versus-V4 demo:

    python -m scripts.demo_v4 --topic 1 --candidate-k 30 --top-k 10

Run the ablation study:

    python -m src.evaluation.ablation_study --candidate-k 30

Run statistical analysis:

    python -m src.evaluation.statistical_analysis --iterations 5000 --seed 42

Benchmark document loading:

    python -m scripts.benchmark_document_loading --topic 1 --candidate-k 30 --runs 5

## Limitations

1. Only Part 1 of the trial corpus is indexed.
2. All 50 topics have been used during development.
3. Unjudged trials have unknown relevance.
4. Demographic and numerical screening is incomplete.
5. Numerical extraction focuses on supported
   testosterone expressions in ng/dL.
6. Complex inclusion/exclusion logic is not
   comprehensively interpreted.
7. No detected conflict does not prove eligibility.
8. The project does not verify current recruitment.
9. Clinical professionals would need to verify
   real-world patient eligibility.

## Next Research Priorities

1. Establish a held-out evaluation protocol.
2. Expand clinical evidence extraction.
3. Improve logical handling of inclusion and
   exclusion requirements.
4. Evaluate additional candidate depths.
5. Analyze performance across clinical subgroups.
6. Compare with stronger retrieval or reranking methods.
7. Integrate the full corpus if resources permit.

## Conclusion

TrialSense V4 showed promising exploratory gains
over BM25 on the current partial-corpus evaluation.

The ablation study showed that the gains were
driven by demographic constraints.

The project also demonstrates how explicit
patient–trial evidence can support transparent
screening decisions and how disk indexing can
reduce practical document-loading overhead.

Further held-out evaluation and more complete
clinical reasoning are required before making
broader performance or clinical claims.
