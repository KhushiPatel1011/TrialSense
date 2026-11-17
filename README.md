
# TrialSense
## Explainable Clinical Trial Retrieval and Eligibility Screening

NLP Final Project — Team Bi-Gram

**Team members:** Khevna Vadaliya and Darshil Prajapati

## 1. Project Objective

TrialSense is an NLP-based clinical trial matching prototype.

Given a patient's medical description, it:

1. Retrieves potentially relevant clinical trials.
2. Ranks them using information retrieval algorithms.
3. Displays their eligibility criteria.
4. Extracts basic patient demographic information.
5. Identifies explicit age and sex eligibility conflicts.

The current implementation is a research prototype, not
a clinical decision-support system.

## 2. Dataset

Source: TREC 2022 Clinical Trials Track

Current evaluation uses Part 1 of the April 27, 2021
ClinicalTrials.gov corpus snapshot.

Dataset statistics:

- Patient descriptions: 50
- Total relevance judgments: 35,394
- Unique judged trial IDs: 26,585
- Indexed clinical trials: 79,749
- Successfully parsed records: 79,749
- Parsing errors: 0

TREC relevance grades:

- 0: Non-relevant
- 1: Excluded
- 2: Eligible

Unjudged documents are not automatically ineligible.

## 3. Retrieval Methods

### V1: Baseline BM25

Uses SQLite FTS5 with BM25 ranking and a broad,
stopword-filtered OR query.

### V2: Clinical Term Selection

Uses heuristic medical vocabulary, morphological
patterns, and query-term filtering to construct a
shorter BM25 query.

### V3: Reciprocal Rank Fusion

Combines the top 100 results from V1 and V2 using:

RRF(d) = sum(1 / (60 + rank(d)))

The two rankings share the same underlying BM25 index.

## 4. Experimental Results

Evaluation across 50 patient topics using Part 1.

| Metric | V1 | V2 | V3 |
|---|---:|---:|---:|
| Precision@10 | 0.2040 | 0.1840 | 0.1840 |
| Recall@10 | 0.1738 | 0.1635 | 0.1622 |
| nDCG@10 | 0.2636 | 0.2432 | 0.2541 |
| MRR@10 | 0.4117 | 0.3886 | 0.4124 |
| Average latency (ms) | 340.14 | 142.25 | 483.24 |

Evaluation conventions:

- Only relevance grade 2 is counted as eligible.
- Unjudged documents receive no positive credit.
- Recall denominators use known eligible trials present
  in the indexed Part 1 corpus.
- nDCG uses binary eligibility relevance.
- Metrics are macro-averaged over patient topics.
- Reported latency measurements are preliminary.

These are exploratory subset results, not official
full-corpus TREC benchmark scores.

## 5. Key Findings

- V1 achieved the highest Precision@10, Recall@10,
  and nDCG@10 of the three approaches.
- V2 had the lowest observed retrieval latency.
- V3 did not outperform V1 on the primary retrieval
  quality metrics.
- Clinical term selection alone is insufficient to
  guarantee better eligibility-focused retrieval.
- Lexical similarity does not establish clinical eligibility.

## 6. Eligibility Screening

The current reasoning module extracts explicit patient
age and sex information and compares them against
structured trial requirements.

Possible demographic screening results:

- PASS: Checked demographic constraints are satisfied.
- CONFLICT: An explicit demographic mismatch is found.
- UNKNOWN: Information is insufficient.

PASS does not imply full clinical eligibility.

Other medical inclusion and exclusion criteria are
displayed as evidence but are not yet automatically
evaluated.

## 7. Live Demonstration

From the project root with the virtual environment active:

```powershell
python -m scripts.demo --topic 1 --top-k 3
```

To display the experiment summary:

```powershell
python -m scripts.demo --summary-only
```

To run the eligibility-screening tests:

```powershell
python -m pytest tests/test_eligibility_screen.py -v
```

## 8. Example: Patient Topic 1

Patient: 19-year-old male with hormonal abnormalities
and low testosterone.

Retrieved examples:

- NCT00064987: TREC excluded; demographic checks pass.
- NCT00493961: TREC eligible; demographic checks pass.
- NCT00240981: TREC excluded; age requirement conflicts
  with the patient's age.

This case demonstrates why clinical trial matching
requires eligibility reasoning beyond text retrieval.

## 9. Current Limitations

- Only Part 1 of the clinical trial corpus is indexed.
- The system has no semantic embedding model yet.
- Medical eligibility criteria are not fully parsed.
- Patient demographic extraction uses limited rules.
- The system does not predict overall clinical eligibility.
- No held-out experimental evaluation has been performed.
- Repeated tuning on the same 50 topics may overfit
  experimental decisions.
- Results from the historical corpus do not establish
  current trial recruitment availability.

## 10. Planned Improvements

1. Expand to the complete TREC trial corpus.
2. Implement semantic retrieval using biomedical
   sentence embeddings.
3. Compare lexical, semantic, and hybrid methods.
4. Extract structured inclusion/exclusion criteria.
5. Add evidence-grounded clinical eligibility reasoning.
6. Evaluate reasoning accuracy separately from retrieval.
7. Add controlled held-out experiments and error analysis.
8. Develop a user-facing interface.

## 11. Research Question

Can combining information retrieval with evidence-based
eligibility reasoning improve clinical trial matching
while providing interpretable explanations?

The current implementation establishes the retrieval
baseline and a limited demographic screening component.
```
