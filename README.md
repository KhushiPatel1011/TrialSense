# TrialSense

**Evidence-Grounded Clinical Trial Retrieval and Eligibility Screening**

**NLP Final Project — Team Bi-Gram**  
**Members:** Khevna Vadaliya and Darshil Prajapati

TrialSense retrieves potentially relevant clinical trials for a free-text patient description, then applies **limited, interpretable eligibility checks** to help expose clear mismatches in requirements. It is an academic **research prototype, not a clinical eligibility determination or enrollment tool**.

## Project question

Can screening for explicit patient–trial conflicts improve clinical-trial retrieval rankings compared with BM25 alone, while also showing the evidence behind screening decisions?

## Current research status

- TREC Clinical Trials 2022: **50 patient topics** and official relevance judgments.
- Local trial corpus: **79,749 records from Part 1** of the ClinicalTrials 2021-04-27 collection (not all five parts).
- Retrieval: BM25 V1, clinical-term V2, reciprocal-rank fusion V3, eligibility-aware V4.
- Reasoning: patient age and sex extraction; supported testosterone measurement and threshold parsing; combined demographic/numerical checks.
- Performance: SQLite FTS5 retrieval and a compact SQLite byte-offset index for direct trial-record loading.
- Verified testing: **72 passing automated tests** at the last recorded checkpoint.

### Research results (exploratory, 50 reused development topics)

| Model | P@10 | Recall@10 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 V1 | 0.2040 | 0.1738 | 0.2636 | 0.4117 |
| Clinical V2 | 0.1840 | 0.1635 | 0.2432 | 0.3886 |
| Fusion V3 | 0.1840 | 0.1622 | 0.2541 | 0.4124 |
| **Eligibility-aware V4 (K30)** | **0.2240** | **0.1902** | **0.2874** | **0.4287** |

V4's observed improvement over BM25 V1 is primarily **driven by demographic filtering**. Limited numerical reasoning adds explainability but not further top-10 improvement in the ablation study. These are **not** held-out or official full-corpus TREC results.

For the detailed methods, bootstrap confidence intervals, error analysis, candidate-depth study, and performance benchmark, see **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

## How TrialSense works

```text
Patient narrative
      |
      v
SQLite FTS5 / BM25 retrieves top K candidates
      |
      v
Load exact trial records via SQLite byte-offset index
      |
      +--> Extract age/sex and supported measurements
      |
      +--> Compare explicit trial demographic / numerical rules
      |
      v
CONFLICT / NO_DETECTED_CONFLICT / INSUFFICIENT_EVIDENCE
      |
      v
V4 demotes explicitly conflicting candidates,
preserving BM25 order within groups
      |
      v
Top 10 results + evidence + separate TREC reference labels
```

**Critical semantics:** `NO_DETECTED_CONFLICT` does not establish medical eligibility, and `INSUFFICIENT_EVIDENCE` does not mean eligible or ineligible. TREC judgments are used for offline evaluation/display, **never for reranking**.

## Setup (Windows PowerShell, Python 3.10+)

From the repository root, with the project virtual environment activated:

```powershell
python -m pip install numpy pandas scikit-learn rank-bm25 sentence-transformers lxml tqdm pytest
```

If a working local environment already exists, reuse it. The full dataset, SQLite databases, and generated indexes are intentionally excluded from Git and must be prepared locally. The command-line demos require these files:

```text
data/raw/topics2022.xml
data/raw/qrels2022.txt
data/processed/trials_part1.jsonl
data/processed/trials_bm25_part1.db
data/processed/trials_offset_index.db
```

The corpus is parsed and BM25 indexed using the existing project scripts (`src/data/parse_trials.py`, `src/retrieval/build_bm25_index.py`). After the JSONL exists, create/rebuild the offset index with:

```powershell
python -m src.data.build_trial_offset_index
```

The SQLite offset index checks the JSONL file size and modification time and will report a stale index if the corpus changes; rebuild it if necessary.

## Command-line demonstrations

**End-to-end evidence screening (Topic 1):**

```powershell
python -m scripts.demo --topic 1 --top-k 3
```

**Side-by-side BM25 V1 vs V4 (Topic 1, candidate depth 30):**

```powershell
python -m scripts.demo_v4 --topic 1 --candidate-k 30 --top-k 10
```

**Known regression (Topic 22):**

```powershell
python -m scripts.demo_v4 --topic 22 --candidate-k 30 --top-k 10
```

**Candidate-depth case (Topic 34):**

```powershell
python -m scripts.demo_v4 --topic 34 --candidate-k 50 --top-k 10
```

### Optional local Streamlit interface

Install the UI dependency:

```powershell
python -m pip install streamlit
```

Launch the app from the repository root:

```powershell
python -m streamlit run app.py
```

The app uses the **existing** retrieval/screening pipeline and indexed store; it does not call an LLM or a remote API. If the UI is unavailable, the command-line demonstrations above remain the reliable fallback.

## Evaluation and quality checks

```powershell
python -m pytest tests/ -q
python -m src.evaluation.evaluate_reranking --candidate-k 30
python -m src.evaluation.ablation_study --candidate-k 30
python -m src.evaluation.statistical_analysis --iterations 5000 --seed 42
python -m src.evaluation.compare_candidate_depths --iterations 5000 --seed 42
python -m scripts.benchmark_document_loading --topic 1 --candidate-k 30 --runs 5
```

**Note:** `evaluate_reranking` writes to `results/bm25_v4_reranking.csv`, which the statistical-analysis script reads by default. Evaluating another candidate depth may overwrite it. Save a named copy and restore the K30 CSV if you need to reproduce K30 results.

### Candidate depth sensitivity

| Candidate pool | P@10 | nDCG@10 | Notes |
|---|---:|---:|---|
| 10 | 0.2040 | 0.2723 | Reorders only existing top 10 |
| **30 (default)** | **0.2240** | **0.2874** | Good current quality/cost balance |
| 50 | 0.2260 | 0.2905 | One additional topic improves over K30 |

## Known limitations

1. Only the first of five trial corpus parts is indexed.
2. The same 50 patient topics were reused during development; no untouched test set is available.
3. Many retrieved documents are unjudged, and grade-2-only binary metrics are incomplete proxies for relevance.
4. Age/sex extraction and testosterone checks use restricted rules, not comprehensive clinical natural-language inference.
5. Inclusion/exclusion Boolean logic, treatment history, negation, unit conversion, temporality, and missing data are incompletely handled.
6. V4 sometimes demotes trials that TREC labels eligible; documented age-conflict cases are in `docs/EXPERIMENTS.md`.
7. Trial recruitment status is not checked live; never use this prototype for real enrollment decisions.

## Engineering notes

The local document loader benchmark (30 identical records, five warm-cache runs) found a mean of **3,337.87 ms** for a full JSONL scan versus **3.31 ms** for indexed lookup, about **1,008.77× faster for document loading**. This is not an end-to-end search speedup.

The current Git checkpoint for the main implementation is `7daa559`. Keep generated raw and processed data outside version control.

## Next milestones

Freeze the current V4 implementation, test on genuinely new patient topics and ideally the complete trial corpus, extend clinical-criteria interpretation, and evaluate supported conflict rules separately from retrieval ranking metrics.
