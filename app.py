"""TrialSense interactive local milestone demo.

Run from repository root:
    python -m pip install streamlit
    python -m streamlit run app.py

Uses the existing locally indexed Part 1 corpus. No remote APIs are called.
"""

import time
from pathlib import Path

import streamlit as st

from src.data.trial_store import INDEX_PATH, TRIALS_PATH, load_trials_by_id
from src.reasoning.combined_screen import combined_eligibility_screen
from src.reasoning.eligibility_screen import extract_patient_age, extract_patient_sex
from src.retrieval.search_bm25 import (
    DB_PATH,
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)
from src.evaluation.evaluate_reranking import rerank_candidates


st.set_page_config(
    page_title="TrialSense | Clinical Trial Retrieval",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1440px; padding-top: 1.4rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(112deg, #101d35 0%, #18385c 100%);
       padding: 1.45rem 1.7rem; border-radius: 16px; color: #eef6ff;
       margin-bottom: 1rem;}
.hero h1 {font-size: 2.15rem; color: #f8fafc; margin: 0 0 0.3rem 0;}
.hero p {color: #dbeafe; margin: 0; font-size: 1rem;}
.micro {font-size: .84rem; opacity: .85;}
[data-testid="stMetric"] {border: 1px solid rgba(119,141,171,.27);
    border-radius: 12px; padding: .8rem 1rem; background: rgba(119,141,171,.06);}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="hero"><h1>TrialSense</h1>
    <p>Evidence-grounded clinical trial retrieval & eligibility-aware reranking</p>
    <p class="micro">NLP Final Project · Team Bi-Gram · Local research prototype</p>
    </div>""",
    unsafe_allow_html=True,
)

st.warning(
    "Research demonstration only. Screening covers a limited set of age, sex, "
    "and testosterone rules; it cannot determine clinical eligibility or recruitment status.",
    icon="⚠️",
)

missing_paths = [
    path for path in (DB_PATH, TRIALS_PATH, INDEX_PATH)
    if not Path(path).exists()
]
if missing_paths:
    st.error("Required local data files are missing:\n" + "\n".join(str(p) for p in missing_paths))
    st.code("python -m src.data.build_trial_offset_index", language="powershell")
    st.stop()


@st.cache_data(show_spinner=False)
def topic_text(topic_number: int) -> str:
    return load_patient_topic(topic_number)


@st.cache_data(show_spinner=False)
def topic_labels(topic_number: int) -> dict:
    return load_relevance_judgments(topic_number)


def reference_name(grade):
    return {
        0: "Non-relevant",
        1: "Excluded",
        2: "Eligible",
        None: "Unjudged",
    }.get(grade, "Unknown")


def make_result_rows(ranked, screenings, trials, judgments, top_k):
    rows = []
    for rank, candidate in enumerate(ranked[:top_k], 1):
        nct = candidate["nct_id"]
        rows.append({
            "Rank": rank,
            "Trial ID": nct,
            "Title": str(trials[nct].get("title") or "Untitled")[:90],
            "Screening": screenings[nct]["status"],
            "Reference label": reference_name(judgments.get(nct)),
        })
    return rows


with st.sidebar:
    st.subheader("Explore a patient case")
    topic = st.selectbox(
        "TREC patient topic",
        options=list(range(1, 51)),
        index=0,
        format_func=lambda value: f"Topic {value:02d}",
    )
    candidate_k = st.select_slider(
        "BM25 candidate pool",
        options=[10, 30, 50],
        value=30,
        help="K30 is the default evaluated configuration. K50 is exploratory.",
    )
    top_k = st.slider("Results displayed", min_value=3, max_value=10, value=10)
    show_labels = st.toggle("Show TREC reference labels", value=True)
    st.divider()
    st.caption("Recommended cases")
    st.markdown(
        "**Topic 1:** V4 gains one eligible-labeled top-10 result at K30.\n\n"
        "**Topic 22:** V4 loses one eligible-labeled result at K30.\n\n"
        "**Topic 34:** K50 improves over K30 for this topic."
    )
    st.caption("No network calls, no LLM inference, and no patient data upload.")


patient = topic_text(topic)
judgments = topic_labels(topic)

case_col, context_col = st.columns([2, 1], gap="large")
with case_col:
    st.subheader(f"Patient topic {topic:02d}")
    st.write(patient)
with context_col:
    st.subheader("Extracted patient descriptors")
    st.metric("Age", str(extract_patient_age(patient) or "Unknown"))
    st.metric("Sex descriptor", str(extract_patient_sex(patient) or "Unknown"))
    st.caption("Rule-based extraction; descriptors can be missing or ambiguous.")

try:
    with st.spinner("Retrieving trial candidates and checking supported eligibility rules..."):
        search_start = time.perf_counter()
        candidates, bm25_seconds = search_bm25(patient, top_k=candidate_k)
        retrieval_ms = bm25_seconds * 1000
        ids = {item["nct_id"] for item in candidates}

        loading_start = time.perf_counter()
        trials = load_trials_by_id(ids)
        loading_ms = (time.perf_counter() - loading_start) * 1000

        screen_start = time.perf_counter()
        screenings = {
            item["nct_id"]: combined_eligibility_screen(
                patient, trials[item["nct_id"]]
            )
            for item in candidates
        }
        reranked = rerank_candidates(candidates, screenings)
        screening_ms = (time.perf_counter() - screen_start) * 1000
        wall_ms = (time.perf_counter() - search_start) * 1000
except Exception as error:
    st.error(f"Unable to run TrialSense locally: {error}")
    st.stop()

v1_top = candidates[:top_k]
v4_top = reranked[:top_k]
v1_eligible = sum(judgments.get(item["nct_id"]) == 2 for item in v1_top)
v4_eligible = sum(judgments.get(item["nct_id"]) == 2 for item in v4_top)
conflict_total = sum(item["status"] == "CONFLICT" for item in screenings.values())

st.divider()
st.subheader("Live pipeline summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"BM25 eligible labels @ {top_k}", v1_eligible)
m2.metric(f"V4 eligible labels @ {top_k}", v4_eligible, delta=f"{v4_eligible - v1_eligible:+d}")
m3.metric("Conflicts in candidate pool", f"{conflict_total}/{len(candidates)}")
m4.metric("Measured pipeline", f"{wall_ms:.1f} ms")
st.caption(
    "Eligible-label counts are an offline TREC evaluation aid, NOT predictions. "
    "Reference labels are never used by the reranking algorithm. "
    "Wall-clock timing excludes rendering the Streamlit page."
)

st.subheader("Side-by-side retrieval rankings")
left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### BM25 V1 — original order")
    v1_rows = make_result_rows(candidates, screenings, trials, judgments, top_k)
    if not show_labels:
        for row in v1_rows:
            row.pop("Reference label", None)
    st.dataframe(v1_rows, use_container_width=True, hide_index=True, height=410)
with right:
    st.markdown("#### V4 — conflict-aware order")
    v4_rows = make_result_rows(reranked, screenings, trials, judgments, top_k)
    if not show_labels:
        for row in v4_rows:
            row.pop("Reference label", None)
    st.dataframe(v4_rows, use_container_width=True, hide_index=True, height=410)

with st.expander("What changed in the top ten?", expanded=(topic == 1)):
    original_ranks = {item["nct_id"]: rank for rank, item in enumerate(candidates, 1)}
    changes = []
    for new_rank, item in enumerate(reranked[:top_k], 1):
        old_rank = original_ranks[item["nct_id"]]
        if old_rank != new_rank:
            changes.append(
                f"**{item['nct_id']}**: BM25 rank {old_rank} → V4 rank {new_rank}"
            )
    if changes:
        st.markdown("\n\n".join(changes))
    else:
        st.write("No displayed positions changed.")

st.divider()
st.subheader("Inspect the evidence for any retrieved trial")
selected_id = st.selectbox(
    "Trial to inspect",
    options=[item["nct_id"] for item in reranked],
    format_func=lambda trial_id: f"{trial_id} — {str(trials[trial_id].get('title') or '')[:85]}",
)
trial = trials[selected_id]
check = screenings[selected_id]

info_left, info_right = st.columns([1, 2], gap="large")
with info_left:
    st.markdown(f"**Trial ID:** `{selected_id}`")
    st.markdown(f"**Screening status:** `{check['status']}`")
    st.markdown(f"**Demographic status:** `{check['demographics']['status']}`")
    st.markdown(f"**Numerical status:** `{check['numerical']['status']}`")
    st.markdown(f"**Trial age range:** {trial.get('minimum_age', 'N/A')} – {trial.get('maximum_age', 'N/A')}")
    st.markdown(f"**Trial sex field:** {trial.get('gender', 'N/A')}")
    if show_labels:
        st.markdown(f"**TREC reference label:** {reference_name(judgments.get(selected_id))}")
with info_right:
    st.markdown(f"#### {trial.get('title', 'Untitled trial')}")
    st.markdown("**Demographic evidence**")
    demographic = check["demographics"]
    for item in demographic["conflicts"]:
        st.error(item, icon="⚠️")
    for item in demographic["checks"]:
        st.success(item, icon="✅")
    for item in demographic["unknowns"]:
        st.info(item, icon="ℹ️")

    st.markdown("**Numerical evidence**")
    numeric = check["numerical"]
    if not numeric["checks"]:
        st.info(numeric["reason"])
    for item in numeric["checks"]:
        message = (
            f"Patient {item['patient_value']} {item['unit']} "
            f"{item['operator']} {item['threshold']} {item['unit']}\n\n"
            f"Patient source: {item['patient_evidence']}\n\n"
            f"Trial source: {item['trial_evidence']}"
        )
        if item["satisfied"]:
            st.success(message)
        else:
            st.error(message)

with st.expander("Original full trial eligibility criteria"):
    st.text(trial.get("eligibility_criteria") or "Not provided in the local record")

st.divider()
with st.expander("Experimental results and latency details"):
    st.markdown(
        """**Exploratory 50-topic Part 1 evaluation — K30**

| Metric | BM25 V1 | Eligibility-aware V4 |
|---|---:|---:|
| Precision@10 | 0.2040 | 0.2240 |
| Recall@10 | 0.1738 | 0.1902 |
| nDCG@10 | 0.2636 | 0.2874 |
| MRR@10 | 0.4117 | 0.4287 |

**Ablation:** Demographic-only equals combined V4 on the measured top-10 metrics; the current testosterone-only numerical logic does not add a measurable gain.

**Limitations:** Part 1 only, reused development topics, pool-based labels, incomplete clinical screening.
"""
    )
    a, b, c = st.columns(3)
    a.metric("BM25 retrieval", f"{retrieval_ms:.2f} ms")
    b.metric("Indexed loading", f"{loading_ms:.2f} ms")
    c.metric("Screening / reranking", f"{screening_ms:.2f} ms")

st.caption(
    "TrialSense | Research demonstration only | "
    "No complete clinical eligibility determination, recruitment verification, or patient-care use."
)
