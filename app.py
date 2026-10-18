"""AI Resume Screening Assistant - Streamlit app (LangChain + RAG + Gemini)."""
import html
import os
from datetime import date

import pandas as pd
import streamlit as st

from evaluator import answer_question, compare_candidates, evaluate_candidate
from llm_factory import DEFAULT_LLM_MODEL, get_embeddings, get_llm
from rag_pipeline import ResumeIndex

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# ====================================================================== styling
# Editorial visual system: warm paper, charcoal ink, terracotta accents.
st.markdown(
    """
    <style>
    :root { --paper:#f7f5f0; --card:#fffefa; --ink:#202a2c; --muted:#687476; --accent:#c45b3c; --line:#e4ded2; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(247,245,240,.92); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.03em; }
    h2 { font-weight: 720 !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stSidebar"] { background: #eeebe3; border-right: 1px solid #ded8cc; }
    [data-testid="stFileUploader"], [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div > div { border-radius: 8px !important; }

    /* buttons */
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 8px; border: 1px solid #c9c2b6; font-weight: 650; min-height: 2.7rem; transition: all .15s ease;
    }
    div.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
        background: var(--accent); border-color: var(--accent); color: #fff;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); }
    div.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
        background: #a94a2e; border-color: #a94a2e; color: #fff;
    }

    /* top navigation (tabs) */
    .st-key-nav { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: .45rem .55rem; margin-bottom: 1.5rem; }
    .st-key-nav [data-testid="stButton"], .st-key-nav button { width: 100% !important; }
    .st-key-nav button { border-radius: 9px !important; border-color: transparent !important; background: transparent !important; color: var(--muted) !important; font-weight: 650; }
    .st-key-nav button:hover:not(:disabled) { background: #f1ede4 !important; color: var(--ink) !important; }
    .st-key-nav button[kind="primary"], .st-key-nav button[data-testid="stBaseButton-primary"] { background: var(--ink) !important; color: #fff !important; }
    .st-key-nav button:disabled { opacity: .4; }

    /* hero */
    .st-key-hero { background: linear-gradient(135deg, #202a2c 0%, #2f3f42 100%); border-radius: 18px; padding: 2.6rem 2.8rem; margin-bottom: 1.8rem; }
    .st-key-hero .hero-title { font-size: clamp(2rem, 4vw, 3.1rem); font-weight: 780; line-height: 1.08; letter-spacing: -0.035em; color: #f7f5f0; margin: 0 0 .9rem; }
    .st-key-hero .subhead { color: #cfd6d4 !important; font-size: 1.08rem; max-width: 720px; margin-bottom: 1.4rem; }
    .st-key-hero .eyebrow { color: #e9a58f; }
    .st-key-hero button[kind="secondary"], .st-key-hero button[data-testid="stBaseButton-secondary"] {
        background: transparent; color: #f7f5f0; border: 1px solid rgba(247,245,240,.55);
    }
    .st-key-hero button[kind="secondary"]:hover, .st-key-hero button[data-testid="stBaseButton-secondary"]:hover { background: rgba(247,245,240,.12); color: #fff; border-color: #fff; }

    /* typography helpers */
    .eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 800; margin-bottom: .45rem; }
    .page-title { font-size: clamp(1.7rem, 3vw, 2.4rem); font-weight: 760; letter-spacing: -0.035em; color: var(--ink); margin: 0 0 .35rem; line-height: 1.1; }
    .subhead { color: var(--muted); font-size: 1.02rem; max-width: 780px; margin-bottom: 1.2rem; }

    /* cards, chips, pills */
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1.25rem 1.35rem; min-height: 158px; }
    .card .num { color: var(--accent); font-weight: 800; letter-spacing: .12em; font-size: .78rem; }
    .card h4 { margin: .3rem 0 .4rem; font-size: 1.05rem; }
    .card p { margin: 0; font-size: .92rem; line-height: 1.5; }
    .chip { display: inline-block; padding: .22rem .7rem; margin: 0 .35rem .4rem 0; border-radius: 999px; font-size: .82rem; font-weight: 650; }
    .chip-ok { background: #e3efe6; color: #2f6a43; }
    .chip-miss { background: #f6e2dc; color: #a4452a; }
    .chip-neutral { background: #ece8de; color: #4b5658; }
    .pill { display: inline-block; padding: .3rem .85rem; border-radius: 999px; font-weight: 750; font-size: .88rem; }
    .pill-strong { background: #cfe6d6; color: #1f5a36; }
    .pill-hire { background: #e3efe6; color: #2f6a43; }
    .pill-maybe { background: #f8ecc9; color: #7a5a0c; }
    .pill-no { background: #f6e2dc; color: #a4452a; }
    .score { font-size: 3.1rem; font-weight: 800; line-height: 1; color: var(--ink); margin-bottom: .6rem; }
    .score small { font-size: 1rem; color: var(--muted); font-weight: 650; }
    .section-label { font-weight: 700; color: var(--ink); margin: .6rem 0 .35rem; }

    [data-testid="stMetric"] { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1.1rem; }
    [data-testid="stProgress"] > div > div { background-color: var(--accent); }
    [data-testid="stExpander"] { border: 1px solid #ded8cc; border-radius: 12px; background: var(--card); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================================================================== constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
SAMPLE_JD_FILE = os.path.join(SAMPLE_DIR, "Job_Description_Data_Scientist.txt")

FALLBACK_JD = """Data Scientist

We are looking for a Data Scientist to build and deploy machine learning models.

Requirements:
- 2+ years of experience in data science or machine learning
- Strong Python (pandas, NumPy, scikit-learn)
- SQL and data wrangling
- Experience with model deployment (Flask/FastAPI, Docker) is a plus
- Data visualization (Power BI, Tableau or Plotly)
- Familiarity with NLP or LLM applications (LangChain, RAG)
- Good communication and stakeholder management
"""

PILL_CLASS = {"Strong Hire": "pill-strong", "Hire": "pill-hire", "Maybe": "pill-maybe", "No Hire": "pill-no"}

PAGES = [
    ("home", "🏠 Home"),
    ("setup", "📤 Screen"),
    ("evaluations", "📋 Evaluations"),
    ("compare", "⚖️ Compare & Rank"),
    ("demo", "🧪 Demo scenarios"),
    ("qa", "💬 Ask resumes"),
]
RESULT_PAGES = {"evaluations", "compare", "demo", "qa"}

QA_SUGGESTIONS = [
    "Who has experience with Docker and FastAPI?",
    "Which candidates have used Tableau or Power BI?",
    "Who has worked with LangChain or RAG?",
    "Summarize each candidate's education.",
]


# ====================================================================== helpers
def resolve_api_key() -> str:
    try:
        key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GOOGLE_API_KEY", "")


@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embeddings(provider: str, api_key: str):
    return get_embeddings(provider, api_key)


def read_sample_jd() -> str:
    try:
        with open(SAMPLE_JD_FILE, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return FALLBACK_JD


def load_sample_files():
    files = []
    if os.path.isdir(SAMPLE_DIR):
        for fn in sorted(os.listdir(SAMPLE_DIR)):
            if fn.lower().endswith(".pdf"):
                with open(os.path.join(SAMPLE_DIR, fn), "rb") as f:
                    files.append((fn, f.read()))
    return files


# ---- button callbacks (run before widgets are drawn, so they may set widget state)
def go(page: str):
    st.session_state.page = page


def use_sample_data():
    st.session_state.use_sample = True
    st.session_state.jd = read_sample_jd()


def try_sample():
    use_sample_data()
    go("setup")


def remove_sample():
    st.session_state.use_sample = False


def set_question(q: str):
    st.session_state.qa_question = q


def reset_session():
    for k in ("evaluations", "eval_jd", "comparison", "pair", "index", "sig", "llm_cfg", "files_cache",
              "use_sample", "jd", "qa_answer", "qa_question", "qa_question_submitted"):
        st.session_state.pop(k, None)
    st.session_state.page = "home"


def pick(names, letter, fallback):
    """Default a selectbox to 'Resume_A...' / 'Resume_B...' when the sample files are used."""
    for i, n in enumerate(names):
        if f"resume_{letter}".lower() in n.lower():
            return i
    return min(fallback, len(names) - 1)


def chips(items, kind: str) -> str:
    if not items:
        return '<span class="chip chip-neutral">None identified</span>'
    return "".join(f'<span class="chip chip-{kind}">{html.escape(str(i))}</span>' for i in items)


def pill(recommendation: str) -> str:
    return f'<span class="pill {PILL_CLASS.get(recommendation, "pill-maybe")}">{html.escape(recommendation)}</span>'


def page_header(eyebrow: str, title: str, sub: str = ""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="subhead">{sub}</div>', unsafe_allow_html=True)


def has_results() -> bool:
    return bool(st.session_state.get("evaluations"))


def require_results() -> bool:
    if has_results():
        return True
    page_header("Nothing here yet", "No evaluations to show", "Upload resumes and run an evaluation first.")
    st.button("📤 Go to Screen", type="primary", on_click=go, args=("setup",), key="empty_go")
    return False


# ---- renderers
def render_evaluation(ev):
    c1, c2 = st.columns([1, 3])
    c1.markdown(f'<div class="score">{ev.match_score}<small>/100</small></div>{pill(ev.recommendation)}', unsafe_allow_html=True)
    c2.progress(ev.match_score / 100)
    c2.markdown(f"**Summary:** {ev.candidate_summary}")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown('<div class="section-label">✅ Matching skills</div>' + chips(ev.matching_skills, "ok"), unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="section-label">❌ Missing skills</div>' + chips(ev.missing_skills, "miss"), unsafe_allow_html=True)

    w1, w2 = st.columns(2)
    with w1:
        st.markdown('<div class="section-label">💪 Strengths</div>', unsafe_allow_html=True)
        for s in ev.strengths:
            st.markdown(f"- {s}")
    with w2:
        st.markdown('<div class="section-label">⚠️ Weaknesses</div>', unsafe_allow_html=True)
        for w in ev.weaknesses:
            st.markdown(f"- {w}")

    st.info(f"**Justification:** {ev.justification}")


def render_comparison(comp):
    st.markdown('<div class="eyebrow">Decision summary</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 🏆 Recommended candidate")
        st.markdown(f"## {comp.best_candidate}")
        st.markdown(comp.final_recommendation)
        st.caption("Decision support only: review the original resumes before making a hiring decision.")

    st.markdown("### Candidate ranking")
    for r in sorted(comp.ranking, key=lambda r: r.rank):
        with st.container(border=True):
            rank_col, detail_col = st.columns([1, 8])
            rank_col.markdown(f"## #{r.rank}")
            detail_col.markdown(f"**{r.candidate}**")
            detail_col.write(r.reason)

    st.markdown("### Why this recommendation?")
    st.info(comp.justification)


def comparison_md(comp) -> list:
    lines = [f"**Best candidate:** {comp.best_candidate}", ""]
    for r in sorted(comp.ranking, key=lambda r: r.rank):
        lines.append(f"{r.rank}. **{r.candidate}** - {r.reason}")
    lines += ["", f"**Recommendation:** {comp.final_recommendation}", "", f"**Justification:** {comp.justification}", ""]
    return lines


def build_report(evaluations, jd, comparison, pair) -> str:
    lines = ["# Resume Screening Report", f"_Generated on {date.today()}_", "", "## Job description", "", jd.strip(), ""]
    lines += ["## Candidate evaluations", ""]
    for name, e in evaluations.items():
        lines += [
            f"### {name}",
            f"- **Match score:** {e.match_score}/100",
            f"- **Recommendation:** {e.recommendation}",
            f"- **Summary:** {e.candidate_summary}",
            f"- **Matching skills:** {', '.join(e.matching_skills) or '-'}",
            f"- **Missing skills:** {', '.join(e.missing_skills) or '-'}",
            "- **Strengths:**",
            *[f"  - {s}" for s in e.strengths],
            "- **Weaknesses:**",
            *[f"  - {w}" for w in e.weaknesses],
            f"- **Justification:** {e.justification}",
            "",
        ]
    if pair:
        names, comp = pair
        lines += [f"## Comparison: {' vs '.join(names)}", ""] + comparison_md(comp)
    if comparison:
        lines += ["## Best candidate among all resumes", ""] + comparison_md(comparison)
    return "\n".join(lines)


def render_download():
    st.divider()
    st.download_button(
        "⬇️ Download screening report (Markdown)",
        data=build_report(
            st.session_state.evaluations, st.session_state.eval_jd,
            st.session_state.get("comparison"), st.session_state.get("pair"),
        ),
        file_name="resume_screening_report.md",
        mime="text/markdown",
    )


def current_llm():
    return get_llm(*st.session_state.llm_cfg)


# ====================================================================== pages
def page_home():
    with st.container(key="hero"):
        st.markdown(
            '<div class="eyebrow">Talent intelligence · Resume review</div>'
            '<div class="hero-title">Screen resumes faster,<br>grounded in evidence.</div>'
            '<div class="subhead">Upload resumes, paste a job description, and get a structured match score, skill gaps '
            "and a hiring recommendation, with every answer drawn only from the resumes you provide.</div>",
            unsafe_allow_html=True,
        )
        b1, b2, _ = st.columns([1, 1.25, 3])
        b1.button("▶  Start screening", type="primary", on_click=go, args=("setup",), key="hero_start")
        b2.button("🧪  Try with sample data", on_click=try_sample, key="hero_sample")

    if has_results():
        with st.container(border=True):
            n = len(st.session_state.evaluations)
            t, b = st.columns([4, 1])
            t.markdown(f"**Your last screening is ready** · {n} candidate(s) evaluated.")
            b.button("View results →", type="primary", on_click=go, args=("evaluations",), key="home_view")

    st.markdown('<div class="eyebrow" style="margin-top:.8rem">How it works</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload & index", "PDF resumes are read, split into chunks, embedded and stored in a private vector index per candidate."),
        ("02", "Evaluate against the JD", "The most relevant resume passages are retrieved and scored by the LLM into a structured evaluation."),
        ("03", "Compare & ask", "Rank candidates, get a best-fit recommendation, or ask free-form questions answered only from the resumes."),
    ]
    for col, (num, title, text) in zip(st.columns(3, gap="medium"), steps):
        col.markdown(f'<div class="card"><div class="num">{num}</div><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:1.6rem">What you get</div>', unsafe_allow_html=True)
    outputs = [
        ("📊", "Match score", "A 0-100 fit score for every resume."),
        ("🎯", "Skill gaps", "Matching and missing skills against the JD."),
        ("⚖️", "Strengths & weaknesses", "Evidence-based notes per candidate."),
        ("✅", "Hiring recommendation", "Strong Hire to No Hire, with justification."),
    ]
    for col, (icon, title, text) in zip(st.columns(4, gap="medium"), outputs):
        col.markdown(f'<div class="card" style="min-height:130px"><h4>{icon} {title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:1.6rem">Built with</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(f'<span class="chip chip-neutral">{t}</span>' for t in
                ["LangChain", "RAG", "FAISS", "Gemini", "HuggingFace embeddings", "Pydantic output parser", "Streamlit"]),
        unsafe_allow_html=True,
    )


def page_setup():
    page_header("Step 1 · Setup", "Screen resumes", "Upload resumes, add the job description, then run the evaluation.")

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 📤 Resumes (PDF)")
            uploaded = st.file_uploader("Upload one or more resumes", type="pdf", accept_multiple_files=True,
                                        label_visibility="collapsed")
            st.button("🧪 Use sample resumes + JD", on_click=use_sample_data, key="setup_sample")
    with right:
        with st.container(border=True):
            st.markdown("#### 📝 Job description")
            st.text_area("Job description", key="jd", height=230, label_visibility="collapsed",
                         placeholder="Paste the job description here...")
    jd = st.session_state.get("jd", "")

    if uploaded:
        files, source = [(f.name, f.getvalue()) for f in uploaded], "uploaded"
    elif st.session_state.get("use_sample"):
        files, source = load_sample_files(), "sample"
    elif st.session_state.get("files_cache"):
        files, source = st.session_state.files_cache, "previous"
    else:
        files, source = [], None

    if source == "sample":
        a, b = st.columns([5, 1])
        a.info("Using the built-in sample resumes: " + ", ".join(n for n, _ in files))
        b.button("✕ Remove sample", on_click=remove_sample, key="rm_sample")
    elif source == "previous":
        st.caption("Using the resumes from your last evaluation: " + ", ".join(n for n, _ in files))

    # readiness checklist
    def ready(ok, text_ok, text_no):
        return f'<span class="chip {"chip-ok" if ok else "chip-miss"}">{"✓" if ok else "✕"} {text_ok if ok else text_no}</span>'

    st.markdown(
        ready(bool(files), f"{len(files)} resume(s) ready", "No resumes yet")
        + ready(bool(jd.strip()), "Job description added", "Job description missing")
        + ready(bool(api_key), "API key found", "API key missing (see sidebar)"),
        unsafe_allow_html=True,
    )

    run = st.button("🔍 Evaluate resumes", type="primary", key="run_eval")
    if not run:
        return

    if not api_key:
        st.error("Please provide a Google API key in the sidebar.")
        return
    if not files:
        st.error("Please upload at least one PDF resume (or click the sample data button).")
        return
    if not jd.strip():
        st.error("Please enter a job description.")
        return

    done = False
    try:
        llm = get_llm(api_key, model_name)
        embeddings = cached_embeddings(provider, api_key)

        sig = (tuple((n, len(b)) for n, b in files), provider)
        if st.session_state.get("sig") != sig:
            with st.spinner("Reading resumes and building vector indexes..."):
                index = ResumeIndex(embeddings)
                for fname, data in files:
                    index.add_resume(fname, data)
            st.session_state.index, st.session_state.sig = index, sig
        index = st.session_state.index

        evaluations = {}
        bar = st.progress(0.0, text="Evaluating...")
        for i, name in enumerate(index.candidates):
            bar.progress(i / len(index.candidates), text=f"Evaluating {name}...")
            evaluations[name] = evaluate_candidate(index, llm, name, jd)
        bar.empty()

        st.session_state.update(
            evaluations=evaluations, eval_jd=jd, comparison=None, pair=None,
            llm_cfg=(api_key, model_name), files_cache=files, qa_answer=None, page="evaluations",
        )
        for k in ("s1", "s2a", "s2b", "s3", "s5", "qa_scope"):  # candidate list changed: reset selections
            st.session_state.pop(k, None)
        done = True
    except Exception as e:
        st.error(f"Something went wrong: {e}")
    if done:
        st.rerun()


def page_evaluations():
    if not require_results():
        return
    evaluations = st.session_state.evaluations
    ranked = sorted(evaluations.items(), key=lambda kv: kv[1].match_score, reverse=True)
    page_header("Results", "Candidate evaluations", f"{len(ranked)} candidate(s) evaluated against the job description.")

    top_name, top_ev = ranked[0]
    avg = sum(e.match_score for _, e in ranked) / len(ranked)
    k1, k2, k3 = st.columns(3)
    k1.metric("Candidates", len(ranked))
    k2.metric("Average score", f"{avg:.0f}/100")
    k3.metric(f"Top match · {top_name}", f"{top_ev.match_score}/100", delta=top_ev.recommendation, delta_color="off")

    st.markdown("&nbsp;", unsafe_allow_html=True)
    for i, (name, ev) in enumerate(ranked, start=1):
        with st.expander(f"#{i}  ·  {name}  ·  {ev.match_score}/100  ·  {ev.recommendation}", expanded=(i == 1)):
            render_evaluation(ev)
    render_download()


def page_compare():
    if not require_results():
        return
    evaluations = st.session_state.evaluations
    page_header("Results", "Compare & rank", "Review match scores and skill coverage side by side.")

    df = pd.DataFrame(
        [
            {
                "Candidate": n,
                "Match score": e.match_score,
                "Recommendation": e.recommendation,
                "Matching skills": len(e.matching_skills),
                "Missing skills": len(e.missing_skills),
            }
            for n, e in evaluations.items()
        ]
    ).sort_values("Match score", ascending=False).reset_index(drop=True)

    metric_cols = st.columns(min(4, len(df)))
    for i, (_, row) in enumerate(df.head(4).iterrows()):
        with metric_cols[i]:
            st.metric(label=f"#{i + 1}  {row['Candidate']}", value=f"{row['Match score']}/100",
                      delta=str(row["Recommendation"]), delta_color="off")

    st.markdown("### Match score overview")
    st.bar_chart(df.set_index("Candidate")[["Match score"]], horizontal=True)
    st.markdown("### Skill coverage")
    for _, row in df.iterrows():
        with st.container(border=True):
            c_name, c_match, c_missing = st.columns([2.4, 1, 1])
            c_name.markdown(f"**{row['Candidate']}**")
            c_name.markdown(pill(row["Recommendation"]), unsafe_allow_html=True)
            c_match.metric("Matching skills", int(row["Matching skills"]))
            c_missing.metric("Missing skills", int(row["Missing skills"]))

    st.markdown("### Best-fit recommendation")
    if len(evaluations) < 2:
        st.caption("Upload at least two resumes to compare candidates.")
    elif st.button("✦ Generate recommendation", key="best_main", type="primary"):
        try:
            with st.spinner("Comparing candidates..."):
                st.session_state.comparison = compare_candidates(current_llm(), evaluations, st.session_state.eval_jd)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
    if st.session_state.get("comparison"):
        render_comparison(st.session_state.comparison)
    render_download()


def page_demo():
    if not require_results():
        return
    evaluations = st.session_state.evaluations
    names = list(evaluations.keys())
    eval_jd = st.session_state.eval_jd
    llm = current_llm()
    page_header("Assignment", "Demo scenarios",
                "The five scenarios from the assignment. In the sample set, Resume A is a strong fit, B partial and C weak.")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Scenario 01</div>', unsafe_allow_html=True)
        st.subheader("Evaluate Resume A for a Data Scientist role")
        s1 = st.selectbox("Resume", names, index=pick(names, "a", 0), key="s1")
        render_evaluation(evaluations[s1])

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Scenario 02</div>', unsafe_allow_html=True)
        st.subheader("Compare Resume A and Resume B for the same JD")
        c1, c2 = st.columns(2)
        pa = c1.selectbox("First resume", names, index=pick(names, "a", 0), key="s2a")
        pb = c2.selectbox("Second resume", names, index=pick(names, "b", 1), key="s2b")
        if pa == pb:
            st.warning("Pick two different resumes.")
        elif st.button("⚖️ Compare these two", key="s2_run", type="primary"):
            try:
                with st.spinner("Comparing..."):
                    res = compare_candidates(llm, {pa: evaluations[pa], pb: evaluations[pb]}, eval_jd)
                st.session_state.pair = ([pa, pb], res)
            except Exception as e:
                st.error(f"Comparison failed: {e}")
        if st.session_state.get("pair"):
            st.caption("Result for: " + " vs ".join(st.session_state.pair[0]))
            render_comparison(st.session_state.pair[1])

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Scenario 03</div>', unsafe_allow_html=True)
        st.subheader("Identify missing skills in Resume C")
        s3 = st.selectbox("Resume", names, index=pick(names, "c", 2), key="s3")
        ev3 = evaluations[s3]
        m1, m2 = st.columns(2)
        m1.markdown('<div class="section-label">❌ Missing skills</div>' + chips(ev3.missing_skills, "miss"), unsafe_allow_html=True)
        m2.markdown('<div class="section-label">✅ Matching skills</div>' + chips(ev3.matching_skills, "ok"), unsafe_allow_html=True)
        st.progress(ev3.match_score / 100, text=f"Match score: {ev3.match_score}/100")

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Scenario 04</div>', unsafe_allow_html=True)
        st.subheader("Recommend the best candidate among multiple resumes")
        if len(names) < 2:
            st.caption("Upload at least two resumes for this scenario.")
        elif st.button("✦ Recommend the best candidate", key="s4_run", type="primary"):
            try:
                with st.spinner("Comparing all candidates..."):
                    st.session_state.comparison = compare_candidates(llm, evaluations, eval_jd)
            except Exception as e:
                st.error(f"Comparison failed: {e}")
        if len(names) >= 2 and st.session_state.get("comparison"):
            render_comparison(st.session_state.comparison)

    with st.container(border=True):
        st.markdown('<div class="eyebrow">Scenario 05</div>', unsafe_allow_html=True)
        st.subheader("Generate a hiring recommendation with justification")
        s5 = st.selectbox("Candidate", names, index=pick(names, "a", 0), key="s5")
        ev5 = evaluations[s5]
        st.markdown(f'{pill(ev5.recommendation)} &nbsp; <b>{ev5.match_score}/100</b>', unsafe_allow_html=True)
        st.info(f"**Justification:** {ev5.justification}")
    render_download()


def page_qa():
    if not require_results():
        return
    index = st.session_state.index
    names = index.candidates
    page_header("Evidence search", "Ask the resumes",
                "Ask about experience, skills, tools or qualifications. Answers come only from the uploaded resumes.")

    st.markdown('<div class="section-label">Try a question</div>', unsafe_allow_html=True)
    for col, q in zip(st.columns(len(QA_SUGGESTIONS)), QA_SUGGESTIONS):
        col.button(q, key=f"sugg_{q[:12]}", on_click=set_question, args=(q,))

    with st.container(border=True):
        scope = st.selectbox("Search scope", ["All candidates"] + names, key="qa_scope")
        question = st.text_input("Your question", key="qa_question",
                                 placeholder="e.g. Which candidate has experience with Docker and FastAPI?")
        if st.button("⌕  Search resumes", key="qa_ask", type="primary"):
            if not question.strip():
                st.warning("Enter a question to continue.")
            else:
                try:
                    with st.spinner("Searching the resume evidence..."):
                        st.session_state.qa_answer = answer_question(index, current_llm(), question, scope)
                        st.session_state.qa_question_submitted = question
                except Exception as e:
                    st.error(f"Failed to answer question: {e}")
    if st.session_state.get("qa_answer"):
        st.markdown("### Answer")
        st.caption(f"Question: {st.session_state.get('qa_question_submitted', '')}")
        with st.container(border=True):
            st.markdown(st.session_state.qa_answer)
        st.caption("Check the original resume before relying on an answer for a hiring decision.")


# ====================================================================== sidebar
st.session_state.setdefault("page", "home")
# Widgets lose their state when their page isn't drawn; re-assigning keeps these two alive across pages.
for _k in ("jd", "qa_question"):
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]

with st.sidebar:
    st.markdown("### 📄 Resume Screener")
    st.header("⚙️ Settings")
    api_key = resolve_api_key()
    if not api_key:
        api_key = st.text_input("Google API key", type="password", help="Get one free at aistudio.google.com")
    else:
        st.success("API key loaded from secrets/environment")
    model_name = st.text_input("Gemini model", value=DEFAULT_LLM_MODEL)
    embed_choice = st.selectbox(
        "Embeddings", ["HuggingFace (local, free)", "Gemini API"], help="Local has no rate limits"
    )
    provider = "gemini" if embed_choice.startswith("Gemini") else "huggingface"
    if has_results():
        st.divider()
        st.caption(f"Session: {len(st.session_state.evaluations)} candidate(s) evaluated")
        st.button("↺ Reset session", on_click=reset_session, key="reset")

# ====================================================================== navigation + routing
with st.container(key="nav"):
    for col, (key, label) in zip(st.columns(len(PAGES)), PAGES):
        col.button(
            label, key=f"nav_{key}", on_click=go, args=(key,),
            type="primary" if st.session_state.page == key else "secondary",
            disabled=(key in RESULT_PAGES and not has_results()),
        )

{
    "home": page_home,
    "setup": page_setup,
    "evaluations": page_evaluations,
    "compare": page_compare,
    "demo": page_demo,
    "qa": page_qa,
}[st.session_state.page]()