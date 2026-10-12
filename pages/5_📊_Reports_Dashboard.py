# pages/5_📊_Reports_Dashboard.py
import io, json, time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from core.parsing import parse_pdf_text          # PDF → text
from ai.scorer import match_score                # -> (score_pct, missing_list, matched_list)

st.set_page_config(page_title="Reports & Dashboard", layout="wide")
st.title("📊 Reports & Dashboard")
st.caption("Upload multiple CVs + a JD, compute ATS scores, explore strengths/gaps, and export. (History kept in session.)")

# ---------- Session history ----------
if "runs_history" not in st.session_state:
    st.session_state.runs_history = []   # list of {meta: {...}, df: dataframe-like records}

# ---------- Inputs ----------
with st.container():
    st.subheader("Input")
    c1, c2 = st.columns([1,1])
    with c1:
        uploaded_cvs = st.file_uploader("Upload CVs (PDF)", type=["pdf"], accept_multiple_files=True)
        save_run = st.checkbox("Save this analysis to session history", value=True)
    with c2:
        jd_text = st.text_area("Job Description (JD)", height=180, placeholder="Paste the JD here…")

    m1, m2, m3 = st.columns(3)
    with m1:
        job_title = st.text_input("Job Title (tag)", value="")
    with m2:
        team_tag = st.text_input("Team/Dept (tag)", value="")
    with m3:
        job_date = st.date_input("Analysis Date")

    run_btn = st.button("⚡ Analyze")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def _parse_cv_bytes(file_bytes: bytes) -> str:
    try:
        return parse_pdf_text(file_bytes) or ""
    except Exception as e:
        return f"__PARSE_ERROR__ {e}"

def analyze_cvs(files, jd: str) -> pd.DataFrame:
    rows = []
    for f in files:
        raw = _parse_cv_bytes(f.read())
        cv_len = len(raw)
        if raw.startswith("__PARSE_ERROR__"):
            rows.append({"Candidate": f.name, "Score": 0.0, "Matched": [], "Missing": [], "Coverage%": 0.0,
                        "Suggested Role": "-", "Error": raw, "CVChars": cv_len})
            continue
        try:
            score, missing, matched = match_score(raw, jd)
        except Exception as e:
            rows.append({"Candidate": f.name, "Score": 0.0, "Matched": [], "Missing": [], "Coverage%": 0.0,
                        "Suggested Role": "-", "Error": f"scoring failed: {e}", "CVChars": cv_len})
            continue
        cov = round(100 * len(matched) / max(1, (len(matched)+len(missing))), 1)
        rows.append({
            "Candidate": f.name,
            "Score": round(float(score), 2),
            "Matched": matched,
            "Missing": missing,
            "Coverage%": cov,
            "Suggested Role": suggest_role(jd, matched),
            "Error": "",
            "CVChars": cv_len
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    return df

# very lightweight role suggestion: map keywords → roles
ROLE_MAP = {
    "power bi": "Business Intelligence Analyst",
    "tableau": "BI Analyst",
    "sql": "Data Analyst",
    "python": "Data Scientist",
    "pytorch": "ML Engineer",
    "tensorflow": "ML Engineer",
    "rag": "AI Engineer",
    "llm": "AI Engineer",
    "aws": "Data Engineer (Cloud)",
    "azure": "Data Engineer (Cloud)",
    "gcp": "Data Engineer (Cloud)",
    "spark": "Data Engineer",
    "hadoop": "Data Engineer",
    "mlops": "MLOps Engineer",
    "nlp": "NLP Engineer",
    "cv": "Computer Vision Engineer",
}
def suggest_role(jd_text: str, matched_kw: list) -> str:
    low = f"{jd_text or ''} {' '.join(matched_kw or [])}".lower()
    for k, r in ROLE_MAP.items():
        if k in low:
            return r
    return "Generalist (Data/AI)"

def style_df(df: pd.DataFrame):
    try:
        return df.style.background_gradient(subset=["Score"], cmap="YlGn") \
                         .bar(subset=["Coverage%"], color="#c6e48b") \
                         .format({"Score": "{:.1f}%", "Coverage%": "{:.1f}%"})
    except Exception:
        return df

def explode_counts(series_like):
    d = {}
    for row in series_like:
        if isinstance(row, list):
            for w in row:
                w = (w or "").strip().lower()
                if not w: continue
                d[w] = d.get(w, 0) + 1
    return pd.Series(d, name="count").sort_values(ascending=False)

def improvement_bullets(name: str, missing: list, role_hint: str) -> list:
    """ATS-friendly tasks to fix gaps quickly."""
    miss = [m for m in (missing or []) if m]
    if not miss: 
        return [f"{name}: CV already covers core keywords for {role_hint}."]
    out = []
    for k in miss[:8]:
        out.append(f"Add a bullet demonstrating hands-on '{k}' (tool/project/result, with % impact).")
    out.append("Mirror exact phrasing from the JD in Skills/Experience (avoid keyword stuffing).")
    out.append("Ensure summary mentions 3–5 critical JD keywords naturally.")
    return out

# ---------- Run / Demo ----------
if run_btn and (not uploaded_cvs or not jd_text.strip()):
    st.warning("Please upload at least one CV and paste a JD.")
    run_btn = False

if run_btn:
    df = analyze_cvs(uploaded_cvs, jd_text)
    meta = {"job_title": job_title, "team": team_tag, "date": str(job_date), "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if save_run:
        # store a “compact” snapshot in session
        st.session_state.runs_history.append({
            "meta": meta,
            "records": df.to_dict(orient="records"),
            "jd_excerpt": (jd_text[:400] + "…") if len(jd_text) > 400 else jd_text
        })
else:
    # demo
    df = pd.DataFrame({
        "Candidate": ["CV1.pdf","CV2.pdf","CV3.pdf","CV4.pdf","CV5.pdf"],
        "Score": [79.5, 61.1, 60.7, 58.2, 55.7],
        "Matched": [["sql","power bi","dashboard"], ["excel","python"], ["python","kpi"], ["ml","cloud"], ["excel"]],
        "Missing": [["ml","python"], ["sql","bi"], ["aws"], ["tableau"], ["python","sql"]],
        "Coverage%": [60.0, 40.0, 45.0, 35.0, 30.0],
        "Suggested Role": ["Data Analyst","Data Analyst","Data Scientist","ML Engineer","Data Analyst"],
        "Error": ["","","","",""],
        "CVChars": [14000, 9000, 11000, 12000, 7500]
    }).sort_values("Score", ascending=False).reset_index(drop=True)

# ---------- Filters ----------
st.markdown("---")
st.subheader("Controls")
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    threshold = st.slider("Pass threshold (%)", 0, 100, 70)
with fc2:
    min_score = st.slider("Min score filter (%)", 0, 100, 0)
with fc3:
    name_filter = st.text_input("Search name/file", value="")
with fc4:
    show = st.selectbox("Show", ["All", "Pass only", "Review only"], index=0)

df_f = df.copy()
if name_filter.strip():
    df_f = df_f[df_f["Candidate"].str.contains(name_filter.strip(), case=False, na=False)]
df_f = df_f[df_f["Score"] >= float(min_score)]
df_f["Status"] = np.where(df_f["Score"] >= threshold, "Pass", "Review")
if show == "Pass only":
    df_f = df_f[df_f["Status"] == "Pass"]
elif show == "Review only":
    df_f = df_f[df_f["Status"] == "Review"]

# ---------- Tabs ----------
tab_overview, tab_strengths, tab_improve, tab_history, tab_export = st.tabs(
    ["Overview", "Strengths & Gaps", "Improvement Tasks", "History Insights", "Export"]
)

# =============== Overview ===============
with tab_overview:
    # KPIs
    st.subheader("KPIs")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("📂 Candidates", len(df_f))
    with k2: st.metric("🏆 Top Score", f"{df_f['Score'].max():.1f}%" if not df_f.empty else "0.0%")
    with k3: st.metric("📉 Lowest Score", f"{df_f['Score'].min():.1f}%" if not df_f.empty else "0.0%")
    with k4: st.metric("📊 Avg Score", f"{df_f['Score'].mean():.1f}%" if not df_f.empty else "0.0%")
    with k5:
        pass_rate = (df_f["Status"].eq("Pass").mean()*100) if not df_f.empty else 0.0
        st.metric("✅ Pass Rate", f"{pass_rate:.1f}%")

    st.markdown("---")
    st.subheader("Detailed Scores")
    view = df_f[["Candidate","Score","Coverage%","Suggested Role","Status","Error"]]
    st.dataframe(style_df(view), use_container_width=True)

    st.subheader("Visual Insights")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Top Candidates (Bar)**")
        fig, ax = plt.subplots()
        df_f.sort_values("Score", ascending=True).plot.barh(
            x="Candidate", y="Score", ax=ax, legend=False, color="skyblue", edgecolor="black"
        )
        ax.set_xlabel("Score (%)"); ax.set_ylabel("Candidate")
        st.pyplot(fig)
    with cb:
        st.markdown("**Score Distribution (Hist)**")
        fig2, ax2 = plt.subplots()
        ax2.hist(df_f["Score"], bins=6, edgecolor="black", color="lightgreen")
        ax2.set_xlabel("Score (%)"); ax2.set_ylabel("Count")
        st.pyplot(fig2)

# =============== Strengths & Gaps ===============
with tab_strengths:
    st.subheader("Per-candidate strengths & gaps")
    if df_f.empty:
        st.info("No data.")
    else:
        for _, r in df_f.iterrows():
            with st.expander(f"📄 {r['Candidate']} — {r['Score']:.1f}% | Coverage {r['Coverage%']:.1f}% | {r['Status']}"):
                cA, cB, cC = st.columns([1,1,1])
                with cA:
                    st.markdown("**Top Matched**")
                    st.success(", ".join((r["Matched"] or [])[:12]) if isinstance(r["Matched"], list) else "-")
                with cB:
                    st.markdown("**Top Missing**")
                    st.warning(", ".join((r["Missing"] or [])[:12]) if isinstance(r["Missing"], list) else "-")
                with cC:
                    st.markdown("**Suggested Role**")
                    st.info(r.get("Suggested Role", "-"))
                st.caption(f"CV length: {r['CVChars']} chars")

# =============== Improvement Tasks ===============
with tab_improve:
    st.subheader("AI-ready improvement tasks (ATS-friendly)")
    if df_f.empty:
        st.info("No candidates.")
    else:
        for _, r in df_f.iterrows():
            bullets = improvement_bullets(r["Candidate"], r["Missing"], r.get("Suggested Role","-"))
            with st.expander(f"🛠 {r['Candidate']} — todo list"):
                st.write("\n".join([f"- {b}" for b in bullets]))
                # quick export
                txt = "\n".join([f"- {b}" for b in bullets])
                st.download_button("⬇️ Download tasks (TXT)", data=txt, file_name=f"{r['Candidate']}_todo.txt")

# =============== History Insights ===============
with tab_history:
    st.subheader("Session History (Top Missing per JD)")
    if not st.session_state.runs_history:
        st.info("No saved analyses yet.")
    else:
        # filters
        h1, h2 = st.columns(2)
        with h1:
            jobs = ["(all)"] + [x["meta"].get("job_title","") or "(untitled)" for x in st.session_state.runs_history]
            sel_job = st.selectbox("Filter by job title (session)", jobs, index=0)
        with h2:
            teams = ["(all)"] + [x["meta"].get("team","") or "(no team)" for x in st.session_state.runs_history]
            sel_team = st.selectbox("Filter by team (session)", teams, index=0)

        # build pool
        pool = []
        for run in st.session_state.runs_history:
            if sel_job != "(all)" and run["meta"].get("job_title","") != sel_job: 
                continue
            if sel_team != "(all)" and run["meta"].get("team","") != sel_team: 
                continue
            pool.extend(run["records"])
        if not pool:
            st.info("No runs match filter.")
        else:
            pool_df = pd.DataFrame(pool)
            # explode missing
            missing_counts = explode_counts(pool_df["Missing"])
            st.markdown("**Top Missing (across session history)**")
            st.dataframe(missing_counts.head(25).to_frame(), use_container_width=True)
            fig, ax = plt.subplots()
            missing_counts.head(15).sort_values().plot.barh(ax=ax, color="salmon", edgecolor="black")
            ax.set_xlabel("Frequency"); ax.set_ylabel("Keyword")
            st.pyplot(fig)

            st.markdown("**Recent runs**")
            for i, run in enumerate(reversed(st.session_state.runs_history[-5:]), 1):
                st.write(f"{i}. {run['meta'].get('date')} • {run['meta'].get('job_title','(untitled)')} • {run['meta'].get('team','')}")
                st.caption(run.get("jd_excerpt",""))

# =============== Export ===============
with tab_export:
    st.subheader("Export current table")
    df_export = df_f.copy()
    df_export["Matched"] = df_export["Matched"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    df_export["Missing"] = df_export["Missing"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="ats_report.csv", mime="text/csv")

    try:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="ATS Report")
        xbuf.seek(0)
        st.download_button("⬇️ Download Excel", data=xbuf, file_name="ats_report.xlsx")
    except Exception as e:
        st.warning(f"Excel export unavailable: {e}")

    payload = {
        "meta": {"job_title": job_title, "team": team_tag, "date": str(job_date)},
        "records": df_f.to_dict(orient="records")
    }
    st.download_button("⬇️ Download JSON", data=json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name="ats_report.json", mime="application/json")
