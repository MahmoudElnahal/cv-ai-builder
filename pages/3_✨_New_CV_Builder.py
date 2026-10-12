# pages/3_✨_New_CV_Builder.py
import re, streamlit as st
from typing import List, Dict, Tuple

from core.export import (
    export_docx, export_pdf,
    export_docx_cover_letter, export_pdf_cover_letter
)
from ai.agent import CVAgent             # generate_summary(seed_cv, jd, strength=?), generate_bullets(role, jd, strength=?), (optional) generate_cover_letter(cv, jd, strength=?)
from ai.scorer import match_score        # returns (score_pct, missing_keywords, matched_keywords)

st.set_page_config(page_title="New CV Builder", layout="wide")
st.title("✨ New CV Builder (Inputs + JD)")
st.caption("Type your info + paste JD → Get a full, ATS-ready Master CV (same quality as CV Update) and optional Cover Letter.")

# =============== Session state for persistence ===============
for k, v in [
    ("newcv_cv", None),
    ("newcv_cover_letter", ""),
    ("newcv_report", None),
]:
    if k not in st.session_state: st.session_state[k] = v

# ===================== Helper utilities ======================
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}")
STOP = {
    "and","or","with","for","the","a","an","in","to","of","on","by","as","at","is","are","be",
    "job","description","position","overview","team","candidate","will","responsible","this","role",
    "experience","skills","strong","advanced","basic","both","company","degree","bachelor","master",
    "requirements","preferred","must","nice","plus"
}
TOOLS = {
    "python","pandas","numpy","pytorch","tensorflow","keras","scikit-learn","xgboost","lightgbm","catboost",
    "sql","dbt","airflow","spark","hadoop","power","tableau","excel","looker","superset","matplotlib","plotly",
    "langchain","transformers","opencv","fastapi","flask","streamlit",
    "docker","kubernetes","mlflow","ray","dask",
    "faiss","weaviate","pinecone","milvus","lancedb","elasticsearch",
    "snowflake","bigquery","redshift","databricks"
}
CLOUD = {"aws","azure","gcp","sagemaker","vertex","athena","glue","emr"}

def _csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]

def _lines(s: str) -> List[str]:
    return [x.strip().lstrip("-• ").strip() for x in (s or "").split("\n") if x.strip()]

def _dedup(seq: List[str], max_len: int = None) -> List[str]:
    out, seen = [], set()
    for x in seq:
        k = x.lower()
        if k not in seen and x:
            out.append(x)
            seen.add(k)
            if max_len and len(out) >= max_len: break
    return out

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 2 and t.lower() not in STOP]

def _normalize_skill(s: str) -> str:
    if not s: return ""
    t = s.replace("_"," ").replace("-"," ").strip()
    acr = {"sql","etl","nlp","rag","llm","bi","kpi","kpis","mlops"}
    if t.lower() in acr: return t.upper()
    return t.title()

def enrich_skills_from_jd(cv_skills: List[str], jd_text: str, cap: int = 22) -> (List[str], List[str]):
    """Merge user's skills with JD tokens; return (merged_skills, tools_cloud)."""
    base = {x.lower(): x.strip() for x in cv_skills if x.strip()}
    toks = _tokens(jd_text)
    tools = sorted({_normalize_skill(t) for t in toks if t in TOOLS})
    cloud = sorted({_normalize_skill(t) for t in toks if t in CLOUD})

    for k in toks:
        if k not in base and k not in STOP:
            base[k] = k
    merged = [_normalize_skill(v) for v in base.values() if v.lower() not in STOP]
    bad = {"data","analysis","experience","strong","skills","role","job","candidate","team","overview","position"}
    merged = [m for m in merged if m and m.lower() not in bad]
    merged = _dedup(merged, max_len=cap)
    tools_cloud = _dedup(tools + cloud, max_len=16)
    return merged, tools_cloud

def merge_bullets(user_bullets: List[str], ai_bullets: List[str], cap: int = 8) -> List[str]:
    ai_bullets = [b.strip().rstrip(".") for b in (ai_bullets or []) if b.strip()]
    merged = _dedup(user_bullets + ai_bullets, max_len=cap)
    return [b.replace("\n"," ").replace("\r"," ")[:220] for b in merged]

# ============================ Form ============================
with st.form("new_cv_form"):
    st.subheader("Contact & Profile")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Full Name*", value="John Doe")
        title = st.text_input("Professional Title*", value="Data Analyst")
        email = st.text_input("Email*", value="john.doe@email.com")
        phone = st.text_input("Phone*", value="+1 555 123 4567")
        location = st.text_input("Location*", value="Cairo, Egypt")
    with c2:
        linkedin = st.text_input("LinkedIn", value="linkedin.com/in/johndoe")
        links    = st.text_input("Other Links (GitHub/Portfolio)", value="github.com/johndoe")

    st.subheader("Summary & Skills")
    c3, c4 = st.columns(2)
    with c3:
        summary = st.text_area(
            "Professional Summary*",
            height=120,
            value="Data analyst with strong SQL, Power BI, and Python skills. Experienced in KPI dashboards and data storytelling."
        )
    with c4:
        skills_str = st.text_area(
            "Core Skills (comma-separated)*",
            height=100,
            value="SQL, Power BI, Tableau, Python, Statistics, KPI tracking, Dashboards"
        )

    st.subheader("Experience (one bullet per line)")
    exp_text = st.text_area(
        "Experience*",
        height=140,
        value="- Built HR dashboards in Power BI leading to 30% faster reporting\n- Automated KPI tracking with SQL (20% effort reduction)\n- Collaborated with stakeholders to define KPIs and success metrics"
    )

    colP, colE = st.columns(2)
    with colP:
        st.subheader("Projects (one per line)")
        proj_text = st.text_area(
            "Projects",
            height=120,
            value="- Fraud Detection — Python/Scikit-learn → 87% accuracy\n- ATS Reporting System — Streamlit + FastAPI\n- Customer Segmentation — scikit-learn + Power BI"
        )
    with colE:
        st.subheader("Education (one per line)")
        edu_lines = st.text_area(
            "Education*",
            height=120,
            value="B.Sc. in Artificial Intelligence — Kafr El-Sheikh University (2024)"
        )

    st.subheader("Certifications (comma-separated)")
    certs_str = st.text_input("Examples: Huawei HCIA-AI (2022), Coursera ML Specialization (2020)")

    st.subheader("Job Description (for tailoring)")
    jd_text = st.text_area(
        "Paste JD here",
        height=180,
        value="We are hiring a Data Analyst with SQL, Power BI, dashboarding, and stakeholder communication experience."
    )

    with st.expander("AI Options", expanded=True):
        ai_tailoring = st.slider("AI tailoring strength (wording only)", 0, 100, 80)
        want_ai_summary = st.checkbox("Generate/Refine Summary with AI", value=True)
        want_ai_bullets = st.checkbox("Generate Tailored Bullets with AI", value=True)
        want_ai_cover   = st.checkbox("Generate Cover Letter (AI)", value=False)
        run_match_report= st.checkbox("Compute Match Score vs JD", value=True)

    submitted = st.form_submit_button("🚀 Generate Master CV")

# ========================== Submit ============================
if submitted:
    # Validate
    errs = []
    if not name.strip():  errs.append("Name is required.")
    if not title.strip(): errs.append("Professional Title is required.")
    if not summary.strip() and not want_ai_summary: errs.append("Summary is required (or enable AI summary).")
    if len(_lines(exp_text)) == 0 and not want_ai_bullets:
        errs.append("Experience bullets are required (or enable AI bullets).")
    if errs:
        for e in errs: st.error(e)
    else:
        skills_user = _csv(skills_str)
        certs       = _csv(certs_str) if certs_str else []
        bullets_user= _lines(exp_text)
        projects    = _lines(proj_text) if proj_text else []
        education   = _lines(edu_lines)

        # Enrich skills from JD (merge + tools/cloud extraction)
        merged_skills, tools_cloud = enrich_skills_from_jd(skills_user, jd_text)

        agent = CVAgent()

        # 1) Summary (AI → fallback to user summary)
        final_summary = summary
        if want_ai_summary and jd_text.strip():
            try:
                final_summary = agent.generate_summary({
                    "name": name,
                    "title": title,
                    "skills": merged_skills,
                    "experience": [{"bullets": bullets_user}],
                    "education": education,
                    "projects": projects,
                    "certificates": certs
                }, jd_text, strength=ai_tailoring) or summary
            except Exception as e:
                st.warning(f"AI summary failed: {e}")
                final_summary = summary

        # 2) Bullets (AI → merge with user entries)
        ai_bullets = []
        if want_ai_bullets and jd_text.strip():
            try:
                ai_bullets = agent.generate_bullets(title, jd_text, strength=ai_tailoring) or []
            except Exception as e:
                st.warning(f"AI bullets failed: {e}")
                ai_bullets = []
        final_bullets = merge_bullets(bullets_user, ai_bullets, cap=8)

        # ===== Build final CV (same structure as page 2) =====
        cv = {
            "name": name,
            "title": title,
            "location": st.text_input if False else (st.session_state.get("tmp", None) or ""),  # avoid lints
            "location": st.session_state.pop("tmp", None) if False else (location or "-"),
            "email": email or "-",
            "phone": phone or "-",
            "linkedin": linkedin or "-",
            "links": links or "-",
            "summary": final_summary,
            "skills": merged_skills,           # Core Competencies
            "tools": tools_cloud,              # Tools & Tech
            "experience": [{"bullets": final_bullets}],
            "projects": projects,
            "education": education,
            "certificates": certs,
            "languages": ["Arabic – Native", "English – Good (Writing & Speaking)"],
            "achievements": [
                "Top 4 Trainee – Samsung Innovation Campus (2023)",
                "Innovation Program – InnovaEgypt (2023)"
            ],
        }

        st.session_state.newcv_cv = cv

        # 3) Match Score (optional)
        report = None
        if run_match_report and jd_text.strip():
            try:
                cv_text_for_score = "\n".join([
                    cv["summary"],
                    ", ".join(cv["skills"]),
                    ", ".join(cv.get("tools", [])),
                    "\n".join(cv["education"]),
                    "\n".join(cv["projects"]),
                    "\n".join(cv["certificates"]),
                    "\n".join(cv["experience"][0]["bullets"])
                ])
                score, missing, matched = match_score(cv_text_for_score, jd_text)
                report = {"score": round(score, 1), "missing": missing, "matched": matched}
                st.session_state.newcv_report = report
            except Exception as e:
                st.warning(f"Match scoring failed: {e}")

        # 4) Cover Letter (optional)
        cover_letter = ""
        if want_ai_cover and jd_text.strip():
            try:
                if hasattr(agent, "generate_cover_letter"):
                    cover_letter = agent.generate_cover_letter(cv, jd_text, strength=ai_tailoring) or ""
                else:
                    # fallback prompt via agent.cover_letter if موجود
                    cover_letter = getattr(agent, "cover_letter", lambda *_: "")(cv, jd_text) or ""
            except Exception as e:
                st.warning(f"Cover letter generation failed: {e}")
        st.session_state.newcv_cover_letter = cover_letter

        # ===================== Preview =====================
        st.markdown("## ✅ CV Preview")
        st.write(f"""**{cv['name']}**  
{cv['title']}  
{cv['location']}  
📧 {cv['email']} | 📱 {cv['phone']}  
🔗 {cv['linkedin']} | {cv['links']}""")

        st.markdown("---")
        st.subheader("Executive Summary");    st.write(cv["summary"] or "-")

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Core Competencies"); st.write(", ".join(cv["skills"]) or "-")
        with cB:
            st.subheader("Tools & Tech");      st.write(", ".join(cv.get("tools", [])) or "-")

        st.subheader("Experience — JD-tailored Results")
        st.write("\n".join(f"- {b}" for b in cv["experience"][0]["bullets"]) or "-")

        if cv["projects"]:     st.subheader("Projects");        st.write("\n".join(f"- {p}" for p in cv["projects"]))
        if cv["education"]:    st.subheader("Education");       st.write("\n".join(cv["education"]))
        if cv["certificates"]: st.subheader("Certifications");  st.write("\n".join(cv["certificates"]))
        if cv.get("languages"):st.subheader("Languages");       st.write(", ".join(cv["languages"]))
        if cv.get("achievements"): st.subheader("Achievements");st.write("\n".join(cv["achievements"]))

        # ATS report
        st.markdown("---")
        st.subheader("ATS Match (quick)")
        if report:
            st.metric("Score", f"{report['score']}%")
            a, b = st.columns(2)
            with a: st.markdown("**Matched**"); st.success(", ".join(report["matched"]) if report["matched"] else "-")
            with b: st.markdown("**Missing**"); st.warning(", ".join(report["missing"]) if report["missing"] else "-")
        else:
            st.info("Enable 'Compute Match Score vs JD' to view the quick ATS report.")

        # ===================== Downloads =====================
        st.markdown("---")
        st.subheader("Downloads")
        try:
            st.download_button("⬇️ Download CV (DOCX)", data=export_docx(cv), file_name="new_master_cv.docx")
        except Exception as e:
            st.error(f"DOCX export failed: {e}")
        try:
            st.download_button("⬇️ Download CV (PDF)", data=export_pdf(cv), file_name="new_master_cv.pdf")
        except Exception as e:
            st.error(f"PDF export failed: {e}")

        if cover_letter:
            st.markdown("---")
            st.subheader("📮 Cover Letter (Preview)")
            st.text_area("Cover Letter", value=cover_letter, height=260)
            try:
                st.download_button("⬇️ Download Cover Letter (DOCX)",
                                   data=export_docx_cover_letter(cv, cover_letter),
                                   file_name="cover_letter.docx")
            except Exception as e:
                st.error(f"Cover Letter DOCX export failed: {e}")
            try:
                st.download_button("⬇️ Download Cover Letter (PDF)",
                                   data=export_pdf_cover_letter(cv, cover_letter),
                                   file_name="cover_letter.pdf")
            except Exception as e:
                st.error(f"Cover Letter PDF export failed: {e}")

# ================= Show last generated outputs =================
if st.session_state.newcv_cv and not submitted:
    st.info("Showing last generated CV from this session.")
    cv = st.session_state.newcv_cv
    st.markdown("## CV Preview")
    st.write(f"""**{cv['name']}**  
{cv['title']}  
{cv['location']}  
📧 {cv['email']} | 📱 {cv['phone']}  
🔗 {cv['linkedin']} | {cv['links']}""")
    st.markdown("---")
    st.subheader("Executive Summary"); st.write(cv["summary"] or "-")

    cA, cB = st.columns(2)
    with cA: st.subheader("Core Competencies"); st.write(", ".join(cv["skills"]) if cv["skills"] else "-")
    with cB: st.subheader("Tools & Tech");      st.write(", ".join(cv.get("tools", [])) if cv.get("tools") else "-")

    st.subheader("Experience — JD-tailored Results")
    st.write("\n".join(f"- {b}" for b in cv["experience"][0]["bullets"]) if cv.get("experience") else "-")
    if cv.get("projects"):     st.subheader("Projects");       st.write("\n".join(f"- {p}" for p in cv["projects"]))
    if cv.get("education"):    st.subheader("Education");      st.write("\n".join(cv["education"]))
    if cv.get("certificates"): st.subheader("Certifications"); st.write("\n".join(cv["certificates"]))
    if cv.get("languages"):    st.subheader("Languages");      st.write(", ".join(cv["languages"]))
    if cv.get("achievements"): st.subheader("Achievements");   st.write("\n".join(cv["achievements"]))

    if st.session_state.newcv_report:
        r = st.session_state.newcv_report
        st.markdown("---"); st.subheader("ATS Match (quick)")
        st.metric("Score", f"{r['score']}%")
        a, b = st.columns(2)
        with a: st.markdown("**Matched**"); st.success(", ".join(r["matched"]) if r["matched"] else "-")
        with b: st.markdown("**Missing**"); st.warning(", ".join(r["missing"]) if r["missing"] else "-")

    if st.session_state.newcv_cover_letter:
        st.markdown("---")
        st.subheader("📮 Cover Letter (Preview)")
        st.text_area("Cover Letter", value=st.session_state.newcv_cover_letter, height=260)
        try:
            st.download_button("⬇️ Download Cover Letter (DOCX)",
                               data=export_docx_cover_letter(cv, st.session_state.newcv_cover_letter),
                               file_name="cover_letter.docx")
        except Exception as e:
            st.error(f"Cover Letter DOCX export failed: {e}")
        try:
            st.download_button("⬇️ Download Cover Letter (PDF)",
                               data=export_pdf_cover_letter(cv, st.session_state.newcv_cover_letter),
                               file_name="cover_letter.pdf")
        except Exception as e:
            st.error(f"Cover Letter PDF export failed: {e}")
