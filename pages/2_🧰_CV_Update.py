# pages/2_🧰_CV_Update.py
import re, json, streamlit as st
from typing import List, Dict, Tuple

from core.parsing import parse_pdf_text
from core.export import (
    export_docx, export_pdf,
    export_docx_cover_letter, export_pdf_cover_letter
)
from ai.agent import CVAgent            # generate_summary(seed_cv, jd), generate_bullets(role, jd)
from ai.scorer import match_score       # (score_pct, missing_keywords, matched_keywords)

st.set_page_config(page_title="CV Update", layout="wide")
st.title("🧰 CV Update (Upload CV + JD)")
st.caption("Upload your CV (PDF) + JD → We’ll generate an ATS-ready tailored CV (clean sections), and optionally a Cover Letter.")

# =========================== Inputs ===========================
uploaded_cv = st.file_uploader("Upload your current CV (PDF)", type=["pdf"])
jd_text     = st.text_area("Job Description (JD)", height=260, placeholder="Paste the full JD here…")
col_top1, col_top2 = st.columns([1,1])
with col_top1:
    run_cv = st.button("⚡ Generate Tailored CV")
with col_top2:
    want_cl = st.checkbox("Also generate a Cover Letter", value=False)

# ====================== Regex & helpers =======================
EMAIL_RE     = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE     = re.compile(r"(\+?\d[\d \-()]{7,}\d)")
NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{2,}$")
LINKEDIN_RE  = re.compile(r"(https?://(www\.)?linkedin\.com/[A-Za-z0-9_\-/]+)", re.I)
LOCATION_RE  = re.compile(r"(?i)\b([A-Z][A-Za-z .'-]{2,}),\s*([A-Z][A-Za-z .'-]{2,})\b")
URL_RE       = re.compile(r"(https?://[^\s)]+)")
TOKEN_RE     = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}")

STOP = {
    "and","or","with","for","the","a","an","in","to","of","on","by","as","at","is","are","be",
    "job","description","position","overview","team","candidate","will","responsible","this","role",
    "experience","skills","strong","advanced","basic","both","company","degree","bachelor","master"
}

def s(x) -> str:
    if x is None: return ""
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False, separators=(", ", ": "))
    return str(x)

def one_line(x: str) -> str:
    return s(x).replace("\r", " ").replace("\n", " ").strip()

def lines_cap(items: List[str], cap: int = 12) -> List[str]:
    out = []
    for it in items or []:
        t = one_line(it)
        if t:
            out.append(t[:300])
            if len(out) >= cap: break
    return out

# ===================== CV parsing from PDF ====================
def extract_contacts(text: str) -> Tuple[str,str,str,str,str,str]:
    name = phone = email = location = linkedin = links = "-"
    m = EMAIL_RE.search(text or "");  email = m.group(0) if m else "-"
    m = PHONE_RE.search(text or "");  phone = m.group(0) if m else "-"
    L = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for ln in L[:10]:
        if NAME_LINE_RE.match(ln) and 1 <= len(ln.split()) <= 6: name = ln; break
    for ln in L[:40]:
        m = LOCATION_RE.search(ln)
        if m: location = f"{m.group(1)}, {m.group(2)}"; break
    for ln in L[:80]:
        m = LINKEDIN_RE.search(ln)
        if m: linkedin = m.group(1); break
    all_links = URL_RE.findall(text or "")
    links = ", ".join(sorted(set(u for u in all_links if "linkedin.com" not in u))) or "-"
    return name or "Candidate Name", phone or "-", email or "-", location or "-", linkedin or "-", links

def section_block(label: str, text: str) -> List[str]:
    L = [l.strip() for l in (text or "").splitlines()]
    items, cap = [], False
    for ln in L:
        if re.match(rf"(?i)^\s*{label}\b", ln): cap = True; continue
        if cap and re.match(r"^[A-Z][A-Za-z \-/]{2,}$", ln): break
        if cap and ln: items.append(ln)
    return items

def extract_education(txt: str, limit=6) -> List[str]:
    # 1) Section if exists
    edu = section_block("education", txt)
    # 2) Heuristics
    if not edu:
        edu = [l for l in txt.splitlines() if re.search(
            r"(?i)(B\.?Sc|M\.?Sc|Bachelor|Master|Ph\.?D|Diploma|University|College|Institute|Faculty)", l)]
    edu = [one_line(x) for x in edu if len(one_line(x)) > 3]
    # Normalize into “Degree – University (Year)” if possible
    cleaned = []
    for e in edu:
        e = re.sub(r"\s{2,}", " ", e)
        e = e.strip("•*-–— ").strip()
        cleaned.append(e)
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned[:limit] if cleaned else ["B.Sc. in Artificial Intelligence – Kafr El-Sheikh University (2024)"]

def extract_projects(txt: str, limit=8) -> List[str]:
    proj = section_block("projects", txt)
    if not proj:
        proj = [l for l in txt.splitlines() if re.search(
            r"(?i)\b(project|dashboard|pipeline|model|system|platform|POC|prototype|segmentation|detection)\b", l)]
    # Merge short fragments / normalize
    merged = []
    buff = ""
    for l in proj:
        L = one_line(l)
        if not L: continue
        if len(L) < 30:
            buff = (buff + " " + L).strip()
        else:
            if buff:
                merged.append(buff); buff = ""
            merged.append(L)
    if buff: merged.append(buff)
    # Add tool/impact hints if missing
    cleaned = []
    for p in merged:
        if "→" not in p and "-" in p:
            # Try: "Title – Org → impact"
            title = p.split("-")[0].strip()
            cleaned.append(f"{title} → Delivered measurable impact with ML/BI stack.")
        else:
            cleaned.append(p)
    # Deduplicate
    cleaned = list(dict.fromkeys([re.sub(r"\s{2,}", " ", x).strip("•*-–— ").strip() for x in cleaned]))
    return cleaned[:limit]

def extract_certificates(txt: str, limit=8) -> List[str]:
    # Section + heuristic keywords
    cert = section_block("cert", txt) + section_block("certificates", txt) + section_block("training", txt)
    cert += [l for l in txt.splitlines() if re.search(
        r"(?i)(HCIA|Coursera|Udacity|Udemy|DeepLearning\.ai|ITI|Samsung|InnovaEgypt|PMP|Scrum|AWS|Azure|GCP|Google|Microsoft)", l)]
    cleaned = []
    for c in cert:
        c = one_line(c)
        c = re.sub(r"\s{2,}", " ", c).strip("•*-–— ").strip()
        if 3 <= len(c) <= 160:
            cleaned.append(c)
    # Normalize year suffix if obvious ranges exist
    cleaned = list(dict.fromkeys(cleaned))
    return cleaned[:limit] if cleaned else [
        "Huawei HCIA-AI (2022)",
        "Coursera Machine Learning Specialization (2020)",
        "ITI – Data & AI Summer Training (2024)"
    ]

def extract_cv_skills(txt: str, limit=30) -> List[str]:
    skl = section_block("skills", txt)
    skills = []
    for ln in skl:
        parts = [p.strip() for p in re.split(r"[•\-\u2022,;/|]", ln) if p.strip()]
        skills.extend(parts)
    out, seen = [], set()
    for s1 in skills:
        key = s1.lower()
        if key not in seen and 2 <= len(s1) <= 40:
            out.append(one_line(s1)); seen.add(key)
    return out[:limit]

def summary_seed(txt: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", txt or "") if p.strip()]
    for p in paras:
        if 120 <= len(p) <= 700: return p[:700]
    return (paras[0][:700] if paras else "")

# ===================== JD → skills merge ======================
def normalize_skill(s: str) -> str:
    if not s: return ""
    t = s.replace("_"," ").replace("-"," ").strip()
    acr = {"sql","etl","nlp","rag","llm","bi","kpi","kpis","mlops"}
    if t.lower() in acr: return t.upper()
    return t.title()

def tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 2 and t.lower() not in STOP]

def keyword_sets_from_text(jd: str) -> Dict[str, List[str]]:
    L = [l.strip("•*-–— ").strip() for l in (jd or "").splitlines() if l.strip()]
    role = (L[0] if L else "").title()
    toks = tokens(jd)
    tools = [t for t in toks if t in {
        "python","pandas","numpy","pytorch","tensorflow","keras","scikit-learn","xgboost","sql",
        "dbt","airflow","spark","hadoop","power","tableau","excel","langchain","transformers",
        "docker","kubernetes","mlflow","fastapi","flask","streamlit","faiss","pinecone","weaviate","snowflake","bigquery"
    }]
    cloud = [t for t in toks if t in {"aws","azure","gcp","sagemaker","vertex","databricks","redshift"}]
    soft  = [t for t in toks if t in {"communication","leadership","collaboration","problem","stakeholder","mentoring","presentation"}]
    text_low = "\n".join(L).lower()
    req, pref = set(), set()
    if re.search(r"(?i)\b(must|required|mandatory)\b", text_low): req = set(toks)
    if re.search(r"(?i)\b(preferred|plus|nice to have)\b", text_low): pref = set(toks)
    noise = set(STOP) | set(tools) | set(cloud) | set(soft)
    req  = [x for x in req  if x not in noise][:25]
    pref = [x for x in pref if x not in noise][:20]
    return {"role_title": role, "required": req, "preferred": pref, "tools": tools, "cloud": cloud, "soft": soft}

def merge_skills(cv_skills: List[str], ks: Dict[str,List[str]], cap=22) -> List[str]:
    base = {x.lower(): x for x in (cv_skills or [])}
    inject = ks.get("required", []) + ks.get("preferred", []) + ks.get("tools", []) + ks.get("cloud", []) + ks.get("soft", [])
    for k in inject:
        if k not in base and k not in STOP:
            base[k] = k
    arr = [normalize_skill(v) for v in base.values() if v.lower() not in STOP]
    bad = {"data","analysis","experience","strong","skills","role","job","candidate","team","overview","position"}
    arr = [x for x in arr if x and x.lower() not in bad]
    return arr[:cap]

# =================== Local bullet fallback ====================
ACTIONS = ["Built","Designed","Implemented","Optimized","Deployed","Automated","Integrated","Scaled","Fine-tuned","Led","Improved","Reduced","Increased","Delivered"]
def local_bullets(missing: List[str], limit=6) -> List[str]:
    out, i = [], 0
    for kw in missing:
        kw = one_line(kw)
        if not kw: continue
        out.append(f"{ACTIONS[i % len(ACTIONS)]} solutions using {kw}, achieving measurable improvements (≥20%)")
        i += 1
        if len(out) >= limit: break
    return [b[:160].rstrip(".") for b in out]

# ============================== RUN ===========================
if run_cv:
    if not uploaded_cv or not jd_text.strip():
        st.warning("Please upload a CV and paste a JD.")
    else:
        # Parse CV
        try:
            raw = parse_pdf_text(uploaded_cv.read())
        except Exception as e:
            raw = ""
            st.error(f"Failed to read CV: {e}")

        name, phone, email, location, linkedin, other_links = extract_contacts(raw)
        edu       = extract_education(raw, limit=6)
        projects  = extract_projects(raw, limit=8)
        cv_skills = extract_cv_skills(raw, limit=30)
        seed_sum  = summary_seed(raw)

        # JD sets (text parsing سريعة وثابتة)
        ks = keyword_sets_from_text(jd_text)
        role_title = ks.get("role_title") or "Target Role"

        # AI Summary + Bullets
        agent = CVAgent()
        try:
            seed_cv = {
                "name": name, "title": role_title,
                "skills": cv_skills[:10], "education": edu[:3], "projects": projects[:5],
                "experience": [{"bullets": []}], "summary": seed_sum[:600]
            }
            ai_summary = agent.generate_summary(seed_cv, jd_text) or seed_sum
        except Exception:
            ai_summary = seed_sum or f"Results-driven {role_title} focused on measurable impact."

        try:
            ai_bullets = agent.generate_bullets(role_title, jd_text) or []
        except Exception:
            ai_bullets = []

        # Fallback bullets guided by JD if empty
        merged_sk = merge_skills(cv_skills, ks, cap=22)
        tools     = sorted(list(dict.fromkeys([normalize_skill(t) for t in (ks.get("tools", []) + ks.get("cloud", []))])))
        try:
            score_text = "\n".join([one_line(ai_summary), ", ".join(merged_sk), ", ".join(tools)])
            score, missing, matched = match_score(score_text, jd_text)
        except Exception:
            score, missing, matched = 0.0, [], []
        if not ai_bullets:
            ai_bullets = local_bullets(ks.get("required", [])[:4] + ks.get("tools", [])[:2] + ks.get("cloud", [])[:2], limit=6)
        ai_bullets = lines_cap([b.strip().rstrip(".") for b in ai_bullets], cap=8)

        # Certificates (من CV + تنظيم)
        certificates = extract_certificates(raw, limit=8)

        # ===== Final CV dict (ordered, export-safe) =====
        final_cv = {
            "name": name or "Candidate Name",
            "title": role_title,
            "location": location or "-",
            "email": email or "-",
            "phone": phone or "-",
            "linkedin": linkedin or "-",
            "links": other_links or "-",
            "summary": one_line(ai_summary),
            "skills": merged_sk,
            "tools": tools,
            "experience": [{"bullets": ai_bullets}],
            "projects": projects,
            "education": edu,
            "certificates": certificates,
            "languages": ["Arabic – Native", "English – Good (Writing & Speaking)"],
            "achievements": [
                "Top 4 Trainee – Samsung Innovation Campus (2023)",
                "Innovation Program – InnovaEgypt (2023)"
            ],
        }

        # ================== UI Preview ==================
        st.markdown("## CV Preview")
        st.write(f"""**{final_cv['name']}**  
{final_cv['title']}  
{final_cv['location']}  
📧 {final_cv['email']} | 📱 {final_cv['phone']}  
🔗 {final_cv['linkedin']} | {final_cv['links']}
""")
        st.markdown("---")
        st.subheader("Executive Summary");    st.write(final_cv["summary"] or "-")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Core Competencies"); st.write(", ".join(final_cv["skills"]) or "-")
        with c2:
            st.subheader("Tools & Tech");      st.write(", ".join(final_cv["tools"]) or "-")
        st.subheader("Experience — JD-tailored Results")
        st.write("\n".join(f"- {b}" for b in (final_cv["experience"][0]["bullets"] or [])) or "-")
        if final_cv["projects"]:     st.subheader("Projects");      st.write("\n".join(f"- {p}" for p in final_cv["projects"]))
        if final_cv["education"]:    st.subheader("Education");     st.write("\n".join(final_cv["education"]))
        if final_cv["certificates"]: st.subheader("Certifications");st.write("\n".join(final_cv["certificates"]))
        if final_cv["languages"]:    st.subheader("Languages");     st.write(", ".join(final_cv["languages"]))
        if final_cv["achievements"]: st.subheader("Achievements");  st.write("\n".join(final_cv["achievements"]))

        # ATS report (مختصر)
        st.markdown("---")
        st.subheader("ATS Match (quick)")
        st.metric("Score", f"{round(score,1)}%")
        a, b = st.columns(2)
        with a: st.markdown("**Matched**"); st.success(", ".join(matched) if matched else "-")
        with b: st.markdown("**Missing**"); st.warning(", ".join(missing) if missing else "-")

        # ================== Downloads ==================
        try:
            st.download_button("⬇️ Download CV (DOCX)",  data=export_docx(final_cv), file_name="tailored_cv.docx")
        except Exception as e:
            st.error(f"DOCX export failed: {e}")
        try:
            st.download_button("⬇️ Download CV (PDF)",    data=export_pdf(final_cv),  file_name="tailored_cv.pdf")
        except Exception as e:
            st.error(f"PDF export failed: {e}")

        # ============== Optional: Cover Letter ==============
        if want_cl:
            st.markdown("---")
            st.subheader("📮 Cover Letter (Preview)")
            agent = CVAgent()
            try:
                letter = agent.generate_cover_letter(final_cv, jd_text)  # لو عندك الدالة دي
            except Exception:
                # لو مش موجودة في الـ Agent نستخدم برومبت خفيف:
                try:
                    from ai.llm_gateway import chat_json
                    pj = ", ".join(final_cv["projects"][:2]) if final_cv.get("projects") else ""
                    cl_json = chat_json(
                        "You are a career coach. Return JSON with 'letter' only.",
                        f"""
Create a concise, professional cover letter (220–320 words) tailored to the JD.
Keep it ATS-friendly, quant-driven, and aligned with the candidate profile.

Candidate:
Name: {final_cv['name']}
Title: {final_cv['title']}
Location: {final_cv['location']}
Email: {final_cv['email']} | Phone: {final_cv['phone']} | LinkedIn: {final_cv['linkedin']}
Key Skills: {", ".join(final_cv['skills'][:12])}
Tools: {", ".join(final_cv['tools'][:8])}
Recent Projects: {pj}

JD:

Return:
{{"letter": "..."}}"""
                    )
                    letter = one_line(cl_json.get("letter")) or ""
                except Exception:
                    letter = (
                        f"Dear Hiring Manager,\n\n"
                        f"I am excited to apply for the {final_cv['title']} position. I have delivered measurable outcomes such as "
                        f"improving reporting efficiency by 30% and deploying production-grade analytics workflows. "
                        f"My toolkit includes {', '.join(final_cv['tools'][:6])}, and I frequently leverage "
                        f"{', '.join(final_cv['skills'][:8])} to build scalable solutions.\n\n"
                        f"Sincerely,\n{final_cv['name']}"
                    )

            st.text_area("Cover Letter", value=letter, height=280)

            # Downloads for Cover Letter
            try:
                st.download_button("⬇️ Download Cover Letter (DOCX)",
                                   data=export_docx_cover_letter(final_cv, letter),
                                   file_name="cover_letter.docx")
            except Exception as e:
                st.error(f"Cover Letter DOCX export failed: {e}")
            try:
                st.download_button("⬇️ Download Cover Letter (PDF)",
                                   data=export_pdf_cover_letter(final_cv, letter),
                                   file_name="cover_letter.pdf")
            except Exception as e:
                st.error(f"Cover Letter PDF export failed: {e}")
