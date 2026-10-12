# pages/1_🧪_ATS_Scoring.py
import re
import json
import pandas as pd
import streamlit as st
from typing import List, Dict, Tuple
from core.parsing import parse_pdf_text
from ai.scorer import match_score
from ai.llm_gateway import chat_json
from ai.agent import CVAgent
from core.export import export_docx, export_pdf

st.set_page_config(page_title="ATS Scoring", layout="wide")
st.title("🧪 ATS Scoring")
st.caption("ارفع السير الذاتية (PDF) والصق الوصف الوظيفي (JD). بنعرض تحليل شامل + توليد CV محسّن يطابق الـ JD ويعدي ATS.")

# ================= Inputs =================
uploaded = st.file_uploader("Upload resumes (PDF)", type=["pdf"], accept_multiple_files=True)
jd_text = st.text_area("Paste JD", height=260, placeholder="Paste or drop the job description here…")
run = st.button("Analyze")

# ===== Hidden defaults (no UI) =====
use_llm      = True
anonymize    = False
w_sem, w_cov, w_ats = 55, 35, 10
req_weight, bonus_tools, min_len = 80, 2, 600

# ===== Regex helpers =====
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d \-()]{7,}\d)")
NAME_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{2,}$")

def deidentify(text: str) -> str:
    if not text: return ""
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    lines = text.splitlines()
    for i, ln in enumerate(lines[:6]):
        if NAME_LINE_RE.match(ln) and len(ln.split()) <= 5:
            lines[i] = "[NAME REDACTED]"
            break
    return "\n".join(lines)

# ===== JD parsing (LLM + Fallback) =====
JD_SYS = "You are a structured JD parser. Return compact JSON only."
def JD_USER(jd: str):
    return f"""
Extract key requirements from this JD. Return JSON with:
{{
  "role_title": "",
  "required": ["..."], "preferred": ["..."],
  "tools": ["..."], "cloud": ["..."], "certs": ["..."], "soft_skills": ["..."], "notes":""
}}
JD:
"""

def norm_list(lst):
    out, seen = [], set()
    for x in (lst or []):
        s = (x or "").strip()
        if not s: continue
        k = s.lower()
        if k not in seen:
            out.append(s); seen.add(k)
    return out

_TOOLS = [
    "python","pytorch","tensorflow","keras","scikit-learn","sklearn","xgboost","lightgbm","catboost",
    "langchain","transformers","hugging face","opencv","spark","hadoop","airflow","dbt",
    "tableau","power bi","lookml","superset",
    "docker","kubernetes","mlflow","ray","dask","fastapi","flask","streamlit",
    "faiss","weaviate","pinecone","milvus","lancedb","elasticsearch","neo4j",
]
_CLOUD = ["aws","azure","gcp","google cloud","amazon web services","sagemaker","vertex ai","databricks","snowflake","bigquery","redshift"]
_CERTS = ["aws certified","azure fundamentals","gcp professional","google professional data engineer","pmp","scrum","hcia-ai","coursera","udacity","deeplearning.ai"]
_SOFT  = ["communication","leadership","collaboration","problem solving","critical thinking","stakeholder","presentation","mentoring","facilitation"]
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{2,}")

def _split_lines(jd: str):
    return [l.strip("•-–— ").strip() for l in (jd or "").splitlines() if l.strip()]

def _pick_tokens(lines):
    text = "\n".join(lines).lower()
    out = set()
    for t in _TOOLS + _CLOUD + _SOFT:
        if t in text: out.add(t)
    for l in lines:
        for tok in _WORD.findall(l):
            out.add(tok.lower())
    return out

def keyword_sets_from_text(jd: str):
    lines = _split_lines(jd)
    buckets = {"required":[], "preferred":[], "tools":[], "cloud":[], "certs":[], "soft":[], "role_title":"", "notes":""}
    head = " ".join(lines[:3]).strip()
    m = re.search(r"(?i)\b(.*?)(?:\s*[-–—]\s*job description|\s*:\s*role|\s*position)\b", head)
    buckets["role_title"] = (m.group(1).strip() if m else (lines[0] if lines else "")).title()
    toks = list(_pick_tokens(lines))
    buckets["tools"] = [t for t in toks if t in _TOOLS]
    buckets["cloud"] = [t for t in toks if t in _CLOUD]
    buckets["soft"]  = [t for t in toks if t in _SOFT]
    text_low = "\n".join(lines).lower()
    req, pref = set(), set()
    for t in toks:
        if re.search(r"(?i)\b(must|required|mandatory)\b", text_low): req.add(t)
        if re.search(r"(?i)\b(preferred|plus|nice to have)\b", text_low): pref.add(t)
    noise = set(["and","with","for","the","that","this","role","responsibilities","requirements","experience","years"])
    req  = [x for x in req  if x not in noise and x not in buckets["tools"]+buckets["cloud"]+buckets["soft"]]
    pref = [x for x in pref if x not in noise and x not in buckets["tools"]+buckets["cloud"]+buckets["soft"]]
    buckets["required"]  = req[:25]
    buckets["preferred"] = pref[:20]
    found_certs = [c for c in _CERTS if c in text_low]
    buckets["certs"] = norm_list(found_certs)[:20]
    for k in ["required","preferred","tools","cloud","certs","soft"]:
        buckets[k] = norm_list(buckets[k])
    return buckets

def keyword_sets_from_llm(jd: str):
    data = chat_json(JD_SYS, JD_USER(jd))
    return {
        "role_title": (data.get("role_title") or "").strip(),
        "required":   norm_list(data.get("required")),
        "preferred":  norm_list(data.get("preferred")),
        "tools":      norm_list(data.get("tools")),
        "cloud":      norm_list(data.get("cloud")),
        "certs":      norm_list(data.get("certs")),
        "soft":       norm_list(data.get("soft_skills")),
        "notes":      (data.get("notes") or "").strip()
    }

def safe_keyword_sets(jd: str):
    try:
        ks = keyword_sets_from_llm(jd)
        if sum(len(ks[k]) for k in ["required","preferred","tools","cloud","certs","soft"]) == 0:
            return keyword_sets_from_text(jd)
        return ks
    except Exception:
        return keyword_sets_from_text(jd)

# ===== Coverage / ATS / Readability =====
def contains_any(text: str, tokens: List[str]) -> Tuple[List[str], List[str]]:
    text_low = (text or "").lower()
    matched, missing = [], []
    for t in tokens or []:
        t_low = t.lower()
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(t_low)}(?![A-Za-z0-9])", text_low):
            matched.append(t)
        else:
            missing.append(t)
    return matched, missing

def coverage_score(cv_text: str, ks: Dict[str, List[str]], req_weight: int, bonus_tools: int) -> Tuple[float, Dict]:
    detail = {}
    total_pts = 0; got_pts = 0
    req = ks.get("required", [])
    total_pts += max(1, len(req)) * req_weight
    m, miss = contains_any(cv_text, req); got_pts += len(m) * req_weight
    detail["required"] = {"matched": m, "missing": miss}
    pref = ks.get("preferred", [])
    total_pts += max(1, len(pref)) * (100 - req_weight)
    m, miss = contains_any(cv_text, pref); got_pts += len(m) * (100 - req_weight)
    detail["preferred"] = {"matched": m, "missing": miss}
    for cat in ["tools","cloud","certs","soft"]:
        items = ks.get(cat, [])
        m, miss = contains_any(cv_text, items)
        detail[cat] = {"matched": m, "missing": miss}
        got_pts += len(m) * bonus_tools
        total_pts += len(items) * bonus_tools
    cov = 0 if total_pts <= 0 else (got_pts / total_pts) * 100
    return round(min(100.0, cov), 1), detail

def ats_compliance(cv_text: str, min_len: int = 600) -> Tuple[float, Dict[str, str]]:
    txt = cv_text or ""; notes = {}; score = 100.0
    if len(txt) < min_len:
        notes["length"] = f"Content seems short ({len(txt)} chars). PDF may be image-only or CV too brief."
        score -= 25
    needed = ["summary","experience","education","skills"]
    found = sum(1 for s in needed if re.search(rf"(?i)\b{s}\b", txt))
    if found < 3:
        notes["sections"] = "Missing common ATS sections (e.g., Summary/Experience/Education/Skills)."; score -= 20
    pipes = txt.count("|") + txt.count("│")
    if pipes > 12:
        notes["tables"] = "Contains many table-like separators (|). ATS may struggle."; score -= 15
    caps_words = sum(1 for w in re.findall(r"[A-Z]{3,}", txt))
    if caps_words > 80:
        notes["caps"] = "Excessive ALL CAPS may reduce readability."; score -= 10
    bullets = len(re.findall(r"(^- |\u2022|\*)", txt, flags=re.M))
    if bullets < 5:
        notes["bullets"] = "Few bullet points; use concise action bullets."; score -= 10
    return max(0.0, round(score, 1)), notes

def readability_score(text: str) -> float:
    if not text: return 0.0
    sents = re.split(r"[.!?]+", text); sents = [s.strip() for s in sents if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    def _sy(w):
        w = w.lower(); v = "aeiouy"; c=0; p=False
        for ch in w:
            iv = ch in v
            if iv and not p: c+=1
            p = iv
        if w.endswith("e") and c>1: c-=1
        return max(1,c)
    syll = sum(_sy(w) for w in words) or 1
    w = max(1, len(words)); s = max(1, len(sents))
    fre = 206.835 - (1.015*(w/s)) - (84.6*(syll/w))
    fre = max(-20, min(120, fre))
    return round((fre-(-20))/(120-(-20))*100.0, 1)

def combined_score(sem: float, cov: float, ats: float, w_sem: int, w_cov: int, w_ats: int) -> float:
    tot = max(1, w_sem + w_cov + w_ats)
    return round((sem*(w_sem/tot)) + (cov*(w_cov/tot)) + (ats*(w_ats/tot)), 1)

# ===== Basic profile extraction from CV text (for tailoring/export) =====
def extract_profile(text: str):
    name = None; phone = None; email = None; location = None; links=[]
    m = EMAIL_RE.search(text or ""); email = m.group(0) if m else "-"
    m = PHONE_RE.search(text or ""); phone = m.group(0) if m else "-"
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    for ln in lines[:10]:
        if NAME_LINE_RE.match(ln) and 1 <= len(ln.split()) <= 6:
            name = ln.strip(); break
    for ln in lines[:25]:
        if re.search(r"^[A-Z][A-Za-z .'\-]{2,},\s*[A-Z][A-Za-z .'\-]{2,}$", ln):
            location = ln; break
    for ln in lines[:40]:
        for u in re.findall(r"(https?://[^\s)]+)", ln):
            u = u.strip().rstrip(").,"); 
            if u not in links: links.append(u)
    return name or "Candidate Name", phone, email, location or "-", ", ".join(links) if links else "-"

# ================= RUN =================
if run:
    if not uploaded or not jd_text.strip():
        st.warning("Please upload at least one PDF and paste a JD.")
    else:
        ks = safe_keyword_sets(jd_text)

        st.markdown("## 🔎 JD Analysis")
        cols = st.columns(6)
        cols[0].markdown(f"**Role**: {ks.get('role_title') or '-'}")
        for i,(label,key) in enumerate([('Required','required'),('Preferred','preferred'),('Tools','tools'),('Cloud','cloud'),('Certs','certs')], start=1):
            vals = ks.get(key) or []
            cols[i].markdown(f"**{label}**: {len(vals)}")
        if ks.get("notes"): st.info(ks["notes"])

        rows, details = [], {}

        with st.spinner("Scoring resumes…"):
            for f in uploaded:
                raw = parse_pdf_text(f.read())
                text = deidentify(raw) if anonymize else raw

                sem_score, missing_kw, matched_kw = match_score(text, jd_text)
                cov_score, cov_detail = coverage_score(text, ks, req_weight=req_weight, bonus_tools=bonus_tools)
                ats_score, ats_notes  = ats_compliance(text, min_len=min_len)
                read = readability_score(text)
                final = combined_score(sem_score, cov_score, ats_score, w_sem, w_cov, w_ats)

                rows.append({
                    "file": f.name, "Final Score": final, "Semantic %": round(sem_score,1),
                    "Coverage %": cov_score, "ATS %": ats_score, "Readability %": read,
                    "Missing (count)": sum(len(v.get("missing", [])) for v in cov_detail.values()),
                    "Matched (count)": sum(len(v.get("matched", [])) for v in cov_detail.values()),
                })

                details[f.name] = {
                    "raw": text, "preview": text[:2000],
                    "cov_detail": cov_detail, "ats_notes": ats_notes,
                    "missing_kw": missing_kw, "matched_kw": matched_kw
                }

        st.markdown("## Results")
        df = pd.DataFrame(rows).sort_values(["Final Score","Coverage %","Semantic %"], ascending=[False, False, False])
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", data=csv, file_name="ats_results.csv")

        # ===== Per-CV Insights & AI Tailored CV =====
        st.markdown("---")
        st.subheader("Per-CV Insights")
        for r in df.to_dict(orient="records"):
            name = r["file"]
            info = details[name]
            with st.expander(f"📄 {name} — Final {r['Final Score']} | Sem {r['Semantic %']} | Cov {r['Coverage %']} | ATS {r['ATS %']}"):
                col1, col2 = st.columns([1,1])

                # Matched/Missing (left)
                with col1:
                    st.markdown("**Matched / Missing by Category**")
                    for cat in ["required","preferred","tools","cloud","certs","soft"]:
                        if cat in info["cov_detail"]:
                            m = info["cov_detail"][cat].get("matched", [])
                            miss = info["cov_detail"][cat].get("missing", [])
                            if not m and not miss:
                                continue
                            st.markdown(f"**{cat.title()}**")
                            if m: st.success(", ".join(m))
                            if miss: st.warning(", ".join(miss))
                    if info.get("ats_notes"):
                        st.markdown("**ATS Compliance Notes**")
                        for _, v in info["ats_notes"].items():
                            st.info(f"• {v}")

                # Preview + AI (right)
                with col2:
                    st.markdown("**CV Preview (first 2000 chars)**")
                    st.code(info["preview"] or "-", language="markdown")

                    st.markdown("**AI Suggestions**")
                    missing_required = info["cov_detail"].get("required", {}).get("missing", [])
                    if st.button(f"Generate Fixes for {name}", key=f"fix_{name}"):
                        # 1) Generate bullets that cover missing (Required/Tools/Cloud/Certs)
                        need_cover = (
                            missing_required +
                            info["cov_detail"].get("tools", {}).get("missing", []) +
                            info["cov_detail"].get("cloud", {}).get("missing", []) +
                            info["cov_detail"].get("certs", {}).get("missing", [])
                        )
                        try:
                            fixes = chat_json(
                                "You are a resume coach. Return compact JSON only.",
                                f"""
CV excerpt:
JD:
Missing keywords to include naturally: {need_cover[:15]}

Task:
- Propose 6-8 concise, action- and results-oriented bullets (<= 18 words).
- Include quantification where possible (%, time, volume).
- Each bullet should include ≥1 of the missing keywords without keyword stuffing.
Return:
{{"bullets": ["..."]}}
                                """.strip()
                            )
                            bullets = [b.strip().rstrip(".") for b in (fixes.get("bullets") or []) if b.strip()]
                        except Exception:
                            bullets = [f"Address missing keywords: {', '.join(need_cover[:8])}"]

                        # 2) Build a tailored CV using CVAgent (Summary) + merged skills
                        agent = CVAgent()
                        try:
                            # summary tuned to JD
                            tailored_summary = agent.generate_summary({}, jd_text) or ""
                        except Exception:
                            tailored_summary = "Results-driven professional aligning experience to the role requirements and organizational outcomes."

                        # basic profile
                        prof_name, prof_phone, prof_email, prof_loc, prof_links = extract_profile(info["raw"])

                        # gather skills from JD buckets + matched
                        jd_skills = norm_list(
                            ks.get("required", []) + ks.get("preferred", []) + ks.get("tools", []) +
                            ks.get("cloud", []) + ks.get("soft", [])
                        )[:22]

                        tailored_cv = {
                            "name": prof_name,
                            "title": ks.get("role_title") or "Target Role",
                            "location": prof_loc or "-",
                            "email": prof_email or "-",
                            "phone": prof_phone or "-",
                            "links": prof_links or "-",
                            "summary": tailored_summary,
                            "skills": jd_skills,
                            "experience": [{"bullets": bullets[:8]}],
                            "education": [],
                            "projects": [],
                            "certificates": ks.get("certs", [])[:6],
                        }

                        # show preview
                        st.markdown("**Tailored CV Preview**")
                        st.write(f"""
**{tailored_cv['name']}**  
{tailored_cv['title']}  
{tailored_cv['location']}  
📧 {tailored_cv['email']} | 📱 {tailored_cv['phone']}  
🔗 {tailored_cv['links']}
""")
                        st.subheader("Summary"); st.write(tailored_cv["summary"])
                        st.subheader("Skills (ATS-friendly)"); st.write(", ".join(tailored_cv["skills"]))
                        st.subheader("Experience — JD-tailored bullets"); st.write("\n".join([f"- {b}" for b in tailored_cv["experience"][0]["bullets"]]))

                        # downloads
                        try:
                            st.download_button("⬇️ Download Tailored DOCX", data=export_docx(tailored_cv),
                                               file_name=f"{name}_tailored.docx")
                        except Exception as e:
                            st.warning(f"DOCX export failed: {e}")
                        try:
                            st.download_button("⬇️ Download Tailored PDF", data=export_pdf(tailored_cv),
                                               file_name=f"{name}_tailored.pdf")
                        except Exception as e:
                            st.warning(f"PDF export failed: {e}")
