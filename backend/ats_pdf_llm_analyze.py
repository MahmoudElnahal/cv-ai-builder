# backend/ats_pdf_llm_analyze.py
import os
import io
import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

# -------- .env (optional) --------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- OpenAI (v1) --------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
DEFAULT_MODEL = (os.getenv("VAI_MODEL") or "gpt-4o-mini").strip() or "gpt-4o-mini"

# ===================== Shared utils =====================
def _clean_space(s: Any) -> str:
    s = "" if s is None else str(s)
    return re.sub(r"\s+", " ", s).strip()

def _pct(n: int) -> int:
    return max(0, min(100, int(n)))

def _bool(x) -> bool:
    return bool(str(x or "").strip())

# ===================== Improve API =====================
class Section(str, Enum):
    summary = "summary"
    education = "education"
    experience = "experience"
    skills = "skills"
    role_highlights = "role_highlights"
    projects = "projects"
    certifications = "certifications"
    languages = "languages"
    achievements = "achievements"
    contact = "contact"

class CVContext(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    location: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    links: Optional[str] = None
    skills: Optional[List[str]] = None

class ImproveRequest(BaseModel):
    section: Section = Field(..., description="Which section to improve")
    content: Any = Field(..., description="Text or list depending on section")
    # NOTE: no job_description here (per requirement)
    target_role: Optional[str] = Field(None, description="e.g., 'Data Analyst'")
    language: str = Field("en", description="Output language code, e.g., 'en' or 'ar'")
    style: str = Field("ats", description="ats|narrative|neutral")
    cv: Optional[CVContext] = None
    max_items: int = Field(8, ge=1, le=50)
    sort_bullets_by_impact: bool = True
    return_mode: str = Field("json", pattern="^(raw|json)$")
    group_skills: bool = False
    force_llm: bool = False

class ImproveResponse(BaseModel):
    section: Section
    improved: Any
    notes: Optional[str] = None
    matched_keywords: Optional[List[str]] = None
    missing_keywords: Optional[List[str]] = None
    model: str

# ===================== Analyze API =====================
class AnalyzeOut(BaseModel):
    ats_score: Dict[str, Any] = Field(
        ..., description='{"value":"68 /100","label":"Good, but needs improvement","subscores":{"Content":"64%","Format":"83%","Skills":"75%"}}'
    )
    content_quality_analysis: List[Dict[str, Any]]
    formatting_ats_compatibility: Dict[str, Any]
    required_sections_check: Dict[str, Any]
    skills_analysis: Dict[str, Any]
    skills_organization_recommendation: str
    keyword_optimization: str
    priority_improvements: List[Dict[str, str]]
    overall_assessment: str
    suggested_roles: List[str]

class FileMeta(BaseModel):
    mimetype: Optional[str] = None
    size_kb: Optional[float] = None
    layout_notes: Optional[str] = None

# ===================== LLM Gateway =====================
class LLMGateway:
    def __init__(self, api_key: Optional[str], model: str):
        self.model = model
        self.enabled = bool(api_key and OpenAI is not None)
        self.client = OpenAI(api_key=api_key) if self.enabled else None

    # ------- impact sorting helpers -------
    @staticmethod
    def _impact_score(line: str) -> int:
        if not line:
            return 0
        s = 0
        L = line.lower()
        for t in ["%", "$", "increase", "decrease", "reduced", "improved", "saved",
                  "growth", "accuracy", "efficiency", "time", "cost"]:
            if t in L:
                s += 2
        if any(ch.isdigit() for ch in line):
            s += 1
        for v in ["built", "designed", "led", "launched", "automated", "optimized",
                  "developed", "deployed", "delivered"]:
            if v in L:
                s += 1
        return s

    def _sort_by_impact(self, items: List[str]) -> List[str]:
        return sorted(items, key=self._impact_score, reverse=True)

    # ------- LLM one-shot -------
    def llm_text(self, prompt: str, temperature: float = 0.2) -> str:
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role":"system","content":"You are a precise, senior ATS resume analyst."},
                {"role":"user","content":prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

gateway = LLMGateway(OPENAI_API_KEY, DEFAULT_MODEL)

# ===================== Improve core =====================
def _base_directives(req_lang: str, style: str, target_role: Optional[str]) -> str:
    tone = "concise, metric-driven, ATS-friendly" if style == "ats" else (
        "professional narrative" if style == "narrative" else "professional"
    )
    return (
        f"You are an expert resume editor. Output in {req_lang}.\n"
        f"Tone: {tone}.\n"
        "Use action verbs, quantify impact (%, time, $), avoid fluff, no personal pronouns.\n"
        f"Target role: {target_role or '(unspecified)'}.\n"
        "No job description is provided; optimize using common expectations for the target role.\n"
        "Return plain text; one bullet per line when bullets are requested.\n"
    )

def _cv_context_block(cv: Optional[CVContext]) -> str:
    if not cv:
        return ""
    return (
        "CV context (may be partial):\n"
        f"- Name: {cv.name or '-'}\n"
        f"- Title: {cv.title or '-'}\n"
        f"- Location: {cv.location or '-'}\n"
        f"- Skills: {', '.join(cv.skills or [])}\n"
    )

def _prompt_improve(section: Section, content: Any, lang: str, style: str,
                    target_role: Optional[str], cap: int, cv: Optional[CVContext]) -> str:
    base = _base_directives(lang, style, target_role) + _cv_context_block(cv)
    if section == Section.summary:
        return (
            base +
            "Task: Rewrite a professional resume summary (3–4 lines max) for the target role.\n"
            f"Content to improve:\n{content}\n"
            "Return: a single paragraph."
        )
    if section == Section.education:
        return (
            base +
            "Task: Normalize Education entries (degree, major, university, year) compactly; "
            "append relevant coursework/awards only if impactful.\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list; one item per line."
        )
    if section == Section.experience:
        return (
            base +
            f"Task: Convert to STAR-style bullets (max {cap}). Each bullet: Action + Scope/Tech + Metric/Outcome. "
            "Prioritize target-role alignment. Remove duplicates.\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list only."
        )
    if section == Section.skills:
        return (
            base +
            f"Task: Produce a deduplicated, ATS-friendly skill list (max {cap}). "
            "Cover essentials for the target role; use common phrasing. No grouping in output.\n"
            f"Content to improve:\n{content}\n"
            "Return: comma-separated items."
        )
    if section == Section.role_highlights:
        return (
            base +
            f"Task: Create role-aligned highlight bullets (5–7 lines; cap {cap}). "
            "Each bullet = Action + Scope/Tool + Metric/Outcome. Avoid company names.\n"
            "Return: bullet list."
        )
    if section == Section.projects:
        return (
            base +
            f"Task: Normalize project entries (max {cap}). Each line: Project — Tech/Scope — Impact/Metric.\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list."
        )
    if section == Section.certifications:
        return (
            base +
            f"Task: Normalize certifications (max {cap}). Each line: Provider — Certificate (Year) | key topics.\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list."
        )
    if section == Section.languages:
        return (
            base +
            f"Task: Normalize languages (max {cap}). Each line: Language — Level (Native/Fluent/Advanced/Intermediate).\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list."
        )
    if section == Section.achievements:
        return (
            base +
            f"Task: Normalize achievements/awards (max {cap}); quantify where possible; "
            "pick items that best support the target role.\n"
            f"Content to improve:\n{content}\n"
            "Return: bullet list."
        )
    # contact
    return (
        base +
        "Task: Normalize contact header fields (Name, Title, Location, Email, Phone, Links). "
        "Fix capitalization/spacing. Only provide JSON object with these fields.\n"
        f"Content to improve:\n{content}\n"
        "Return strictly JSON with keys: name, title, location, email, phone, links."
    )

def _sanitize_list(text: str, cap: int) -> List[str]:
    items = [x.strip("•- \t") for x in str(text).split("\n") if str(x).strip()]
    return items[:cap]

def _dedup_list(items: List[str], cap: int) -> List[str]:
    out, seen = [], set()
    for it in items:
        key = (it or "").strip().lower()
        if not key:
            continue
        if key not in seen:
            out.append(it.strip())
            seen.add(key)
        if len(out) >= cap:
            break
    return out

def _postprocess(section: Section, text: str, cap: int,
                 sort_bullets: bool, return_mode: str, group_skills: bool,
                 sort_fn) -> Any:
    if section in (
        Section.experience, Section.education, Section.role_highlights,
        Section.projects, Section.certifications, Section.languages, Section.achievements
    ):
        items = _sanitize_list(text, cap)
        if sort_bullets and section in (Section.experience, Section.role_highlights, Section.achievements, Section.projects):
            items = sort_fn(items)
        return {"bullets": items} if return_mode == "json" else items

    if section == Section.skills:
        parts = [p.strip(" ,;|") for p in str(text).replace("\n", ",").split(",") if p.strip()]
        parts = _dedup_list(parts, cap)
        return {"skills": parts} if return_mode == "json" else parts

    if section == Section.contact:
        try:
            obj = json.loads(text)
            out = {
                "name": _clean_space(obj.get("name", "")),
                "title": _clean_space(obj.get("title", "")),
                "location": _clean_space(obj.get("location", "")),
                "email": _clean_space(obj.get("email", "")),
                "phone": _clean_space(obj.get("phone", "")),
                "links": _clean_space(obj.get("links", "")),
            }
            return out if return_mode == "json" else str(out)
        except Exception:
            return {"raw": text} if return_mode == "json" else text

    # summary/plain
    clean = " ".join(line.strip() for line in str(text).splitlines())
    return {"summary": clean} if return_mode == "json" else clean

def _fallback_improve(req: ImproveRequest) -> ImproveResponse:
    content = req.content or ""
    out: Any
    if req.section == Section.summary:
        base = str(content).strip()
        if len(base) < 40 and req.target_role:
            base = f"Results-driven {req.target_role} with hands-on experience and measurable impact."
        out = {"summary": base[:700]} if req.return_mode == "json" else base[:700]
        return ImproveResponse(section=req.section, improved=out, notes="Fallback: lightly cleaned.", model="fallback")

    if req.section in (
        Section.experience, Section.education, Section.role_highlights,
        Section.projects, Section.certifications, Section.languages, Section.achievements
    ):
        items = content if isinstance(content, list) else [str(content)]
        items = [i.strip("•- \t") for i in items if str(i).strip()]
        items = _dedup_list(items, req.max_items)
        if req.sort_bullets_by_impact and req.section in (Section.experience, Section.role_highlights, Section.achievements, Section.projects):
            items = gateway._sort_by_impact(items)
        out = {"bullets": items} if req.return_mode == "json" else items
        return ImproveResponse(section=req.section, improved=out, notes="Fallback: normalized list.", model="fallback")

    if req.section == Section.skills:
        skills = content if isinstance(content, list) else [
            s.strip() for s in str(content).replace("\n", ",").split(",")
        ]
        skills = [s for s in skills if s]
        skills = _dedup_list(skills, req.max_items)
        out = {"skills": skills} if req.return_mode == "json" else skills
        return ImproveResponse(section=req.section, improved=out, notes="Fallback: deduplicated.", model="fallback")

    if req.section == Section.contact:
        if isinstance(content, dict) and req.return_mode == "json":
            keys = ["name","title","location","email","phone","links"]
            out = {k: _clean_space(content.get(k,"")) for k in keys}
            return ImproveResponse(section=req.section, improved=out, notes="Fallback: normalized fields.", model="fallback")
        return ImproveResponse(section=req.section, improved=content, notes="Fallback: raw contact.", model="fallback")

def improve_core(req: ImproveRequest) -> ImproveResponse:
    if not gateway.enabled:
        if req.force_llm:
            raise HTTPException(status_code=503, detail="LLM disabled: set OPENAI_API_KEY and install openai.")
        return _fallback_improve(req)

    prompt = _prompt_improve(
        section=req.section, content=req.content, lang=req.language, style=req.style,
        target_role=req.target_role, cap=req.max_items, cv=req.cv
    )
    try:
        text = gateway.llm_text(prompt, temperature=0.2)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    improved = _postprocess(
        section=req.section, text=text, cap=req.max_items,
        sort_bullets=req.sort_bullets_by_impact,
        return_mode=req.return_mode, group_skills=req.group_skills,
        sort_fn=gateway._sort_by_impact
    )
    return ImproveResponse(
        section=req.section,
        improved=improved,
        notes="AI enhanced (role-only). Verify facts/metrics before submission.",
        model=gateway.model if gateway.enabled else "fallback"
    )

# ===================== ATS Analyze helpers =====================
_METRIC_PAT = re.compile(r"(\b\d{1,3}%|\b\d{1,3}\s*(?:days?|weeks?|months?|years?)\b|\b\d{1,3}(?:\.\d+)?\b)")

def _has_quant_metrics(text: str) -> bool:
    return bool(_METRIC_PAT.search(text or ""))

def _has_linkedin(text: str) -> bool:
    return "linkedin.com" in (text or "").lower()

def _detect_sections(text: str) -> Dict[str, bool]:
    t = (text or "").lower()
    def any_of(keys: List[str]) -> bool:
        return any(k in t for k in keys)
    return {
        "Contact_Information": any_of(["email", "@", "phone", "linkedin"]),
        "Professional_Experience": any_of(["experience", "work history", "employment"]),
        "Education": any_of(["education", "b.sc", "bachelor", "msc", "university", "faculty"]),
        "Skills_Section": any_of(["skills", "technologies", "tech stack"]),
        "Projects": any_of(["projects", "portfolio", "case study"]),
        "Summary": any_of(["summary", "objective", "profile"])
    }

def _grammar_warnings(text: str) -> int:
    cnt = 0
    cnt += text.count("  ")
    cnt += text.count("..")
    for ln in text.splitlines():
        w = ln.strip()
        if len(w) > 3 and w.isupper() and re.search(r"[A-Z]{4,}", w):
            cnt += 1
    return min(12, cnt)

def _score_content(cv_text: str) -> Tuple[int, List[Dict[str, Any]]]:
    bullets: List[Dict[str, Any]] = []
    score = 50
    if _has_quant_metrics(cv_text):
        score += 12
    else:
        bullets.append({
            "title":"Quantifiable Achievements",
            "desc":"Your CV lacks measurable results. Add specific numbers (%, time, $) to demonstrate impact (e.g., “Increased sales by 30%”)."
        })
    if re.search(r"\b(built|developed|led|launched|optimized|deployed|implemented|designed)\b", cv_text.lower()):
        score += 6
        bullets.append({"title":"Action Verbs Usage",
                        "desc":"Good use of strong action verbs (Developed/Implemented/Led)."})
    else:
        bullets.append({"title":"Action Verbs Usage",
                        "desc":"Use stronger action verbs (Built, Implemented, Optimized) to increase impact."})
    secs = _detect_sections(cv_text)
    if secs.get("Summary"): score += 4
    else:
        bullets.append({"title":"Professional Summary",
                        "desc":"Add a concise professional summary tailored to the role with 1–2 quantified achievements."})
    g = _grammar_warnings(cv_text)
    if g > 0:
        score -= min(12, g)
        bullets.append({"title":"Grammar & Spelling",
                        "desc":"We found potential inconsistencies—unify date formats, verb tenses, and punctuation."})
    return _pct(score), bullets

def _has_tables_or_columns(layout: Optional[str]) -> bool:
    if not layout:
        return False
    L = layout.lower()
    return any(k in L for k in ["table", "columns", "2-column", "textbox", "many blocks"])

def _score_format(cv_text: str, file_meta: Optional[FileMeta]) -> Tuple[int, Dict[str, Any]]:
    score = 85
    ats_parsing = 82
    layout_simple = True
    if file_meta and _has_tables_or_columns(file_meta.layout_notes):
        score -= 8
        ats_parsing -= 8
        layout_simple = False
    fmt = {
        "ATS Parsing Rate": f"{_pct(ats_parsing)}%",
        "File Format": "PDF; keep total size <2MB for broad ATS compatibility.",
        "Font & Readability": "Use clean, ATS-friendly fonts; avoid text inside images.",
        "Section Headers": "Clear, conventional headers are present" if "skills" in cv_text.lower() else "Ensure clear, conventional headers.",
        "Layout Structure": "Simple/Clean" if layout_simple else "Consider single-column (avoid tables/columns)."
    }
    return _pct(score), fmt

def _extract_skills_simple(text: str) -> List[str]:
    lex = [
        "python","pandas","numpy","sql","power bi","tableau","excel","statistics","eda",
        "matplotlib","seaborn","scikit-learn","tensorflow","pytorch","xgboost",
        "etl","dbt","airflow","bigquery","redshift","snowflake","dax","power query",
        "langchain","openai","llm","rag","faiss","chromadb","docker","git","jupyter",
        "aws","gcp","azure","spark","hadoop","kafka","bi","dashboard","cohort","a/b testing",
        "forecasting","time series","nlp","classification","segmentation",
        "api","fastapi","flask","streamlit","databricks","sql server","postgres","mysql",
        "php","laravel","vue.js","javascript","rest apis","photoshop","figma","ms office","social media marketing","kubernetes","terraform","ci/cd","scrum","agile","microservices","tdd","stripe","paymob","digitalocean","typescript","html","css"
    ]
    T = text.lower()
    out = []
    for k in lex:
        if k in T:
            out.append(k)
    # unique preserve order
    seen=set(); uniq=[]
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq

def _skills_block(cv_text: str, jd_text: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    cv_sk = set(_extract_skills_simple(cv_text))
    jd_sk = set(_extract_skills_simple(jd_text or "")) if jd_text else set()
    if not jd_sk:
        breadth = len(cv_sk)
        score = 50 + min(40, breadth)  # cap 90
        return _pct(score), {
            "matched": sorted(list(cv_sk)),
            "unmatched": [],
            "suggested": []
        }
    matched = sorted(list(cv_sk & jd_sk))
    unmatched = sorted(list(jd_sk - cv_sk))
    suggest_pool = ["ci/cd","terraform","microservices","scrum","agile","a/b testing","dbt","airflow"]
    suggested = [s for s in suggest_pool if s not in cv_sk and (not jd_sk or s in jd_sk)]
    score = int(round(100.0 * (len(matched) / max(1, len(jd_sk)))))
    return _pct(score), {
        "matched": matched,
        "unmatched": unmatched[:20],
        "suggested": suggested[:10]
    }

def _priority_list(cv_text: str, jd_text: Optional[str], wants: int = 10) -> List[Dict[str,str]]:
    out: List[Dict[str,str]] = []
    if not _has_quant_metrics(cv_text):
        out.append({"title":"Add Quantifiable Metrics (Critical)",
                    "desc":"Turn achievements into measurable results (%, time, $, error-rate)."})
    if not _has_linkedin(cv_text):
        out.append({"title":"Add LinkedIn Profile (High)",
                    "desc":"Include linkedin.com/in/yourprofile in the header."})
    out.append({"title":"Fix Grammar Issues (High)",
                "desc":"Normalize date formats, verb tenses, and punctuation across bullets."})
    out.append({"title":"Reorganize Skills (Medium)",
                "desc":"Group skills into Backend/Frontend/DevOps/Tools; remove unrelated items."})
    out.append({"title":"Optimize Layout for ATS (Low)",
                "desc":"Prefer single-column, avoid tables/columns and text boxes."})
    if jd_text:
        out.append({"title":"Mirror JD Terminology",
                    "desc":"Use exact phrases from the JD (e.g., “RESTful APIs”)."})
    # pad to 10
    while len(out) < wants:
        out.append({"title":"Strengthen Top Bullets",
                    "desc":"Move highest-impact, role-aligned bullets to the top with metrics."})
    return out[:wants]

def _suggest_roles(cv_text: str, jd_text: Optional[str], k: int = 10) -> List[str]:
    roles: List[str] = []
    def add(r):
        if r not in roles and len(roles) < k:
            roles.append(r)
    L = (jd_text or cv_text or "").lower()
    cv_sk = set(_extract_skills_simple(cv_text))
    if {"python","sql","power bi"} & cv_sk: add("Data Analyst")
    if {"python","sql","tableau"} & cv_sk: add("Business Intelligence Analyst")
    if {"scikit-learn","tensorflow","pytorch"} & cv_sk: add("Machine Learning Engineer")
    if {"langchain","llm","rag"} & cv_sk: add("Generative AI Engineer")
    if {"fastapi","docker"} & cv_sk: add("Data/AI Engineer")
    if "php" in L or "laravel" in L: add("Laravel Developer"); add("Full Stack Developer"); add("Backend Engineer")
    if "vue.js" in L or "javascript" in L: add("Vue.js Frontend Developer"); add("Web Application Developer")
    if not roles: add("Analytics Specialist")
    while len(roles) < k:
        for r in ["Software Engineer","API Developer","Cloud Solutions Developer","Reporting Analyst","Product Analyst"]:
            add(r)
            if len(roles) >= k: break
    return roles[:k]

def _ensure_list_of_dicts(lst, min_len=5, max_len=8) -> List[Dict[str, Any]]:
    out = []
    if isinstance(lst, list):
        for it in lst:
            if isinstance(it, dict):
                title = _clean_space(it.get("title") or it.get("heading") or "")
                desc  = _clean_space(it.get("desc") or it.get("description") or "")
                present = it.get("present", None)
                row = {"title": title or "—", "desc": desc or "—"}
                if present is not None:
                    row["present"] = bool(present)
                out.append(row)
    out = out[:max_len]
    while len(out) < max(1, min_len):
        out.append({"title": "—", "desc": "—"})
    return out

def _ensure_priority_items(lst, exactly=10) -> List[Dict[str, str]]:
    out = []
    if isinstance(lst, list):
        for it in lst:
            if isinstance(it, dict):
                t = _clean_space(it.get("title", ""))
                d = _clean_space(it.get("desc", ""))
                if t or d:
                    out.append({"title": t or "—", "desc": d or "—"})
            elif isinstance(it, str):
                out.append({"title": it[:80] or "Improvement", "desc": it})
    out = out[:exactly]
    while len(out) < exactly:
        out.append({"title": "Improvement", "desc": "Add a concrete, metric-driven enhancement."})
    return out

def _ensure_string_list(lst, exactly: Optional[int] = None, fallback: Optional[List[str]] = None) -> List[str]:
    out = []
    if isinstance(lst, list):
        for it in lst:
            if isinstance(it, str) and it.strip():
                out.append(_clean_space(it))
    if exactly is not None:
        out = out[:exactly]
        if fallback:
            i = 0
            while len(out) < exactly and i < len(fallback):
                if fallback[i] not in out:
                    out.append(fallback[i])
                i += 1
        while len(out) < exactly:
            out.append("Role")
    return out

def _ensure_skills_block(obj: Any) -> Dict[str, Any]:
    matched   = obj.get("matched")   if isinstance(obj, dict) else []
    unmatched = obj.get("unmatched") if isinstance(obj, dict) else []
    suggested = obj.get("suggested") if isinstance(obj, dict) else []
    return {
        "matched":   _ensure_string_list(matched) or [],
        "unmatched": _ensure_string_list(unmatched) or [],
        "suggested": _ensure_string_list(suggested) or [],
    }

def _coerce_to_analyze_schema(raw: Dict[str, Any],
                              heuristic_scores: Dict[str, int],
                              roles_fallback: List[str]) -> Dict[str, Any]:
    ats_out = {
        "value": f'{heuristic_scores["total"]} /100',
        "label": "Excellent" if heuristic_scores["total"] >= 85 else "Good, but needs improvement",
        "subscores": {
            "Content": f'{heuristic_scores["content"]}%',
            "Format":  f'{heuristic_scores["format"]}%',
            "Skills":  f'{heuristic_scores["skills"]}%'
        }
    }
    cqa = _ensure_list_of_dicts(raw.get("content_quality_analysis"), min_len=5, max_len=8)
    fmt = raw.get("formatting_ats_compatibility") if isinstance(raw, dict) else {}
    fmt_out = {
        "ATS Parsing Rate": _clean_space(fmt.get("ATS Parsing Rate", "78%")),
        "File Format": _clean_space(fmt.get("File Format", "PDF; keep total size <2MB.")),
        "Font & Readability": _clean_space(fmt.get("Font & Readability", "Use ATS-friendly fonts.")),
        "Section Headers": _clean_space(fmt.get("Section Headers", "Clear, conventional headers.")),
        "Layout Structure": _clean_space(fmt.get("Layout Structure", "Prefer single-column layout."))
    }
    req = raw.get("required_sections_check") if isinstance(raw, dict) else {}
    req_out = {
        "Contact Information": _clean_space(req.get("Contact Information", "Phone and email present; add LinkedIn.")),
        "Professional Experience": _clean_space(req.get("Professional Experience", "Ensure titles, companies, dates, bullets.")),
        "Education": _clean_space(req.get("Education", "Include degree, institution, dates.")),
        "Skills Section": _clean_space(req.get("Skills Section", "Group skills by category.")),
        "Projects/Portfolio": _clean_space(req.get("Projects/Portfolio", "Add 2–3 projects with links."))
    }
    skills_block = _ensure_skills_block(raw.get("skills_analysis") or {})
    sor = _clean_space(raw.get("skills_organization_recommendation", "")) or \
          "Organize skills into distinct groups (Backend/Frontend/DevOps/Tools)."
    kw = _clean_space(raw.get("keyword_optimization", "")) or \
         "Mirror exact phrases from the JD in Summary/Skills/Experience."
    pri = _ensure_priority_items(raw.get("priority_improvements"), exactly=10)
    overall = _clean_space(raw.get("overall_assessment", "")) or \
              "Solid base; focus on quantified outcomes, JD-aligned phrasing, and clean layout to raise ATS score."
    roles = _ensure_string_list(raw.get("suggested_roles"), exactly=10, fallback=roles_fallback)
    return {
        "ats_score": ats_out,
        "content_quality_analysis": cqa,
        "formatting_ats_compatibility": fmt_out,
        "required_sections_check": req_out,
        "skills_analysis": skills_block,
        "skills_organization_recommendation": sor,
        "keyword_optimization": kw,
        "priority_improvements": pri,
        "overall_assessment": overall,
        "suggested_roles": roles
    }

# ===================== PDF extraction =====================
def _extract_text_from_pdf_bytes(data: bytes) -> tuple[str, str]:
    text, layout = "", []
    # 1) PyMuPDF
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") or ""
                blocks = page.get_text("blocks")
                if blocks and len(blocks) > 20:
                    layout.append("many blocks (possible multi-column)")
    except Exception:
        pass
    # 2) pdfplumber
    if not text.strip():
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text() or ""
                    text += txt + "\n"
                    if p.extract_tables():
                        layout.append("tables detected")
        except Exception:
            pass
    # 3) pypdf
    if not text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            pass
    return text or "", ("; ".join(sorted(set(layout))) if layout else "")

# ===================== FastAPI App =====================
app = FastAPI(title="CV ATS Agent", version="1.0", openapi_version="3.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_enabled": gateway.enabled,
        "model": gateway.model if gateway.enabled else "fallback",
    }

# ---------- POST /v1/improve ----------
@app.post("/v1/improve", response_model=ImproveResponse)
def improve(req: ImproveRequest):
    try:
        return improve_core(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- POST /v1/analyze ----------
@app.post("/v1/analyze", response_model=AnalyzeOut)
async def analyze_pdf(
    file: UploadFile = File(..., description="CV PDF file"),
    job_description: Optional[str] = Form(None, description="(Optional) JD text"),
    language: str = Form("en"),
):
    """
    Upload a PDF CV → get full ATS analysis (score, sections, skills, improvements, roles).
    - Accepts optional job_description (plain string).
    - Uses LLM if available with robust coercion to schema; otherwise heuristic fallback.
    """
    if (file.content_type or "").lower() not in {"application/pdf", "pdf"}:
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    cv_text, layout_notes = _extract_text_from_pdf_bytes(data)
    if not cv_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

    # 1) Heuristic subscores
    content_score, content_bullets = _score_content(cv_text)
    file_meta = FileMeta(mimetype=file.content_type or "application/pdf",
                         size_kb=round(len(data)/1024.0,2),
                         layout_notes=layout_notes or None)
    format_score, fmt_block = _score_format(cv_text, file_meta)
    skills_score, skills_block = _skills_block(cv_text, job_description)

    total = _pct(round(content_score*0.45 + format_score*0.25 + skills_score*0.30))
    label = "Excellent" if total >= 85 else "Good, but needs improvement" if total >= 60 else "Needs significant improvement"

    # 2) Required sections (human-friendly text like المثال)
    sec_flags = _detect_sections(cv_text)
    req_check = {
        "Contact Information": "Phone and email present. Add LinkedIn URL if missing." if sec_flags["Contact_Information"] else "Missing or incomplete — include phone, email, and LinkedIn.",
        "Professional Experience": "Experience section complete with titles/companies/dates/bullets." if sec_flags["Professional_Experience"] else "Add roles, companies, dates, and impact bullets.",
        "Education": "Education section present (degree, institution, dates)." if sec_flags["Education"] else "Add degree, institution, and graduation year.",
        "Skills Section": "Skills are listed; consider better organization." if sec_flags["Skills_Section"] else "Add a clear Skills section.",
        "Projects/Portfolio": "Projects section included; add live links if possible." if sec_flags["Projects"] else "Add 2–3 projects with links (GitHub/Live)."
    }

    content_quality = content_bullets
    kw_opt = (
        "Mirror exact JD phrases in Summary/Skills/Experience (e.g., “RESTful APIs”, “CI/CD”, “Microservices”)."
        if job_description else
        "Include common role keywords relevant to your target (e.g., SQL, Power BI, EDA, A/B testing, ETL)."
    )
    pri = _priority_list(cv_text, job_description, wants=10)
    roles_fallback = _suggest_roles(cv_text, job_description, k=10)

    # 3) Try LLM to write the narrative blocks in chosen language; coerce to schema
    if gateway.enabled:
        try:
            prompt = f"""
You are an ATS resume evaluator. Return **valid JSON only** with keys:
- "content_quality_analysis": array of 5–8 objects {{title, desc}}
- "formatting_ats_compatibility": object with keys:
  "ATS Parsing Rate","File Format","Font & Readability","Section Headers","Layout Structure"
- "required_sections_check": object with human-friendly sentences for:
  "Contact Information","Professional Experience","Education","Skills Section","Projects/Portfolio"
- "skills_analysis": object {{matched:[], unmatched:[], suggested:[]}}
- "skills_organization_recommendation": string
- "keyword_optimization": string
- "priority_improvements": array of 10 objects {{title, desc}}
- "overall_assessment": string
- "suggested_roles": array of 10 strings

Write in language: {language}.
CV (excerpt):
{cv_text[:3000]}

Job Description (optional):
{job_description or "(none)"}

Guidance:
- Be specific and helpful.
- Avoid company names; emphasize outcomes and ATS phrasing.
- If unsure, provide sensible defaults.
"""
            raw = gateway.llm_text(prompt, temperature=0.25)
            try:
                obj = json.loads(raw)
            except Exception:
                s = raw.find("{"); e = raw.rfind("}")
                if s != -1 and e != -1 and e > s:
                    obj = json.loads(raw[s:e+1])
                else:
                    obj = {}

            heuristic = {"total": total, "content": content_score, "format": format_score, "skills": skills_score}
            coerced = _coerce_to_analyze_schema(obj, heuristic, roles_fallback)

            try:
                return AnalyzeOut(**coerced)
            except ValidationError as ve:
                raise HTTPException(status_code=502, detail=f"Response validation failed: {ve.errors()}")

        except HTTPException:
            raise
        except Exception as e:
            # fall back to heuristic below
            pass

    # 4) Heuristic fallback (no LLM or LLM failed) — ensure schema
    fallback_obj = {
        "ats_score": {
            "value": f"{total} /100",
            "label": label,
            "subscores": {"Content": f"{content_score}%", "Format": f"{format_score}%", "Skills": f"{skills_score}%"}
        },
        "content_quality_analysis": content_quality,
        "formatting_ats_compatibility": fmt_block,
        "required_sections_check": req_check,
        "skills_analysis": {
            "matched": skills_block.get("matched", []),
            "unmatched": skills_block.get("unmatched", []),
            "suggested": skills_block.get("suggested", []),
        },
        "skills_organization_recommendation": "Organize skills into distinct groups (Backend/Frontend/DevOps/Tools); remove unrelated items.",
        "keyword_optimization": kw_opt,
        "priority_improvements": pri,
        "overall_assessment": (
            "Your CV is generally well-structured. Focus on adding quantified results, aligning keywords with the JD, "
            "and using a clean single-column layout to improve ATS parsing and recruiter impact."
        ),
        "suggested_roles": roles_fallback
    }
    return AnalyzeOut(**fallback_obj)
