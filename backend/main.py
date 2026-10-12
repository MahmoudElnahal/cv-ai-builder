# backend/main.py
import os
import re
import json
from enum import Enum
from typing import Any, List, Optional, Union, Dict, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# OPTIONAL: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI v1 (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("VAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


# ----------------- Schemas -----------------
class Section(str, Enum):
    summary = "summary"
    education = "education"
    experience = "experience"
    skills = "skills"
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


class FullCVIn(BaseModel):
    contact: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    experience: Optional[List[str]] = None
    education: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    projects: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    achievements: Optional[List[str]] = None


class ExtractResponse(BaseModel):
    raw_text: str


# ImproveAllRequest - now using Form parameters directly in endpoint


class ImproveAllResponse(BaseModel):
    model: str
    llm_enabled: bool
    result: Dict[str, Any]
    matched_keywords: Optional[List[str]] = None
    missing_keywords: Optional[List[str]] = None
    notes: Optional[str] = None


# ----------------- Helpers / heuristics -----------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?)?\d{3,4}[\s\-\.]?\d{3,4}")
LINK_RE = re.compile(r"(https?://[^\s]+|www\.[^\s]+|linkedin\.com/[^\s,;]+|github\.com/[^\s,;]+)")
DATE_RE = re.compile(r"\b(?:\d{4}|\d{2}\/\d{4}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4})\b", re.I)

# Location patterns - cities and countries
LOCATION_PATTERNS = [
    # Countries (English)
    r"\b(Egypt|Cairo|Alexandria|Giza|United States|USA|US|UK|United Kingdom|UAE|Dubai|Abu Dhabi|Saudi Arabia|Riyadh|Jeddah|Qatar|Doha|Kuwait|Bahrain|Oman|Jordan|Amman|Lebanon|Beirut|Morocco|Casablanca|Tunisia|Algeria|Iraq|Baghdad|Syria|Palestine|Libya|Sudan|Khartoum|Yemen|India|Pakistan|Bangladesh|Indonesia|Malaysia|Philippines|Vietnam|Thailand|Turkey|Iran|Israel|Germany|France|Italy|Spain|Netherlands|Belgium|Switzerland|Austria|Poland|Sweden|Norway|Denmark|Finland|Ireland|Portugal|Greece|Czech Republic|Romania|Hungary|Ukraine|Russia|Canada|Mexico|Brazil|Argentina|Colombia|Chile|Peru|Venezuela|Australia|New Zealand|South Africa|Nigeria|Kenya|Ghana)\b",
    # Major cities worldwide
    r"\b(New York|Los Angeles|San Francisco|Chicago|Boston|Seattle|Austin|Denver|Atlanta|Miami|Washington|Houston|Dallas|Philadelphia|Phoenix|San Diego|London|Manchester|Birmingham|Edinburgh|Glasgow|Paris|Lyon|Marseille|Berlin|Munich|Frankfurt|Hamburg|Amsterdam|Rotterdam|Brussels|Zurich|Geneva|Vienna|Warsaw|Stockholm|Oslo|Copenhagen|Helsinki|Dublin|Lisbon|Madrid|Barcelona|Milan|Rome|Prague|Budapest|Bucharest|Athens|Istanbul|Tel Aviv|Toronto|Vancouver|Montreal|Ottawa|Calgary|Sydney|Melbourne|Brisbane|Perth|Auckland|Wellington|Singapore|Hong Kong|Tokyo|Osaka|Seoul|Busan|Beijing|Shanghai|Shenzhen|Guangzhou|Taipei|Mumbai|Bangalore|Delhi|Hyderabad|Chennai|Pune|Kolkata|Jakarta|Kuala Lumpur|Bangkok|Ho Chi Minh|Manila|Cape Town|Johannesburg|Lagos|Nairobi|Sao Paulo|Rio de Janeiro|Buenos Aires|Mexico City|Bogota|Lima|Santiago|Caracas|Remote|Hybrid)\b",
    # Egyptian cities
    r"\b(Mansoura|Tanta|Aswan|Luxor|Port Said|Suez|Ismailia|Fayoum|Minya|Assiut|Sohag|Qena|Damietta|Zagazig|Beni Suef|Kafr El Sheikh|Sharqia|Gharbia|Dakahlia|Beheira|Monufia|Helwan|Nasr City|Maadi|Dokki|Mohandessin|Heliopolis|6th of October|New Cairo|Sheikh Zayed|Obour|10th of Ramadan|Shoubra|Ain Shams)\b",
    # Arabic location names
    r"\b(مصر|القاهرة|الاسكندرية|الجيزة|المنصورة|طنطا|أسوان|الأقصر|بورسعيد|السويس|الإسماعيلية|الفيوم|المنيا|أسيوط|سوهاج|قنا|دمياط|الزقازيق|بني سويف|كفر الشيخ|الشرقية|الغربية|الدقهلية|البحيرة|المنوفية|حلوان|مدينة نصر|المعادي|الدقي|المهندسين|مصر الجديدة|السادس من أكتوبر|القاهرة الجديدة|الشيخ زايد|العبور|العاشر من رمضان|شبرا|عين شمس|السعودية|الرياض|جدة|الإمارات|دبي|أبوظبي|قطر|الدوحة|الكويت|البحرين|عمان|الأردن|عمّان|لبنان|بيروت)\b",
]
LOCATION_RE = re.compile("|".join(LOCATION_PATTERNS), re.I)

# Extended skills list
SKILLS_LIST = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl", "bash", "shell", "powershell",
    # Data Science & ML
    "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm", "catboost", "opencv", "nltk", "spacy", "huggingface", "transformers",
    # AI & LLM
    "langchain", "llamaindex", "openai", "gpt", "llm", "chatgpt", "claude", "gemini", "bert", "embedding", "rag", "vector database", "faiss", "chromadb", "pinecone", "weaviate", "milvus",
    # Data Engineering
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch", "cassandra", "oracle", "sql server", "sqlite", "neo4j", "dynamodb",
    "etl", "airflow", "apache airflow", "dbt", "kafka", "spark", "pyspark", "hadoop", "hive", "presto", "flink", "nifi", "talend", "informatica",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "ci/cd", "github actions", "gitlab ci", "circleci",
    "ec2", "s3", "lambda", "sagemaker", "emr", "glue", "athena", "redshift", "bigquery", "snowflake", "databricks", "synapse",
    # BI & Visualization
    "power bi", "powerbi", "tableau", "looker", "metabase", "superset", "qlik", "qlikview", "qliksense", "data studio", "quicksight", "grafana", "kibana",
    "excel", "advanced excel", "vba", "macros", "pivot tables", "vlookup", "xlookup", "power query", "power pivot", "dax",
    # Web Development
    "html", "css", "react", "reactjs", "react.js", "angular", "vue", "vue.js", "vuejs", "next.js", "nextjs", "nuxt", "svelte", "node.js", "nodejs", "express", "fastapi", "flask", "django", "spring", "spring boot", "laravel", ".net", "asp.net",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin", "ionic",
    # Tools & Version Control
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello", "asana", "notion", "slack", "teams",
    # Data Analysis
    "data analysis", "data analytics", "business intelligence", "bi", "statistical analysis", "predictive modeling", "machine learning", "deep learning", "nlp", "computer vision",
    "a/b testing", "hypothesis testing", "regression", "classification", "clustering", "time series", "forecasting",
    # Soft Skills (commonly in CVs)
    "agile", "scrum", "kanban", "project management", "team leadership", "communication", "problem solving", "critical thinking",
    # Other tools
    "jupyter", "colab", "streamlit", "gradio", "mlflow", "kubeflow", "dvc", "weights & biases", "wandb", "optuna",
    "api", "rest api", "restful", "graphql", "websocket", "microservices", "serverless",
    "linux", "unix", "windows server", "nginx", "apache",
    "selenium", "scrapy", "beautifulsoup", "requests", "playwright", "puppeteer",
    "figma", "adobe xd", "photoshop", "illustrator",
    "sap", "salesforce", "odoo", "erp", "crm",
]

def _extract_location(text: str) -> str:
    """Extract location (city/country) from text."""
    # Try to find location in first few lines (usually in contact section)
    top_text = "\n".join(_normalize_lines(text)[:10])
    matches = LOCATION_RE.findall(top_text)
    if matches:
        # Return first match, cleaned up
        loc = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0] else ""
        return loc.strip()
    # Fallback: search entire text
    matches = LOCATION_RE.findall(text)
    if matches:
        loc = matches[0] if isinstance(matches[0], str) else matches[0][0] if matches[0] else ""
        return loc.strip()
    return ""

def _extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using extended skills list."""
    text_lower = text.lower()
    found = []
    for skill in SKILLS_LIST:
        # Use word boundary for more accurate matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            # Capitalize properly
            found.append(skill.title() if len(skill) > 3 else skill.upper())
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in found:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique.append(s)
    return unique

def _has_date_in_line(line: Optional[str]) -> bool:
    """
    Return True if the given line contains a plausible date token.
    Uses DATE_RE defined above.
    """
    if not line:
        return False
    return bool(DATE_RE.search(line))

def _safe_text(x: Optional[str]) -> str:
    return (x or "").replace("\r", "\n").strip()

def _normalize_lines(text: str) -> List[str]:
    return [l.strip() for l in re.split(r"\n{1,}", text) if l.strip()]

def _extract_emails(text: str) -> List[str]:
    return list({m.group(0) for m in EMAIL_RE.finditer(text)})

def _extract_phones(text: str) -> List[str]:
    # attempt to extract plausible phone numbers
    found = []
    for m in PHONE_RE.finditer(text):
        s = m.group(0)
        # filter short numbers
        digits = re.sub(r"\D", "", s)
        if 6 <= len(digits) <= 15:
            found.append(s)
    return list(dict.fromkeys(found))

def _extract_links(text: str) -> Dict[str, str]:
    out = {"linkedin": "", "github": "", "website": "", "other": []}
    for m in LINK_RE.finditer(text):
        s = m.group(0).strip().strip(".,;()[]")
        s_norm = s if s.startswith("http") else "https://" + s
        if "linkedin.com" in s.lower():
            out["linkedin"] = s_norm
        elif "github.com" in s.lower():
            out["github"] = s_norm
        elif any(k in s.lower() for k in [".com", ".io", ".me", ".app"]):
            # heuristics: treat first non-github/linkedin domain as website
            if not out["website"]:
                out["website"] = s_norm
            else:
                out["other"].append(s_norm)
        else:
            out["other"].append(s_norm)
    return out

def _has_tables_or_columns(layout: Optional[str]) -> bool:
    if not layout:
        return False
    L = layout.lower()
    return any(k in L for k in ["table", "column", "2-column", "two-column", "textbox"])

def _section_markers() -> Dict[str, List[str]]:
    return {
        "summary": ["summary", "profile", "about me", "objective"],
        "experience": ["experience", "work experience", "employment", "professional experience", "work history"],
        "education": ["education", "academic", "qualifications", "degree", "university", "college", "bachelor", "master", "phd"],
        "skills": ["skills", "technical skills", "technologies", "competencies"],
        "projects": ["projects", "portfolio", "notable projects", "selected projects"],
        "certifications": ["certifications", "certificates", "licenses"],
        "languages": ["languages", "language"],
        "achievements": ["achievements", "awards", "honors"],
        "contact": ["contact", "contact info", "contact information"]
    }

# Education extraction patterns
EDUCATION_PATTERNS = [
    # Degrees
    r"(Bachelor|B\.?S\.?c?|B\.?A\.?|Master|M\.?S\.?c?|M\.?A\.?|MBA|Ph\.?D\.?|Diploma|Associate|Certificate)[\s\w]*(?:in|of)?\s*[\w\s]+",
    # Universities/Colleges
    r"(University|College|Institute|School|Academy|Faculty)[\s\w]+",
    # Common degree fields
    r"(Computer Science|Engineering|Business|Information Technology|IT|Data Science|Software|Mathematics|Economics|Commerce|Accounting|Medicine|Law|Arts|Science)",
    # Graduation years
    r"(Graduated?|Class of|Batch)?\s*:?\s*(19|20)\d{2}",
    # GPA
    r"(GPA|Grade|CGPA|Score)\s*:?\s*[\d\.]+\s*/?[\d\.]*",
]

def _extract_education_section(text: str) -> str:
    """Extract education-related content from CV text."""
    edu_content = []
    lines = text.split('\n')
    in_edu_section = False

    for line in lines:
        line_lower = line.lower().strip()

        # Check if we're entering education section
        if any(marker in line_lower for marker in ['education', 'academic', 'qualification', 'degree']):
            in_edu_section = True
            edu_content.append(line)
            continue

        # Check if we're leaving education section (new section header)
        if in_edu_section and any(marker in line_lower for marker in ['experience', 'skill', 'project', 'work', 'employment', 'certification']):
            in_edu_section = False
            continue

        # Collect education content
        if in_edu_section and line.strip():
            edu_content.append(line)

        # Also check for education keywords anywhere
        if not in_edu_section:
            for pattern in EDUCATION_PATTERNS:
                if re.search(pattern, line, re.I):
                    edu_content.append(line)
                    break

    return '\n'.join(edu_content)

def _find_section_splits(text: str) -> Dict[str, Tuple[int, int]]:
    """
    Primitive splitter: finds approximate positions of known section headers.
    Returns dict of section -> (start_idx, end_idx)
    """
    txt = text.lower()
    markers = _section_markers()
    positions = {}
    # find all marker occurrences
    found = []
    for sec, keys in markers.items():
        for k in keys:
            i = txt.find(k)
            if i != -1:
                found.append((i, sec, k))
    # sort by position
    found.sort()
    if not found:
        return {}
    # build ranges: each found -> start at its index; end at next found index
    for idx, (pos, sec, key) in enumerate(found):
        start = pos
        end = found[idx + 1][0] if idx + 1 < len(found) else len(txt)
        positions[sec] = (start, end)
    return positions

# ----------------- PDF extraction -----------------
def _extract_hyperlinks_from_pdf(data: bytes) -> List[str]:
    """
    Extract hyperlinks/URLs from PDF annotations (clickable links).
    These are often not visible in plain text extraction.
    """
    links = []
    # Try PyMuPDF (fitz) - best for extracting annotations
    try:
        import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                # Get links from annotations
                for link in page.get_links():
                    uri = link.get("uri", "")
                    if uri and uri.startswith(("http://", "https://", "www.", "linkedin.com", "github.com")):
                        links.append(uri)
                # Also check annotations directly
                for annot in page.annots() or []:
                    if annot.type[0] == 2:  # Link annotation
                        uri = annot.info.get("uri", "") or annot.info.get("url", "")
                        if uri:
                            links.append(uri)
    except Exception:
        pass

    # Try pypdf as fallback
    if not links:
        try:
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages:
                if "/Annots" in page:
                    annotations = page["/Annots"]
                    for annot in annotations:
                        obj = annot.get_object()
                        if obj.get("/Subtype") == "/Link":
                            action = obj.get("/A")
                            if action:
                                uri = action.get("/URI")
                                if uri:
                                    links.append(str(uri))
        except Exception:
            pass

    # Remove duplicates, preserve order
    seen = set()
    unique = []
    for link in links:
        link_clean = link.strip().rstrip("/")
        if link_clean.lower() not in seen:
            seen.add(link_clean.lower())
            unique.append(link_clean)
    return unique


def _extract_text_from_pdf_bytes(data: bytes) -> Tuple[str, str]:
    """
    Try multiple extraction methods and combine results for best coverage.
    Returns (text, layout_notes).
    """
    texts = []
    layout_hints = []

    # Extract hyperlinks first (these are often clickable but not visible in text)
    hyperlinks = _extract_hyperlinks_from_pdf(data)

    # 1) PyMuPDF - try multiple extraction modes
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            fitz_text = ""
            for page in doc:
                # Try "text" mode first (preserves reading order)
                ptxt = page.get_text("text") or ""
                fitz_text += ptxt + "\n"
                # Also try "blocks" mode for better structure detection
                blocks = page.get_text("blocks") or []
                if blocks and len(blocks) > 18:
                    layout_hints.append("many blocks (possible multi-column)")
            if fitz_text.strip():
                texts.append(fitz_text)
    except Exception:
        pass

    # 2) pdfplumber - good for tables and structured layouts
    try:
        import io, pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            plumber_text = ""
            for p in pdf.pages:
                # Extract text with layout preservation
                txt = p.extract_text(layout=True) or p.extract_text() or ""
                plumber_text += txt + "\n"
                # Also extract tables
                tables = p.extract_tables()
                if tables:
                    layout_hints.append("tables detected")
                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join(str(cell or "") for cell in row)
                                plumber_text += row_text + "\n"
            if plumber_text.strip():
                texts.append(plumber_text)
    except Exception:
        pass

    # 3) pypdf fallback
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pypdf_text = ""
        for page in reader.pages:
            pypdf_text += (page.extract_text() or "") + "\n"
        if pypdf_text.strip():
            texts.append(pypdf_text)
    except Exception:
        pass

    # Choose the longest/most complete extraction
    if texts:
        # Use the longest text as it likely has most complete extraction
        text = max(texts, key=len)
    else:
        text = ""

    # Append extracted hyperlinks to text (so they're available for parsing)
    if hyperlinks:
        links_section = " --- Extracted Links --- "
        for link in hyperlinks:
            links_section += f"{link} "
        text += links_section

    # Clean up text: remove excessive newlines and normalize spacing
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
    text = text.strip()

    layout_notes = "; ".join(sorted(set(layout_hints)))
    return text, layout_notes

# ----------------- LLM Gateway -----------------
class LLMGateway:
    def __init__(self, api_key: Optional[str], model: str = DEFAULT_MODEL):
        self.model = model
        self.enabled = bool(api_key and OpenAI is not None)
        self.client = OpenAI(api_key=api_key) if self.enabled else None

    def summarize_sections_with_llm(self, raw_text: str, language: str = "en") -> Dict[str, Any]:
        """
        If LLM enabled, ask it to produce a structured CV JSON with fixed 'item' keys.
        Keeps prompt restricted and asks for JSON only. If LLM not enabled, return None.
        """
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        short_raw = raw_text[:6000]  # limit
        prompt = (
            f"You are a robust CV parser. Input is plain extracted CV text. "
            f"Return JSON only. Language: {language}.\n\n"
            "Output structure must be exactly:\n"
            "{\n"
            '  "contact": {"item": {"name":"", "email":"", "phone":"", "location":"", "links": {"linkedin":"", "github":"", "website":""}}},\n'
            '  "summary": {"item": "..."},\n'
            '  "experience": {"item": [ {"company":"", "title":"", "date":"", "description":""}, ... ] },\n'
            '  "education": {"item": [ {"degree":"", "institution":"", "date":"", "notes":""}, ... ] },\n'
            '  "skills": {"item": ["skill1", "skill2", ...]},\n'
            '  "projects": {"item": [ {"name":"", "tech":"", "impact":""}, ... ] },\n'
            '  "certifications": {"item": ["cert1 (provider, year)", ...]},\n'
            '  "languages": {"item": ["English - Fluent", ...]},\n'
            '  "achievements": {"item": ["..."]}\n'
            "}\n\n"
            "Use best effort to extract names, emails, phones, links. If missing, keep empty strings. "
            "Keep dates compact like '2021-2023' or 'Jan 2020 - Dec 2021'. "
            "Do not include extraneous keys. Parse only from the following text:\n\n"
            f"---BEGIN CV---\n{short_raw}\n---END CV---\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise CV parsing assistant that returns JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # try parse
        try:
            # LLM may include backticks or text; extract first {...}
            s = raw.find("{")
            e = raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                parsed = json.loads(raw[s:e+1])
                return parsed
        except Exception:
            pass
        # fallback: raise to caller
        raise ValueError("LLM returned non-JSON or unparsable structure.")

    def improve_cv_section_with_llm(self, section: str, content: Any, target_role: Optional[str],
                                    jd_text: Optional[str], language: str, style: str,
                                    max_items: int) -> Any:
        """
        One-shot LLM-based improvement for a section. For improve_all usage.
        """
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        # craft prompt per section
        prompt = (
            f"You are an expert resume editor. Output language: {language}. Tone: {'ATS-friendly' if style=='ats' else 'professional'}.\n"
            f"Target role: {target_role or '(unspecified)'}.\n"
            f"Job Description:\n{jd_text or '(none)'}\n\n"
            f"Task: Improve the '{section}' section. Return JSON only with a single key 'item'. "
            "If the section is a list, return item as an array; if it's a plain summary, return item as a string.\n\n"
            f"Content to improve:\n{json.dumps(content, ensure_ascii=False)}\n"
            f"Max items: {max_items}\n"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior resume editor and ATS optimization expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # parse JSON
        try:
            s = raw.find("{")
            e = raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                parsed = json.loads(raw[s:e+1])
                return parsed.get("item", parsed)
        except Exception:
            # if not JSON, return raw text (caller will post-process)
            return raw

    def _jd_text(self, job_description: Optional[Union[str, List[str]]]) -> str:
        """Convert job description (string or list) to plain text."""
        if not job_description:
            return ""
        if isinstance(job_description, list):
            return "\n".join(str(jd) for jd in job_description)
        return str(job_description)

    def _llm(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """Generic LLM call - returns raw response text."""
        if not self.enabled:
            raise RuntimeError("LLM disabled")
        # Some models (like o1) don't support temperature parameter
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as e:
            if "temperature" in str(e).lower():
                # Retry without temperature for models that don't support it
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            else:
                raise
        return (resp.choices[0].message.content or "").strip()

# ----------------- App & gateway -----------------
app = FastAPI(title="CV ATS AI", version="1.0", openapi_version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

gateway = LLMGateway(api_key=OPENAI_API_KEY, model=DEFAULT_MODEL)


# ----------------- Endpoint: extract_cv_full -----------------
@app.post("/v1/extract_cv_full", response_model=ExtractResponse)
async def extract_cv_full(
    file: UploadFile = File(..., description="PDF CV file")
):
    """
    Upload a PDF CV -> returns raw_text only.
    Use this raw_text in /v1/improve_all or /v1/advice endpoints.
    """
    if (file.content_type or "").lower() not in {"application/pdf", "pdf"}:
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    raw_text, _ = _extract_text_from_pdf_bytes(data)
    if not raw_text.strip():
        raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")

    return ExtractResponse(raw_text=raw_text)


# ----------------- Endpoint: improve_all (HTML + feedback) -----------------
def _make_ats_friendly_html(cv_struct: Dict[str, Any], max_items: int = 10) -> str:
    """
    Build a clean, ATS-friendly HTML CV with changes feedback section at the bottom.
    """
    def esc(text):
        if not text:
            return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def clean_url(url):
        """Clean URL - remove ALL quotes, escapes, and ensure proper format."""
        if not url:
            return ""
        # Convert to string
        url = str(url)
        # Remove ALL types of quotes and escapes first
        for char in ['"', "'", '\\"', "\\'", '&quot;', '&#34;', '&#39;', '%22', '%27']:
            url = url.replace(char, '')
        # Strip whitespace
        url = url.strip()
        # Ensure https prefix
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        # Final cleanup - remove any trailing slashes for consistency, then add one
        url = url.rstrip('/')
        return url

    def get_val(obj, key, default=None):
        if not isinstance(obj, dict):
            return default
        val = obj.get(key, default)
        if isinstance(val, dict) and "item" in val:
            return val["item"]
        return val

    # Extract contact info
    contact = get_val(cv_struct, "contact", {})
    name = esc(contact.get("name", "") if isinstance(contact, dict) else "")
    email = contact.get("email", "") if isinstance(contact, dict) else ""
    phone = esc(contact.get("phone", "") if isinstance(contact, dict) else "")
    location = esc(contact.get("location", "") if isinstance(contact, dict) else "")
    links = contact.get("links", {}) if isinstance(contact, dict) else {}
    linkedin = links.get("linkedin", "") if isinstance(links, dict) else ""
    github = links.get("github", "") if isinstance(links, dict) else ""
    website = links.get("website", "") if isinstance(links, dict) else ""

    # A4 page size CSS with print styling (single line, no newlines)
    css = "@page { size: A4; margin: 15mm; } @media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } } * { box-sizing: border-box; } html, body { width: 210mm; min-height: 297mm; margin: 0 auto; padding: 0; } body { font-family: Arial, Helvetica, sans-serif; line-height: 1.4; color: #111; padding: 15mm; max-width: 210mm; background: #fff; } header { border-bottom: 2px solid #eee; padding-bottom: 12px; margin-bottom: 18px; } h1 { margin: 0; font-size: 24px; } .contact { margin-top: 6px; font-size: 14px; color: #222; } .contact a { color: #0066cc; text-decoration: none; } .contact a:hover { text-decoration: underline; } section { margin-bottom: 18px; } h2 { font-size: 16px; margin: 6px 0 8px 0; color: #111; } .job { margin-bottom: 10px; } .job .meta { font-size: 13px; color: #333; margin-bottom: 6px; } ul { margin: 6px 0 12px 18px; } li { margin-bottom: 6px; } .skills { list-style: none; margin: 0; padding: 0; } .skills li { display: inline-block; margin-right: 8px; padding: 4px 8px; border-radius: 4px; background: #f3f3f3; font-size: 13px; } .small { font-size: 13px; color: #444; }"

    html = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{name} — CV</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>{css}</style></head><body>']

    # Header
    html.append("<header>")
    html.append(f"<h1>{name}</h1>")
    contact_line = []
    if location:
        contact_line.append(location)
    if phone:
        contact_line.append(phone)
    if email:
        contact_line.append(email)
    if contact_line:
        html.append(f'<div class="contact">{" | ".join(contact_line)}</div>')
    # Links - must be clean URLs without any quotes
    linkedin_clean = clean_url(linkedin)
    github_clean = clean_url(github)
    website_clean = clean_url(website)
    if linkedin_clean or github_clean or website_clean:
        link_parts = []
        if linkedin_clean:
            link_parts.append('LinkedIn: <a href="' + linkedin_clean + '" target="_blank">' + linkedin_clean + '</a>')
        if github_clean:
            link_parts.append('GitHub: <a href="' + github_clean + '" target="_blank">' + github_clean + '</a>')
        if website_clean:
            link_parts.append('Website: <a href="' + website_clean + '" target="_blank">' + website_clean + '</a>')
        html.append('<div class="contact">' + " | ".join(link_parts) + '</div>')
    html.append("</header>")

    # Summary
    summary = get_val(cv_struct, "summary", "")
    if summary:
        html.append('<section id="summary">')
        html.append("<h2>Professional Summary</h2>")
        html.append(f'<p class="small">{esc(summary)}</p>')
        html.append("</section>")

    # Experience
    exps = get_val(cv_struct, "experience", [])
    if exps and isinstance(exps, list):
        html.append('<section id="experience">')
        html.append("<h2>Work Experience</h2>")
        for e in exps[:max_items]:
            if isinstance(e, dict):
                title = esc(e.get("title", ""))
                company = esc(e.get("company", ""))
                date = esc(e.get("date", ""))
                emp_type = esc(e.get("type", ""))
                desc = e.get("description", [])
                html.append('<div class="job">')
                # Line 1: title - company
                line1 = f"<strong>{title}</strong>"
                if company:
                    line1 += f" - {company}"
                html.append(f'<div class="meta">{line1}</div>')
                # Line 2: type | from-to
                line2_parts = []
                if emp_type:
                    line2_parts.append(emp_type)
                if date:
                    line2_parts.append(date)
                if line2_parts:
                    html.append(f'<div class="small">{" | ".join(line2_parts)}</div>')
                if desc:
                    html.append("<ul>")
                    bullets = desc if isinstance(desc, list) else [desc]
                    for b in bullets[:6]:
                        html.append(f"<li>{esc(b)}</li>")
                    html.append("</ul>")
                html.append("</div>")
        html.append("</section>")

    # Projects
    projs = get_val(cv_struct, "projects", [])
    if projs and isinstance(projs, list):
        html.append('<section id="projects">')
        html.append("<h2>Key Projects</h2>")
        html.append("<ul>")
        for p in projs[:max_items]:
            if isinstance(p, dict):
                pname = esc(p.get("name", ""))
                tech = esc(p.get("tech", ""))
                impact = esc(p.get("impact", ""))
                line = f"<strong>{pname}:</strong>"
                if tech:
                    line += f" {tech}."
                if impact:
                    line += f" {impact}"
                html.append(f"<li>{line}</li>")
            else:
                html.append(f"<li>{esc(str(p))}</li>")
        html.append("</ul>")
        html.append("</section>")

    # Skills
    skills = get_val(cv_struct, "skills", [])
    if skills and isinstance(skills, list):
        html.append('<section id="skills">')
        html.append("<h2>Technical Skills</h2>")
        html.append('<ul class="skills">')
        for sk in skills[:30]:
            html.append(f"<li>{esc(sk)}</li>")
        html.append("</ul>")
        html.append("</section>")

    # Certifications
    certs = get_val(cv_struct, "certifications", [])
    if certs and isinstance(certs, list):
        html.append('<section id="certifications">')
        html.append("<h2>Certifications</h2>")
        html.append("<ul>")
        for c in certs[:max_items]:
            html.append(f"<li>{esc(str(c))}</li>")
        html.append("</ul>")
        html.append("</section>")

    # Languages
    langs = get_val(cv_struct, "languages", [])
    if langs and isinstance(langs, list):
        html.append('<section id="languages">')
        html.append("<h2>Languages</h2>")
        html.append("<ul>")
        for lang in langs[:max_items]:
            html.append(f"<li>{esc(str(lang))}</li>")
        html.append("</ul>")
        html.append("</section>")

    # Achievements
    ach = get_val(cv_struct, "achievements", [])
    if ach and isinstance(ach, list):
        html.append('<section id="achievements">')
        html.append("<h2>Achievements</h2>")
        html.append("<ul>")
        for a in ach[:max_items]:
            html.append(f"<li>{esc(str(a))}</li>")
        html.append("</ul>")
        html.append("</section>")

    # Education
    edu = get_val(cv_struct, "education", [])
    if edu and isinstance(edu, list):
        html.append('<section id="education">')
        html.append("<h2>Education</h2>")
        html.append("<ul>")
        for ed in edu[:max_items]:
            if isinstance(ed, dict):
                deg = esc(ed.get("degree", ""))
                inst = esc(ed.get("institution", ""))
                dt = esc(ed.get("date", ""))
                line = f"<strong>{deg}</strong>"
                if inst:
                    line += f" - {inst}"
                if dt:
                    line += f" ({dt})"
                html.append(f"<li>{line}</li>")
            else:
                html.append(f"<li>{esc(str(ed))}</li>")
        html.append("</ul>")
        html.append("</section>")

    # No feedback section in HTML - it's returned separately
    html.append("</body></html>")
    # Join and clean up the HTML
    result = "".join(html)
    # Remove newlines
    result = result.replace("\n", "")
    return result

def _diff_structs(orig: Dict[str, Any], improved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a simple diff: for each section, list added/removed/updated items.
    This is naive but practical for feedback_json.
    """
    out = {}
    for sec in ["summary","experience","education","skills","projects","certifications","languages","achievements","contact"]:
        o = orig.get(sec, {}).get("item") if isinstance(orig.get(sec), dict) else orig.get(sec)
        n = improved.get(sec, {}).get("item") if isinstance(improved.get(sec), dict) else improved.get(sec)
        # normalize to strings/lists
        if isinstance(o, list) and isinstance(n, list):
            oset = set(json.dumps(x, ensure_ascii=False) if isinstance(x, (dict,list)) else str(x) for x in o)
            nset = set(json.dumps(x, ensure_ascii=False) if isinstance(x, (dict,list)) else str(x) for x in n)
            added = [json.loads(x) if x.startswith("{") else x for x in (nset - oset)]
            removed = [json.loads(x) if x.startswith("{") else x for x in (oset - nset)]
            out[sec] = {"added": added, "removed": removed}
        else:
            if (o or "") != (n or ""):
                out[sec] = {"before": o, "after": n}
    return out

def _extract_kws_from_text(text: str) -> List[str]:
    toks = {w.strip(".,:;()|").lower() for w in text.split() if len(w)>2}
    return sorted(list(toks))[:80]

@app.post("/v1/improve_all", response_model=ImproveAllResponse)
def improve_all(
    raw_text: str = Form(..., description="Raw CV text from PDF extraction"),
    job_description: str = Form("", description="Job description to tailor CV"),
    target_role: str = Form("", description="Target role (e.g., 'AI Engineer', 'Data Scientist')"),
    language: str = Form("en", description="Output language (en/ar)")
):
    """
    Improve CV using raw_text + optional job_description + target_role.
    Each input is a separate form field.
    Returns improved HTML CV using LLM.
    """
    if not gateway.enabled:
        raise HTTPException(status_code=503, detail="LLM disabled: set OPENAI_API_KEY and install openai.")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text is required.")

    jd_text = job_description or ""
    target_role_str = target_role or ""
    language_str = language or "en"

    # Extract education section specifically to help LLM
    edu_section = _extract_education_section(raw_text)

    # Build prompt for LLM to improve the CV
    prompt = f"""You are an expert ATS CV optimizer. Analyze this CV and improve it.

CV Text:
{raw_text}

=== EDUCATION SECTION (extracted) ===
{edu_section if edu_section else "No education section found - please extract from CV text above"}

Target Role: {target_role_str or "(not specified)"}

Job Description (if provided):
{jd_text or "(none)"}

Language: {language_str}

Task: Return a JSON object with:
1. Improved CV sections
2. A "changes_made" list describing what you improved

Structure:
{{
  "contact": {{"name": "", "email": "", "phone": "", "location": "", "links": {{"linkedin": "", "github": "", "website": ""}}}},
  "summary": "improved professional summary (2-3 sentences, ATS-friendly)",
  "experience": [{{"company": "", "title": "", "type": "Full-time/Part-time/Contract/Freelance/Internship", "date": "Jan 2020 - Dec 2023", "description": ["bullet1", "bullet2"]}}],
  "education": [{{"degree": "Bachelor of Science in Computer Science", "institution": "Cairo University", "date": "2018 - 2022", "notes": "GPA: 3.5/4.0"}}],
  "skills": ["skill1", "skill2"],
  "projects": [{{"name": "", "tech": "", "impact": ""}}],
  "certifications": ["cert1", "cert2"],
  "languages": ["English - Fluent"],
  "achievements": ["achievement1"],
  "changes_made": [
    "Rewrote summary to highlight {target_role_str or 'target role'} skills",
    "Added action verbs and metrics to experience bullets",
    "Prioritized skills relevant to job description",
    "..."
  ]
}}

IMPORTANT INSTRUCTIONS:
- EDUCATION is REQUIRED: Extract ALL education entries from the CV. Look for degrees (Bachelor, Master, PhD, Diploma), universities, colleges, schools, graduation years. NEVER leave education empty if any educational background exists in the CV.
- Make bullets action-oriented with metrics where possible.
- Tailor to the target role "{target_role_str}" and job description if provided.
- Emphasize relevant skills and experience for the target role.
- In "changes_made", list specific improvements you made (5-10 items).
- Return JSON only.
"""

    def clean_url_string(url):
        """Clean URL - remove ALL quotes and escapes."""
        if not url:
            return ""
        url = str(url)
        # Remove backslash-quote sequences
        url = url.replace('\\"', '').replace("\\'", '')
        # Remove plain quotes
        url = url.replace('"', '').replace("'", '')
        # Remove HTML entities
        url = url.replace('&quot;', '').replace('&#34;', '').replace('&#39;', '')
        # Remove URL-encoded quotes
        url = url.replace('%22', '').replace('%27', '')
        url = url.strip()
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        return url.rstrip('/')

    def clean_contact_links(cv_data):
        """Clean all URLs in contact.links."""
        if not isinstance(cv_data, dict):
            return cv_data
        contact = cv_data.get("contact", {})
        if isinstance(contact, dict):
            links = contact.get("links", {})
            if isinstance(links, dict):
                for key in ["linkedin", "github", "website"]:
                    if key in links and links[key]:
                        links[key] = clean_url_string(links[key])
        return cv_data

    try:
        raw_response = gateway._llm(prompt, "You are a senior CV/resume expert and ATS optimization specialist.")
        # Parse JSON from response
        s = raw_response.find("{")
        e = raw_response.rfind("}")
        if s != -1 and e != -1 and e > s:
            improved_cv = json.loads(raw_response[s:e+1])
            # Clean URLs in contact links
            improved_cv = clean_contact_links(improved_cv)
        else:
            raise ValueError("No JSON in response")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(ex)}")

    # Extract changes_made from response
    changes_made = improved_cv.pop("changes_made", [])
    if not isinstance(changes_made, list):
        changes_made = []

    # Wrap sections with 'item' key for HTML generator
    wrapped_cv = {}
    for key, val in improved_cv.items():
        wrapped_cv[key] = {"item": val}

    # Build HTML CV (clean, no feedback section)
    html_cv = _make_ats_friendly_html(wrapped_cv, max_items=10)

    # Format changes_made as readable text
    changes_text = ""
    if changes_made:
        changes_text = "🔧 Changes Made by AI:\n"
        for i, change in enumerate(changes_made, 1):
            changes_text += f"  {i}. {change}\n"

    # Keyword matching
    matched_kws = []
    missing_kws = []
    if jd_text:
        jd_tokens = set(_extract_kws_from_text(jd_text))
        cv_tokens = set(_extract_kws_from_text(json.dumps(improved_cv, ensure_ascii=False)))
        matched_kws = sorted(list(jd_tokens & cv_tokens))[:20]
        missing_kws = sorted(list(jd_tokens - cv_tokens))[:20]

    return ImproveAllResponse(
        model=gateway.model,
        llm_enabled=True,
        result={
            "html_cv": html_cv,
            "improved_cv": improved_cv,
            "changes_made": changes_made,
            "changes_text": changes_text
        },
        matched_keywords=matched_kws if matched_kws else None,
        missing_keywords=missing_kws if missing_kws else None,
        notes="Improved CV. Verify all facts and metrics."
    )

# ----------------- /v1/improve (unchanged behavior; role-only) -----------------
# For completeness we include a minimal /v1/improve that uses the existing LLM mini-improver
class ImproveRequest(BaseModel):
    section: str
    content: Any
    target_role: Optional[str] = None
    language: str = "en"
    style: str = "ats"
    max_items: int = 8
    return_mode: str = "json"
    group_skills: bool = False
    sort_bullets_by_impact: bool = True
    force_llm: bool = False

@app.post("/v1/improve")
def improve_endpoint(req: ImproveRequest):
    """
    Improve single section using LLM if available (tailored by target_role only).
    Kept intentionally simple and backward compatible.
    """
    if req.force_llm and not gateway.enabled:
        raise HTTPException(status_code=503, detail="LLM disabled and force_llm requested.")
    try:
        if gateway.enabled:
            res = gateway.improve_cv_section_with_llm(req.section, req.content, req.target_role, None, req.language, req.style, req.max_items)
            # return in consistent wrapper
            return {"section": req.section, "improved": {"item": res}, "model": gateway.model}
        else:
            # fallback: lightly cleaned
            if isinstance(req.content, list):
                return {"section": req.section, "improved": {"item": req.content[:req.max_items]}, "model": "fallback"}
            return {"section": req.section, "improved": {"item": req.content}, "model": "fallback"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- /v1/advice (JD-aware advice) -----------------
@app.post("/v1/advice")
def give_advice(
    raw_text: str = Form(..., description="Raw CV text from PDF extraction"),
    job_description: str = Form("", description="Job description to tailor advice"),
    target_role: str = Form("", description="Target role (e.g., 'AI Engineer', 'Data Scientist')"),
    language: str = Form("en", description="Output language (en/ar)")
):
    """
    Get career advice for CV improvement.
    Each input is a separate form field.
    """
    if not gateway.enabled:
        raise HTTPException(status_code=503, detail="LLM disabled: set OPENAI_API_KEY and install openai.")

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text is required.")

    jd_text = job_description or ""
    target_role_str = target_role or ""
    language_str = language or "en"

    # Extract education section specifically
    edu_section = _extract_education_section(raw_text)

    prompt = f"""You are a senior AI Career Advisor and ATS Expert.
Analyze this CV and provide actionable advice.

CV:
{raw_text}

=== EDUCATION SECTION (extracted) ===
{edu_section if edu_section else "No education section found - please extract from CV text above"}

Target Role: {target_role_str or "(not specified)"}

Job Description (if provided):
{jd_text or "(none)"}

Language: {language_str}

Provide advice specifically tailored to help this person become a successful "{target_role_str}" if specified.

Return JSON only with these keys:
{{
  "section_feedback": {{"summary": "...", "experience": "...", "skills": "...", "projects": "...", "education": "..."}},
  "missing_skills": ["skill1", "skill2"],
  "recommended_projects": ["project idea 1 relevant to {target_role_str}", "project idea 2"],
  "recommended_certifications": ["cert1 relevant to {target_role_str}", "cert2"],
  "ats_tips": ["tip1", "tip2"],
  "career_advice": "overall career advice paragraph for becoming {target_role_str}"
}}
"""

    try:
        raw = gateway._llm(prompt, "You are an expert career advisor.")
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(raw[s:e+1])
        else:
            raise ValueError("No JSON in response")
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(ex)}")


# ----------------- Endpoint: improve_all_html (returns raw HTML) -----------------
@app.post("/v1/improve_all_html", response_class=HTMLResponse)
def improve_all_html(
    raw_text: str = Form(..., description="Raw CV text from PDF extraction"),
    job_description: str = Form("", description="Job description to tailor CV"),
    target_role: str = Form("", description="Target role (e.g., 'AI Engineer', 'Data Scientist')"),
    language: str = Form("en", description="Output language (en/ar)")
):
    """
    Same as /v1/improve_all but returns raw HTML directly (not wrapped in JSON).
    Use this endpoint when you need clean HTML without JSON escaping.
    """
    # Call the main improve_all logic
    response = improve_all(raw_text, job_description, target_role, language)
    # Return just the HTML
    return HTMLResponse(content=response.result["html_cv"], media_type="text/html")


# ----------------- Health -----------------
@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": gateway.enabled, "model": gateway.model if gateway.enabled else "fallback"}
