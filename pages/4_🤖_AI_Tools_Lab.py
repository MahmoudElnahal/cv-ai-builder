# pages/4_🤖_AI_Tools_Lab.py
import json, re, time
import streamlit as st
from typing import List, Dict, Any

# يعتمد على طبقة الـ Agent عندك (ai/agent.py) اللي بدورها تكلم llm_gateway
from ai.agent import CVAgent

st.set_page_config(page_title="AI Tools Lab", layout="wide")
st.title("🤖 AI Tools Lab")
st.caption("One lab for pro-grade CV writing: Summary, STAR Bullets, Rewriting, and Cover Letter — ATS-ready & JD-aware.")

# -------------------- Session --------------------
if "lab_history" not in st.session_state:
    st.session_state.lab_history = []  # list of dicts: {"tool":..., "input":..., "output":..., "ts": ...}
if "last_outputs" not in st.session_state:
    st.session_state.last_outputs = {}  # tool -> text

agent = CVAgent()

# -------------------- Helpers --------------------
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}")
STOP = {
    "and","or","with","for","the","a","an","in","to","of","on","by","as","at","is","are","be",
    "job","description","position","overview","team","candidate","will","responsible","this","role",
    "experience","skills","strong","advanced","basic","both","company","degree","bachelor","master",
    "requirements","preferred","must","nice","plus"
}

def jd_keywords(jd: str) -> List[str]:
    toks = [t.lower() for t in TOKEN.findall(jd or "") if len(t) > 2]
    words = [t for t in toks if t not in STOP]
    # شوية normalization مبسّط
    # شيل تكرارات وحاجات عامة جدًا
    dedup = []
    seen = set()
    bad = {"data","analysis","experience","skills","role","job","candidate","team","overview","position"}
    for w in words:
        if w in bad: 
            continue
        if w not in seen:
            seen.add(w)
            dedup.append(w)
    return dedup[:40]

def coverage(text: str, kws: List[str]) -> float:
    if not text or not kws: 
        return 0.0
    low = text.lower()
    hits = sum(1 for k in kws if k in low)
    return round(100.0 * hits / max(1, len(kws)), 1)

def wc(text: str) -> int:
    return len((text or "").split())

def reading_time_minutes(text: str) -> float:
    # 200 wpm تقريبًا
    return round(wc(text) / 200.0, 2)

def to_lines(s: str) -> List[str]:
    return [x.strip().lstrip("-• ").strip() for x in (s or "").split("\n") if x.strip()]

def from_lines(lines: List[str]) -> str:
    return "\n".join(f"- {l}" for l in lines)

def record(tool: str, user_input: Dict[str, Any], output: str):
    st.session_state.lab_history.append({
        "tool": tool,
        "input": user_input,
        "output": output,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    st.session_state.last_outputs[tool] = output

def download_buttons(label_prefix: str, text: str, filename_stub: str):
    col_a, col_b, col_c = st.columns(3)
    col_a.download_button(f"⬇️ {label_prefix} TXT", data=text, file_name=f"{filename_stub}.txt")
    md = f"```text\n{text}\n```"
    col_b.download_button(f"⬇️ {label_prefix} MD", data=md, file_name=f"{filename_stub}.md")
    payload = json.dumps({"content": text}, ensure_ascii=False, indent=2)
    col_c.download_button(f"⬇️ {label_prefix} JSON", data=payload, file_name=f"{filename_stub}.json")

def smart_inject_keywords(base_text: str, kws: List[str], max_k: int = 12) -> str:
    """يحقن كلمات من الـ JD بشكل طبيعي لو مش موجودة—بدون إفساد الأسلوب."""
    if not kws: return base_text
    missing = [k for k in kws if k not in (base_text or "").lower()]
    inject = ", ".join(missing[:max_k])
    if not inject:
        return base_text
    # إدخال لطيف في آخر فقرة
    return (base_text or "").rstrip() + f"\n\nKeywords focus: {inject}"

def star_bullets(lines: List[str]) -> List[str]:
    """تحويل bullets لصيغة S/T-A-R بشكل مختصر قدر الإمكان."""
    out = []
    for l in lines:
        l = l.strip().rstrip(".")
        # heuristic مبسّطة
        if ":" in l or "→" in l:
            out.append(l)
        else:
            out.append(f"Situation/Task: {l} → Action: Drove execution → Result: +20–40% impact.")
    return out

def quantify(lines: List[str]) -> List[str]:
    out = []
    for l in lines:
        if re.search(r"\d+%|\d+\+|\d+k|\d+K|\d+ months|\d+ users", l, flags=re.I):
            out.append(l)
        else:
            out.append(l + " (+20–30% improvement)")
    return out

# -------------------- Sidebar Settings --------------------
st.sidebar.header("Model Settings")
creativity = st.sidebar.slider("Creativity (temperature)", 0.0, 1.2, 0.6, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 128, 2048, 512, 64)
lang = st.sidebar.selectbox("Output language", ["English", "Arabic", "Bilingual (EN first)"], index=0)
tone = st.sidebar.selectbox("Tone", ["Professional", "Confident", "Concise", "Storytelling", "Academic"], index=0)
length_pref = st.sidebar.selectbox("Length", ["Short", "Medium", "Long"], index=1)

st.sidebar.header("ATS / JD Options")
auto_inject = st.sidebar.checkbox("Auto-inject JD keywords", value=True)
force_star = st.sidebar.checkbox("STAR-style bullets", value=True)
auto_quant = st.sidebar.checkbox("Quantify results automatically", value=True)

st.sidebar.header("Utilities")
show_history = st.sidebar.checkbox("Show session history", value=False)

# -------------------- Tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Bullets", "Rewriter", "Cover Letter"])

# ========================= TAB 1: SUMMARY =========================
with tab1:
    st.subheader("Summary Generator (ATS-friendly)")
    c1, c2 = st.columns(2)
    with c1:
        cv_json = st.text_area(
            "CV (JSON, minimally include name/title/skills)",
            value=json.dumps({"name":"John Doe","title":"Data Analyst","skills":["SQL","Power BI","Python"]}, indent=2),
            height=220
        )
    with c2:
        jd_text = st.text_area("Job Description (optional but recommended)", height=220,
                               value="We need a data analyst with SQL, Power BI, Python, and dashboarding.")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Generate Summary", key="btn_sum"):
            try:
                seed = json.loads(cv_json)
            except Exception as e:
                st.error(f"Invalid CV JSON: {e}")
                seed = {"name":"", "title":"", "skills":[]}

            try:
                out = agent.generate_summary(
                    seed, jd_text,
                    strength=round(creativity*100),
                    options={"tone": tone, "length": length_pref, "language": lang, "max_tokens": max_tokens}
                ) or ""
            except Exception as e:
                st.warning(f"AI summary failed: {e}")
                out = f"{seed.get('title','Professional')} with strong skills in {', '.join(seed.get('skills', [])[:5])}."

            kws = jd_keywords(jd_text) if auto_inject else []
            final = smart_inject_keywords(out, kws) if auto_inject else out

            st.session_state.last_outputs["summary"] = final
            record("summary", {"cv": seed, "jd": jd_text}, final)
    with colB:
        if st.button("Use demo sample", key="btn_sum_demo"):
            st.session_state.last_outputs["summary"] = (
                "Results-driven Data Analyst with advanced SQL, Python, and Power BI; "
                "builds KPI dashboards and turns data into clear decisions."
            )

    summary_out = st.session_state.last_outputs.get("summary", "")
    if summary_out:
        st.markdown("### Output")
        st.write(summary_out)
        # quality
        kws = jd_keywords(jd_text)
        st.caption(f"Word count: {wc(summary_out)} | Read time: {reading_time_minutes(summary_out)} min | JD coverage: {coverage(summary_out, kws)}%")
        download_buttons("Summary", summary_out, "summary")

# ========================= TAB 2: BULLETS =========================
with tab2:
    st.subheader("STAR Bullets Generator (JD-tailored)")
    role = st.text_input("Target role", value="Data Analyst", key="role_bul")
    jd2 = st.text_area("JD for bullets (paste full JD or top requirements)", height=200,
                       value="SQL, Power BI, dashboarding, stakeholder communication, Python.")
    prev_lines = st.text_area("Existing bullets (optional, one per line)", height=140,
                              value="- Built HR dashboards in Power BI\n- Automated KPI tracking with SQL")

    if st.button("Generate Bullets", key="btn_bul"):
        base_lines = to_lines(prev_lines)
        try:
            gen = agent.generate_bullets(
                role, jd2,
                strength=round(creativity*100),
                options={"tone": tone, "length": length_pref, "language": lang, "max_tokens": max_tokens}
            ) or []
        except Exception as e:
            st.warning(f"AI bullets failed: {e}")
            gen = [
                "Built interactive dashboards that reduced reporting time by 30%",
                "Automated KPI pipelines in SQL improving accuracy by 20%"
            ]
        # تنظيف + دمج
        gen = [x.strip().rstrip(".") for x in gen if x.strip()]
        merged = []
        seen = set()
        for x in base_lines + gen:
            k = x.lower()
            if k not in seen:
                merged.append(x)
                seen.add(k)

        if force_star:
            merged = star_bullets(merged)
        if auto_quant:
            merged = quantify(merged)

        # حقن كلمات الـ JD في bullets النهائية
        if auto_inject:
            kws = jd_keywords(jd2)
            # حقن لطيف كسطر أخير لو ناقص تغطية
            txt_merge = "\n".join(merged).lower()
            missing = [k for k in kws if k not in txt_merge]
            if missing:
                merged.append(f"Keywords focus: {', '.join(missing[:10])}")

        out = from_lines(merged[:8])
        st.session_state.last_outputs["bullets"] = out
        record("bullets", {"role": role, "jd": jd2, "prev": base_lines}, out)

    bullets_out = st.session_state.last_outputs.get("bullets", "")
    if bullets_out:
        st.markdown("### Output")
        st.write(bullets_out)
        kws = jd_keywords(jd2)
        st.caption(f"Lines: {len(to_lines(bullets_out))} | JD coverage: {coverage(bullets_out, kws)}%")
        download_buttons("Bullets", bullets_out, "bullets")

# ========================= TAB 3: REWRITER =========================
with tab3:
    st.subheader("ATS Rewriter")
    txt = st.text_area("Text to rewrite (paste any paragraph or bullet list)", height=180,
                       value="Responsible for data analysis and reporting.")
    jd3 = st.text_area("JD context (optional)", height=140, value="SQL, Power BI, stakeholder communication.")
    c1, c2, c3 = st.columns(3)
    with c1:
        keep_keywords = st.checkbox("Preserve existing keywords", value=True)
    with c2:
        simplify = st.checkbox("Simplify sentences", value=True)
    with c3:
        formalize = st.checkbox("More formal tone", value=False)

    if st.button("Rewrite", key="btn_rew"):
        opts = {"tone": tone, "length": length_pref, "language": lang, "max_tokens": max_tokens,
                "preserve": keep_keywords, "simplify": simplify, "formalize": formalize}
        try:
            out = agent.rewrite_text(txt, jd3, strength=round(creativity*100), options=opts) or ""
        except Exception as e:
            st.warning(f"AI rewrite failed: {e}")
            out = txt

        if auto_inject:
            out = smart_inject_keywords(out, jd_keywords(jd3))
        st.session_state.last_outputs["rewritten"] = out
        record("rewriter", {"text": txt, "jd": jd3, "opts": opts}, out)

    rw_out = st.session_state.last_outputs.get("rewritten", "")
    if rw_out:
        st.markdown("### Output")
        st.write(rw_out)
        st.caption(f"Word count: {wc(rw_out)} | Read time: {reading_time_minutes(rw_out)} min | JD coverage: {coverage(rw_out, jd_keywords(jd3))}%")
        download_buttons("Rewritten", rw_out, "rewritten")

    st.markdown("#### Compare (Before → After)")
    colx, coly = st.columns(2)
    with colx:
        st.text_area("Before", value=txt, height=180, key="cmp_before")
    with coly:
        st.text_area("After", value=rw_out, height=180, key="cmp_after")

# ========================= TAB 4: COVER LETTER =========================
with tab4:
    st.subheader("Cover Letter Generator")
    cv_json2 = st.text_area(
        "Candidate Profile (JSON)",
        value=json.dumps({"name":"John Doe","title":"Data Analyst","skills":["SQL","Power BI","Python"]}, indent=2),
        height=180
    )
    jd4 = st.text_area("JD (paste full job description)", height=200,
                       value="We are hiring a Data Analyst at ACME. SQL, Power BI, dashboards, communication.")
    c1, c2 = st.columns(2)
    with c1:
        words_min = st.number_input("Min words", 180, 600, 220, 10)
    with c2:
        words_max = st.number_input("Max words", 220, 1200, 320, 10)

    if st.button("Generate Cover Letter", key="btn_cl"):
        try:
            profile = json.loads(cv_json2)
        except Exception as e:
            st.error(f"Invalid profile JSON: {e}")
            profile = {"name":"", "title":""}

        try:
            if hasattr(agent, "generate_cover_letter"):
                letter = agent.generate_cover_letter(
                    profile, jd4,
                    strength=round(creativity*100),
                    options={"tone": tone, "length": length_pref, "language": lang,
                             "max_tokens": max_tokens, "min_words": int(words_min), "max_words": int(words_max)}
                ) or ""
            else:
                # fallback لميثود cover_letter القديمة لو متاحة
                letter = getattr(agent, "cover_letter", lambda *_: "")(profile, jd4) or ""
        except Exception as e:
            st.warning(f"AI cover letter failed: {e}")
            letter = ""

        if auto_inject:
            letter = smart_inject_keywords(letter, jd_keywords(jd4))
        st.session_state.last_outputs["cover_letter"] = letter
        record("cover_letter", {"profile": profile, "jd": jd4}, letter)

    cl_out = st.session_state.last_outputs.get("cover_letter", "")
    if cl_out:
        st.markdown("### Output")
        st.write(cl_out)
        st.caption(f"Word count: {wc(cl_out)} | Read time: {reading_time_minutes(cl_out)} min | JD coverage: {coverage(cl_out, jd_keywords(jd4))}%")
        download_buttons("Cover Letter", cl_out, "cover_letter")

# ========================= History =========================
if show_history and st.session_state.lab_history:
    st.markdown("---")
    st.subheader("Session History")
    for i, item in enumerate(reversed(st.session_state.lab_history[-20:]), 1):
        with st.container():
            st.markdown(f"**{i}. {item['tool']}** — _{item['ts']}_")
            st.caption(f"Input: {json.dumps(item['input'], ensure_ascii=False)[:280]}{'…' if len(json.dumps(item['input'], ensure_ascii=False))>280 else ''}")
            st.code(item["output"][:1200] + ("…" if len(item["output"])>1200 else ""), language="markdown")
