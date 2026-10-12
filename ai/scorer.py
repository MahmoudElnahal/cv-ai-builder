from typing import List, Tuple
import numpy as np
import re
from .embeddings import embed_texts, cosine

def extract_keywords_simple(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}", text or "")
    stop = set(["and","or","with","for","the","a","an","in","to","of","on","by","as","at","is","are","be","we","you"])
    return [t.lower() for t in toks if len(t)>2 and t.lower() not in stop]

def match_score(cv_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    vecs = embed_texts([cv_text, jd_text])
    sim = cosine(vecs[0], vecs[1])  # [-1..1]
    sim_score = max(0.0, min(1.0, (sim+1)/2)) * 100

    cv_kw = set(extract_keywords_simple(cv_text))
    jd_kw = set(extract_keywords_simple(jd_text))
    missing = sorted(list(jd_kw - cv_kw))[:30]
    matched = sorted(list(jd_kw & cv_kw))[:30]

    coverage = (len(matched) / max(1, len(jd_kw))) * 100
    final = round((sim_score * 0.7) + (coverage * 0.3), 1)
    return final, missing, matched
