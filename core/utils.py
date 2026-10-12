import re, os, json
from typing import List

def clean_text(t: str) -> str:
    if not t: return ""
    t = re.sub(r"\s+", " ", t).strip()
    return t

def split_lines(text: str) -> List[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]

def as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [str(x)]
