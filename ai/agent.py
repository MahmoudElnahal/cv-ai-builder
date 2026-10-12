import json
from .llm_gateway import chat_json
from .prompts import SUMMARY_SYS, SUMMARY_USER, BULLETS_SYS, BULLETS_USER, REWRITE_SYS, REWRITE_USER, CL_SYS, CL_USER

class CVAgent:
    def generate_summary(self, cv_obj, jd_text):
        return chat_json(SUMMARY_SYS, SUMMARY_USER(json.dumps(cv_obj), jd_text)).get("summary","")

    def generate_bullets(self, role, jd_text):
        return chat_json(BULLETS_SYS, BULLETS_USER(role, jd_text)).get("bullets",[])

    def rewrite_text(self, text):
        return chat_json(REWRITE_SYS, REWRITE_USER(text)).get("rewritten","")

    def cover_letter(self, cv_obj, jd_text):
        return chat_json(CL_SYS, CL_USER(json.dumps(cv_obj), jd_text)).get("cover_letter","")
