SUMMARY_SYS = "You are an ATS-savvy resume assistant. Return JSON only."
def SUMMARY_USER(cv_json, jd_text):
    return f"""
Given this CV (JSON) and Job Description, write a concise ATS-friendly professional summary (3-5 lines).
Return: {{"summary": "..."}}
CV: ```json
{cv_json}
```
JD: ```
{jd_text}
```
"""

BULLETS_SYS = "You write strong, action-oriented, quantified resume bullets. Return JSON only."
def BULLETS_USER(role, jd_text):
    return f"""
Generate 5 resume experience bullets tailored to the role '{role}' and this JD. Each bullet <= 20 words.
Return: {{"bullets": ["...", "..."]}}
JD: ```
{jd_text}
```
"""

REWRITE_SYS = "Rewrite given resume text in ATS-friendly, concise style. Return JSON only."
def REWRITE_USER(text):
    return f"""
Rewrite the following text in an ATS-friendly concise style. Return: {{"rewritten": "..."}}
Text: ```
{text}
```
"""

CL_SYS = "You write tailored cover letters in 120-180 words. Return JSON only."
def CL_USER(cv_json, jd_text):
    return f"""
Write a tailored cover letter (120-180 words). Return: {{"cover_letter": "..."}}
CV: ```json
{cv_json}
```
JD: ```
{jd_text}
```
"""
