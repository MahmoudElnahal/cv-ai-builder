import os, time, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

def chat_json(system: str, user: str, model: str = DEFAULT_MODEL, max_retries=3):
    for i in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                response_format={"type":"json_object"},
                temperature=0.3,
            )
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception:
            if i == max_retries-1: raise
            time.sleep(0.8 * (i+1))
