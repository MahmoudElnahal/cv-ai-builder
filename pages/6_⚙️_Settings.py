import streamlit as st, os
st.set_page_config(page_title="Settings", layout="wide")
st.title("⚙️ Settings")

st.write("Set your API keys in a .env file at project root.")
st.code("""
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
""", language="bash")

if "OPENAI_API_KEY" in os.environ:
    st.success("OPENAI_API_KEY found in environment.")
else:
    st.warning("OPENAI_API_KEY not found. Create a .env or set environment variable.")
