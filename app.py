# app.py
import streamlit as st

st.set_page_config(
    page_title="V-AI — Resume Optimizer",
    page_icon="📝",
    layout="wide"
)

# ---- Sidebar Navigation ----
st.sidebar.title("📂 V-AI Prototype")
st.sidebar.markdown("All-in-one AI Resume builder/optimizer")

pages = {
    "🧪 ATS Scoring": "pages/1_🧪_ATS_Scoring.py",
    "🧰 CV Update": "pages/2_🧰_CV_Update.py",
    "✨ New CV Builder": "pages/3_✨_New_CV_Builder.py",
    "🤖 AI Tools Lab": "pages/4_🤖_AI_Tools_Lab.py",
    "📊 Reports Dashboard": "pages/5_📊_Reports_Dashboard.py",
    "⚙️ Settings": "pages/6_⚙️_Settings.py",
}

st.sidebar.markdown("---")
st.sidebar.caption("Use the menu above to navigate")

# ---- Main Landing ----
st.title("📝 V-AI — Streamlit Prototype")
st.markdown("""
Welcome to **V-AI**, your AI-powered resume assistant.  
Navigate using the sidebar to:

- 🧪 **ATS Scoring** → Upload CV + JD, check ATS match score.
- 🧰 **CV Update** → Optimize your existing CV for a target role.
- ✨ **New CV Builder** → Build a CV from scratch with AI support.
- 🤖 **AI Tools Lab** → Experiment with summary, bullets, and cover letters.
- 📊 **Reports Dashboard** → Visualize CV scores & insights.
- ⚙️ **Settings** → Configure API keys & preferences.
""")

st.info("💡 Tip: Start with *ATS Scoring* to analyze your current CV.")
