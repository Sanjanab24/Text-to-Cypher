"""Demo app: natural language -> Cypher with the baseline and the GAT model.

Run from C:\\Projects\\NLP_SEE (Neo4j must be running):
    .venv\\Scripts\\streamlit.exe run streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

st.set_page_config(page_title="Text-to-Cypher with graph attention", page_icon=":material/hub:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/ask.py", title="Ask a question", icon=":material/chat:", default=True),
        st.Page("app_pages/results.py", title="Results", icon=":material/bar_chart:"),
    ],
    position="top",
)
st.title(page.title, icon=page.icon)
page.run()
