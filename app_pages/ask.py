import pandas as pd
import streamlit as st

from app_utils import GRAPHS, demo_run, device, driver, examples, model
from inference import answer, candidates

DATABASES = ["aml", "covid", "healthcare", "wwc", "bloom50", "legis_graph", "pole", "twitter_trolls", "osm", "gdsc",
             "er"]


class NoAnswer(Exception):
    """Raised inside the cached function so failed answers (e.g. timeouts) are not cached."""


@st.cache_data(show_spinner=False, max_entries=500)
def _ask(graph, question):
    m, tok, gat = model(demo_run(graph))
    cands = candidates(m, tok, gat, question, graph, device())
    k, cols, rows, err = answer(driver(), cands, graph)
    res = {"cypher": cands[k], "cols": cols, "rows": rows, "err": err}
    if err:
        raise NoAnswer(res)
    return res


def ask(graph, question):
    try:
        return _ask(graph, question)
    except NoAnswer as e:
        return e.args[0]


def show(v):
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


graph = st.selectbox("Database", DATABASES, format_func=lambda g: GRAPHS[g][0], key="ask_db", width=360)
samples = [e["question"] for e in examples(graph)][:4]
if not st.session_state.get(f"ask_ready_{graph}"):
    with st.spinner("Getting ready..."):  # load the model and answer the samples once, so clicks are instant
        for s in samples:
            ask(graph, s)
    st.session_state[f"ask_ready_{graph}"] = True
picked = st.pills("Sample questions", samples, key=f"ask_sample_{graph}")

with st.form("ask_form", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        question = st.text_input("Your question", value=picked or "", placeholder="Ask anything about this database")
        asked = st.form_submit_button("Ask", type="primary", icon=":material/send:")

if asked and question.strip():
    with st.spinner("Thinking..."):
        st.session_state.ask_last = (graph, question.strip(), ask(graph, question.strip()))

last = st.session_state.get("ask_last")
if last and last[0] == graph:
    _, q, res = last
    rows, cols = res["rows"], res["cols"]
    with st.container(border=True):
        st.caption(q)
        if res["err"] or not rows:
            st.subheader("Sorry, I couldn't find an answer.", anchor=False)
            st.caption("Try rephrasing the question.")
        elif len(rows) == 1 and len(cols) == 1:
            st.header(show(rows[0][0]), anchor=False)
        elif len(rows) == 1:
            st.header(show(rows[0][0]), anchor=False)
            st.caption(" · ".join(f"{c}: {show(v)}" for c, v in zip(cols[1:], rows[0][1:])))
        else:
            df = pd.DataFrame([[show(v) for v in r] for r in rows], columns=cols)
            st.dataframe(df, hide_index=True, height=min(360, 38 + 35 * len(df)), alt="Answer")
    with st.expander("Generated Cypher query"):
        st.code(res["cypher"], language="cypher", wrap_lines=True)
