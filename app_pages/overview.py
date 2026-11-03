import json

import streamlit as st

from app_utils import RESULTS

st.markdown("#### Ask a graph database a question in plain English, and get the Cypher query that answers it.")
st.caption("Graph Attention Transformer Framework for Natural Language to Cypher Query Generation in Financial "
           "Transaction Graphs · Team 15: Sanjana B, R Muni Disnitha, Saanvi Mahipati, Sanjana Gothe · "
           "Guide: Prof. Priyanka S Shivaram")

with st.container(border=True):
    st.markdown("**Example** on the financial transaction graph (5.1 million transfers)")
    left, mid, right = st.columns([1, 1.6, 1], vertical_alignment="center")
    left.markdown(":material/chat: **Question**\n\nWhich entity owns the most accounts?")
    mid.code("MATCH (e:Entity)-[:OWNS]->(a:Account)\nRETURN e.name, count(a) AS accounts\n"
             "ORDER BY accounts DESC LIMIT 1", language="cypher")
    right.markdown(":material/table: **Answer**\n\nCorporation #34854, 7,820 accounts")

st.markdown("##### Why it is hard")
c1, c2, c3 = st.columns(3)
with c1.container(border=True, height="stretch"):
    st.markdown(":material/translate: **Different words**")
    st.caption("People say \"entity\", \"bank country\", \"laundering\"; the database says Entity.name, "
               "Bank.country, TRANSFERRED_TO.is_laundering.")
with c2.container(border=True, height="stretch"):
    st.markdown(":material/hub: **Paths, not tables**")
    st.caption("\"Accounts per bank country\" means following Account -HELD_AT-> Bank before counting.")
with c3.container(border=True, height="stretch"):
    st.markdown(":material/new_releases: **New databases**")
    st.caption("Every database has its own schema; the model must link words to a schema it may never have seen.")

st.markdown("##### Results")
unseen = json.load(open(RESULTS / "compare_test_unseen_1000_eg.json"))
acc = {k.split(":")[0] + (" eg" if k.endswith(".eg") else ""): v["overall"]["exec_acc"] for k, v in unseen.items()}
base, best = acc["baseline_t5"], acc["gat_t5 eg"]


def financial(name):
    rows = json.load(open(RESULTS / f"{name}.json", encoding="utf-8"))
    return 100 * max(sum(r[k]["exec_guided"] for r in rows) for k in ["baseline", "gat"]) / len(rows)


zero, adapted = financial("aml_check"), financial("aml_check_adapted")
with st.container(horizontal=True):
    st.metric("Unseen databases: GAT + execution check", f"{best:.1f}%", f"{best - base:+.1f} pts vs baseline",
              border=True, height="stretch", help="Execution accuracy on 1,000 questions from 3 databases never "
                                                   "seen in training")
    st.metric("Financial questions: best setup after domain adaptation", f"{adapted:.0f}%",
              f"{adapted - zero:+.0f} pts vs zero-shot", border=True, height="stretch",
              help="Best of baseline and GAT, with execution check, on 55 hand-written financial questions")
    st.metric("Benchmark data", "27,529 pairs", border=True, height="stretch",
              help="Question-Cypher pairs over 11 Neo4j databases (Mind the Query benchmark)")
