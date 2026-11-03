import streamlit as st

st.mermaid_chart("""
graph LR
    Q[Question +<br/>database schema] --> E[Text encoder<br/>CodeT5+]
    Q --> G[GAT schema encoder<br/>links words to the schema]
    E --> D[Decoder writes<br/>4 candidate queries]
    G --> D
    D --> X{Execution check<br/>on Neo4j}
    X --> C[Cypher + answer]
""", alt="Pipeline: question and schema go through the text encoder and GAT schema encoder; the decoder writes "
         "candidates; an execution check on Neo4j picks the final query")

c1, c2, c3 = st.columns(3)
with c1.container(border=True, height="stretch"):
    st.markdown(":material/text_fields: **1. Read the question**")
    st.caption("A pretrained Transformer (CodeT5+, 220M parameters) reads the question together with the "
               "database schema written as text. This model alone is our baseline.")
with c2.container(border=True, height="stretch"):
    st.markdown(":material/hub: **2. Link words to the schema (GAT)**")
    st.caption("Each label, relationship and property becomes a node in a small graph. Nodes attend to the "
               "question, then graph attention passes this along the schema's relationships, so the decoder "
               "knows which parts of the database the question is about.")
with c3.container(border=True, height="stretch"):
    st.markdown(":material/fact_check: **3. Check by running it**")
    st.caption("The decoder proposes 4 queries. Each is run on Neo4j (read-only); the first that runs and "
               "returns rows is chosen. The correct answer is never used.")

with st.expander("Training and evaluation"):
    st.markdown("""
- **Training data:** Mind the Query benchmark, 27,529 question-Cypher pairs over 11 databases. Train on 7,
  validate on 1, test on 3 databases never seen in training.
- **Financial domain:** IBM AMLworld transactions (5.1 million transfers) loaded into Neo4j; 55 hand-written test
  questions. Domain adaptation adds ~1,000 synthetic financial training questions (test questions held out).
- **Metric:** execution accuracy, meaning the generated query returns the same answer as the reference query.
""")
