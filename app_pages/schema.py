import streamlit as st

from app_utils import pick_database
from gat_model import KINDS, schema_graph
from serialize import SCHEMAS, grouped_edges, internal, schema_text

graph = pick_database("schema")
view = st.segmented_control("View", ["Labels and relationships", "Graph seen by the GAT encoder"],
                            default="Labels and relationships", key="schema_view")

names, kinds, edges = schema_graph(graph)
with st.container(horizontal=True):
    for kind in KINDS:
        st.metric(f"{kind.replace('_', ' ').capitalize()} nodes", sum(KINDS[k] == kind for k in kinds), border=True)
    st.metric("Edges in the GAT graph", len(edges) // 2, border=True)

s = SCHEMAS[graph]
if view == "Graph seen by the GAT encoder":
    st.caption("One node per label (blue), relationship type (orange) and property (grey); "
               "edges: label has property, start label -> relationship type -> end label.")
    style = {0: 'shape=box style="rounded,filled" fillcolor="#cde2fb"',
             1: 'shape=ellipse style=filled fillcolor="#fbd5c4"',
             2: 'shape=plaintext fontsize=10 fontcolor="#52514e"'}
    lines = [f'n{i} [label="{n}" {style[k]}];' for i, (n, k) in enumerate(zip(names, kinds))]
    lines += [f"n{a} -> n{b};" for a, b, t in edges if t < 4]  # forward direction only
    dot = "digraph { layout=neato; overlap=false; sep=\"+6\"; node [fontname=Helvetica fontsize=11]; " \
          "edge [color=\"#9a9993\" arrowsize=0.5 len=1.3];\n" \
          + "\n".join(lines) + "\n}"
else:
    st.caption("Node labels with their properties; arrows are relationship types.")
    lines = []
    for label, props in s["node_props"].items():
        if internal(label):
            continue
        plist = "\\l".join(p["property"] for p in props[:10]) + ("\\l..." if len(props) > 10 else "") + "\\l"
        lines.append(f'"{label}" [shape=record style="rounded,filled" fillcolor="#cde2fb" '
                     f'label="{{{label}|{plist}}}"];')
    for t, (starts, ends) in grouped_edges(s["relationships"]).items():
        for a in starts:
            for b in ends:
                lines.append(f'"{a}" -> "{b}" [label="{t}" fontsize=10];')
    dot = "digraph { rankdir=LR; node [fontname=Helvetica fontsize=11]; edge [fontname=Helvetica color=\"#9a9993\"];\n" \
          + "\n".join(lines) + "\n}"

st.graphviz_chart(dot, width="stretch")

with st.expander("Schema as text (what the baseline reads)"):
    st.code(schema_text(graph), language="text", wrap_lines=True)
