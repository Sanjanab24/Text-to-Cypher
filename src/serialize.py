"""Turn a graph schema into model input text (used by the text-only baseline).

Format (one line):
  question: <q> | nodes: Label(prop TYPE, ...); ... | rel props: TYPE(prop TYPE, ...); ...
  | edges: (Start|Start2)-[:TYPE]->(End); ...
Labels with many properties are pruned to the question-matched ones plus a few others (see prune).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMAS = json.load(open(ROOT / "data/processed/schemas.json", encoding="utf-8"))

# shorter type names keep long schemas within the input budget
SHORT = {"STRING": "str", "INTEGER": "int", "FLOAT": "float", "BOOLEAN": "bool", "DATE": "date",
         "DATE_TIME": "datetime", "LOCAL_DATE_TIME": "localdatetime", "POINT": "point",
         "LIST": "list", "DURATION": "duration"}


MAX_PROPS = 12  # per label / relationship type, after question-matched properties


def _props(props):
    return ", ".join(f"{p['property']} {SHORT.get(p['datatype'], p['datatype'].lower())}" for p in props)


def name_tokens(name):
    """'primarySubstance' / 'call_date' / 'alt_name:es' -> {'primary', 'substance'} etc."""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return {t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t}


def _matches(prop, q_tokens, q_text):
    toks = name_tokens(prop)
    return bool(toks & q_tokens) or (len(prop) >= 4 and prop.lower() in q_text)


def prune(props, question):
    """Keep every property if few; otherwise the question-matched ones first, then the rest up to MAX_PROPS.
    Uses only the question, never the answer, so it behaves the same at training and test time."""
    if question is None or len(props) <= MAX_PROPS:
        return props
    q_text = question.lower()
    q_tokens = name_tokens(question)
    hit = [p for p in props if _matches(p["property"], q_tokens, q_text)]
    rest = [p for p in props if p not in hit]
    return hit + rest[:max(0, MAX_PROPS - len(hit))]


def internal(label):
    return label.startswith("_")  # Neo4j tool labels such as _Bloom_Perspective_


def grouped_edges(relationships):
    """Group edges by type: multi-label graphs list every label combination, which explodes the schema."""
    by_type = {}
    for r in relationships:
        if internal(r["start"]) or internal(r["end"]):
            continue
        starts, ends = by_type.setdefault(r["type"], ([], []))
        if r["start"] not in starts:
            starts.append(r["start"])
        if r["end"] not in ends:
            ends.append(r["end"])
    return by_type


def schema_text(graph, question=None):
    s = SCHEMAS[graph]
    nodes = "; ".join(f"{label}({_props(prune(props, question))})"
                      for label, props in s["node_props"].items() if not internal(label))
    rels = "; ".join(f"{t}({_props(prune(props, question))})" for t, props in s["rel_props"].items() if props)
    edges = "; ".join(f"({'|'.join(a)})-[:{t}]->({'|'.join(b)})"
                      for t, (a, b) in grouped_edges(s["relationships"]).items())
    return f"nodes: {nodes} | rel props: {rels} | edges: {edges}"


def model_input(question, graph):
    return f"question: {question} | {schema_text(graph, question)}"
