"""Cached resources and data shared by the demo app pages."""
import json
from pathlib import Path

import streamlit as st
import torch

from aml_questions import AML_QUESTIONS, DEMO
from inference import load_model
from neo4j_utils import get_driver

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data/processed/mtq"

MODELS = {"baseline": ("baseline_t5", "Baseline (CodeT5+)"), "gat": ("gat_t5", "GAT model (ours)")}
# the financial graph uses the models fine-tuned on synthetic financial pairs (domain adaptation)
ADAPTED = {"baseline": "baseline_aml", "gat": "gat_aml"}

# graph key -> (display name, how the model relates to it)
GRAPHS = {
    "aml": ("IBM AML financial transactions", "financial domain"),
    "wwc": ("Women's World Cup", "unseen test schema"),
    "healthcare": ("Healthcare", "unseen test schema"),
    "bloom50": ("Bloom", "unseen test schema"),
    "er": ("Entity resolution", "validation schema"),
    "legis_graph": ("US legislature", "seen in training"),
    "covid": ("COVID-19", "seen in training"),
    "pole": ("Crime investigation (POLE)", "seen in training"),
    "twitter_trolls": ("Twitter trolls", "seen in training"),
    "osm": ("OpenStreetMap", "seen in training"),
    "gdsc": ("Graph data science", "seen in training"),
}

GROUPS = {
    "Financial": ["aml"],
    "Unseen test databases": ["wwc", "healthcare", "bloom50"],
    "Training databases": ["legis_graph", "covid", "pole", "twitter_trolls", "osm", "gdsc"],
    "Validation": ["er"],
}
COMBINED = "all_unseen"


def graph_label(g):
    name, kind = GRAPHS[g]
    return f"{name} ({kind})"


def pick_database(key, default_group="Financial", combined=False):
    """Database group, then a database within it (no second step for one-database groups).
    combined=True adds an 'all three combined' choice to the unseen group."""
    with st.container(horizontal=True, vertical_alignment="bottom"):
        group = st.segmented_control("Databases", list(GROUPS), default=default_group,
                                     key=f"{key}_group") or default_group
        options = ([COMBINED] if combined and group == "Unseen test databases" else []) + GROUPS[group]
        if len(options) == 1:
            return options[0]
        return st.selectbox("Database", options, key=f"{key}_{group}", width=280,
                            format_func=lambda g: "All three combined" if g == COMBINED else GRAPHS[g][0])


@st.cache_resource
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def demo_run(graph):
    """The model the demo answers with: our GAT model, fine-tuned on financial data for the financial graph."""
    return run_name("gat", graph)


def run_name(key, graph, use_adapted=True):
    adapted = ADAPTED[key]
    if graph == "aml" and use_adapted and (ROOT / "models" / adapted / "best").exists():
        return adapted
    return MODELS[key][0]


@st.cache_resource(show_spinner="Loading model...")
def model(run):
    return load_model(run, device())


@st.cache_resource
def driver():
    return get_driver()


def _jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


@st.cache_data
def details(run, name):
    return {d["id"]: d for d in _jsonl(RESULTS / run / f"{name}.details.jsonl")}


@st.cache_data
def examples(graph, n=6):
    """[{question, gold}] for the example picker. Dataset graphs show test questions the GAT model
    answered correctly, those the baseline missed first."""
    if graph == "aml":
        by_question = {i["question"]: i["cypher"] for i in AML_QUESTIONS}
        return [{"question": q, "gold": by_question[q]} for q in DEMO]
    split, gat_name, base_name = {
        "unseen test schema": ("test_unseen", "test_unseen.sample1000.eg", "test_unseen"),
        "seen in training": ("test_seen", "test_seen", "test_seen"),
    }.get(GRAPHS[graph][1], ("val", None, None))
    pairs = {p["id"]: p for p in _jsonl(DATA / f"{split}.jsonl") if p["graph"] == graph}
    if gat_name is None:
        chosen = [p for p in pairs.values() if p["verified"]][:n]
    else:
        gat, base = details("gat_t5", gat_name), details("baseline_t5", base_name)
        right = [i for i in pairs if i in gat and gat[i]["exec"] and not gat[i]["gold_empty"]]
        right.sort(key=lambda i: base.get(i, {}).get("exec", True))  # baseline misses first
        chosen = [pairs[i] for i in right[:n]]
    return [{"question": p["question"], "gold": p["cypher"]} for p in chosen]
