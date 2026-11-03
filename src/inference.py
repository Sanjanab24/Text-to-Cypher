"""Single-question inference for the demo app: generate beams, pick one by execution, run it on Neo4j."""
import logging
from pathlib import Path

import neo4j
import torch
from neo4j import Query
from transformers import T5ForConditionalGeneration

from evaluate import fingerprint, is_ordered, run
from gat_model import GATText2Cypher
from model_utils import load_tokenizer
from neo4j_utils import DB_NAMES
from text2cypher_data import encode, generate

ROOT = Path(__file__).resolve().parent.parent
DISPLAY_ROWS = 200
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)  # "label does not exist" warnings for wrong guesses


def load_model(run, device):
    tok = load_tokenizer()
    best = ROOT / "models" / run / "best"
    gat = (best / "gat.pt").exists()
    if gat:
        model = GATText2Cypher.load(best, tok)
    else:
        model = T5ForConditionalGeneration.from_pretrained(best, dtype=torch.float32)
    return model.to(device).eval(), tok, gat


def candidates(model, tok, gat, question, graph, device, beams=4):
    """Beam candidates for one question, best first."""
    ex = encode(tok, [{"id": 0, "question": question, "graph": graph}], with_labels=False, with_graph=gat)
    out = generate(model, tok, ex, device, batch_size=1, num_beams=beams, num_return=beams)[0]
    return out if isinstance(out, list) else [out]


def _plain(v):
    """Neo4j values -> things a dataframe can show."""
    if isinstance(v, neo4j.graph.Node):
        return {"labels": sorted(v.labels), **dict(v)}
    if isinstance(v, neo4j.graph.Relationship):
        return {"type": v.type, **dict(v)}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def run_query(driver, cypher, graph, limit=DISPLAY_ROWS):
    """Returns (columns, rows, error); read-only, 30 s timeout."""
    try:
        with driver.session(database=DB_NAMES[graph], default_access_mode=neo4j.READ_ACCESS) as s:
            result = s.run(Query(cypher, timeout=30))
            cols, rows = list(result.keys()), []
            for rec in result:
                rows.append([_plain(v) for v in rec.values()])
                if len(rows) >= limit:
                    break
            return cols, rows, None
    except Exception as e:
        return [], [], f"{type(e).__name__}: {str(e)[:300]}"


def answer(driver, cands, graph):
    """Execution-guided answer in one pass: run beams best-first and keep the first that returns rows (else the
    first that runs), without running the chosen query twice. Returns (index, columns, rows, error)."""
    fallback = None
    for k, q in enumerate(cands):
        cols, rows, err = run_query(driver, q, graph)
        if err and "TimedOut" in err:  # a slow but plausible query: do not fall back to a weaker candidate
            return k, [], [], err
        if err:
            continue
        if rows:
            return k, cols, rows, None
        fallback = fallback or (k, cols, rows, None)
    if fallback:
        return fallback
    return 0, [], [], run_query(driver, cands[0], graph, limit=1)[2]


def same_result(driver, cypher, gold, graph):
    """True when the query returns the same result as the reference query (the execution-accuracy test);
    None when the reference itself fails (e.g. a timeout on a cold cache), since nothing can be compared."""
    db = DB_NAMES[graph]
    gold_rows, gold_err = run(driver, gold, db)
    if gold_err:
        return None
    rows, err = run(driver, cypher, db)
    if err:
        return False
    ordered = is_ordered(gold)
    return fingerprint(rows, ordered) == fingerprint(gold_rows, ordered)


def choose(driver, cands, graph):
    """Execution-guided choice: first beam that runs and returns rows, else first that runs, else the top beam.
    Returns (index, [status per tried beam])."""
    status, first_valid = [], None
    for k, q in enumerate(cands):
        cols, rows, err = run_query(driver, q, graph, limit=1)
        if err:
            status.append("error")
            continue
        if rows:
            status.append("returns rows")
            return k, status
        status.append("runs, no rows")
        first_valid = k if first_valid is None else first_valid
    return (first_valid if first_valid is not None else 0), status
