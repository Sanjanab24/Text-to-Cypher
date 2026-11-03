"""Execution-based evaluation for Text-to-Cypher.

Metrics
  execution accuracy  predicted and gold queries return the same result
  exact match         queries are identical after whitespace/case normalisation
  validity            predicted query executes without error

Result comparison: rows are compared as a multiset, or as a list when the gold query has
ORDER BY; inside a row, column order and aliases are ignored; floats are rounded.
Gold queries that fail to run are excluded; scores are also reported without empty-result golds.

Usage
  python src/evaluate.py gold   SPLIT          cache gold results for a split (val, test_unseen, ...)
  python src/evaluate.py score  SPLIT PREDS    score a predictions file ({"id", "prediction"} per line)
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import neo4j
from neo4j import Query

from neo4j_utils import get_driver

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/mtq"
TIMEOUT_S = 30
MAX_ROWS = 10_000
FLOAT_DIGITS = 4


def canon(v):
    """Turn a Neo4j value into a hashable, comparable form."""
    if isinstance(v, float):
        return round(v, FLOAT_DIGITS) + 0.0  # + 0.0 turns -0.0 into 0.0
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, (list, tuple)):
        return tuple(canon(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, canon(x)) for k, x in v.items()))
    if isinstance(v, neo4j.graph.Node):
        return ("node", tuple(sorted(v.labels)), canon(dict(v)))
    if isinstance(v, neo4j.graph.Relationship):
        return ("rel", v.type, canon(dict(v)))
    if isinstance(v, neo4j.graph.Path):
        return ("path", tuple(canon(n) for n in v.nodes))
    return str(v)  # dates, times, points, durations


def canon_row(values):
    # column order and aliases do not matter
    return tuple(sorted((canon(v) for v in values), key=repr))


def run(driver, cypher, database):
    """Execute a query; return (rows, error)."""
    try:
        with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as s:
            result = s.run(Query(cypher, timeout=TIMEOUT_S))
            rows = []
            for rec in result:
                rows.append(canon_row(rec.values()))
                if len(rows) >= MAX_ROWS:
                    break
            return rows, None
    except Exception as e:  # syntax errors, unknown labels, timeouts, write attempts
        return None, f"{type(e).__name__}: {str(e)[:300]}"


def is_ordered(cypher):
    return re.search(r"\border\s+by\b", cypher, re.IGNORECASE) is not None


def fingerprint(rows, ordered):
    rows = rows if ordered else sorted(rows, key=repr)
    return hashlib.sha1(repr(rows).encode()).hexdigest()


def norm_query(q):
    return re.sub(r"\s+", " ", q.strip().rstrip(";")).lower()


def load_split(split):
    return [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]


def gold_path(split):
    return DATA / f"{split}.gold_results.jsonl"


def cache_gold(split, workers=8):
    pairs = [p for p in load_split(split) if p["database"]]
    driver = get_driver()

    def one(p):
        rows, err = run(driver, p["cypher"], p["database"])
        ordered = is_ordered(p["cypher"])
        return {"id": p["id"], "ok": err is None, "error": err, "ordered": ordered,
                "n_rows": None if rows is None else len(rows),
                "hash": None if rows is None else fingerprint(rows, ordered)}

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, pairs))
    driver.close()
    with open(gold_path(split), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in out if r["ok"]]
    print(f"{split}: {len(ok)}/{len(out)} gold queries run; {sum(r['n_rows'] == 0 for r in ok)} return no rows")
    errors = Counter(r["error"].split(":")[0] for r in out if not r["ok"])
    if errors:
        print("  errors:", dict(errors))


def score(split, preds_file, workers=8, quiet=False):
    preds = {r["id"]: r["prediction"] for r in map(json.loads, open(preds_file, encoding="utf-8"))}
    return score_predictions(split, preds, workers, quiet)


def score_predictions(split, preds, workers=8, quiet=False):
    """preds: {id: cypher}. Returns (report, per-example results)."""
    pairs = {p["id"]: p for p in load_split(split)}
    gold = {r["id"]: r for r in map(json.loads, open(gold_path(split), encoding="utf-8")) if r["ok"]}
    ids = [i for i in preds if i in gold]  # gold queries that fail to run are excluded
    driver = get_driver()

    def one(i):
        p, g = pairs[i], gold[i]
        rows, err = run(driver, preds[i], p["database"])
        ex = rows is not None and fingerprint(rows, g["ordered"]) == g["hash"]
        return {"id": i, "category": p["category"], "graph": p["graph"], "verified": p["verified"],
                "gold_empty": g["n_rows"] == 0, "prediction": preds[i], "error": err,
                "valid": err is None, "exec": ex, "exact": norm_query(preds[i]) == norm_query(p["cypher"])}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, ids))
    driver.close()

    def summary(rows):
        n = len(rows) or 1
        return {"n": len(rows), "exec_acc": round(100 * sum(r["exec"] for r in rows) / n, 2),
                "exact_match": round(100 * sum(r["exact"] for r in rows) / n, 2),
                "validity": round(100 * sum(r["valid"] for r in rows) / n, 2)}

    # an empty gold result is matched by any wrong query that also returns nothing, so it is reported apart
    report = {"overall": summary(res),
              "nonempty_gold": summary([r for r in res if not r["gold_empty"]]),
              "verified_only": summary([r for r in res if r["verified"]])}
    for key in ["category", "graph"]:
        groups = defaultdict(list)
        for r in res:
            groups[r[key]].append(r)
        report[f"by_{key}"] = {k: summary(v) for k, v in sorted(groups.items())}
    if not quiet:
        print(json.dumps(report, indent=1))
    return report, res


if __name__ == "__main__":
    if sys.argv[1] == "gold":
        cache_gold(sys.argv[2])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
