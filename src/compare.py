"""Compare runs on exactly the same questions (e.g. the GAT model scored on a test_unseen sample
against the baseline scored on the full split).

Usage: .venv\\Scripts\\python.exe src\\compare.py test_unseen.sample1000 baseline_t5:test_unseen gat_t5:test_unseen.sample1000
Each run is <run>:<details name>; only questions present in every run are scored.
The first argument names the output file, results/compare_<name>.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def details(run, name):
    return {d["id"]: d for d in map(json.loads, open(ROOT / "results" / run / f"{name}.details.jsonl", encoding="utf-8"))}


def summary(rows):
    n = len(rows) or 1
    return {"n": len(rows), "exec_acc": round(100 * sum(r["exec"] for r in rows) / n, 2),
            "exec_nonempty_gold": round(100 * sum(r["exec"] for r in rows if not r["gold_empty"])
                                        / (sum(not r["gold_empty"] for r in rows) or 1), 2),
            "exact_match": round(100 * sum(r["exact"] for r in rows) / n, 2),
            "validity": round(100 * sum(r["valid"] for r in rows) / n, 2)}


def main():
    runs = [a.split(":") for a in sys.argv[2:]]
    loaded = {f"{r}:{n}": details(r, n) for r, n in runs}
    ids = set.intersection(*(set(d) for d in loaded.values()))
    print(f"{len(ids)} shared questions")
    out = {}
    for key, d in loaded.items():
        rows = [d[i] for i in ids]
        out[key] = {"overall": summary(rows),
                    "by_category": {c: summary([r for r in rows if r["category"] == c])
                                    for c in sorted({r["category"] for r in rows})}}
        print(key, json.dumps(out[key]["overall"]))
    json.dump(out, open(ROOT / "results" / f"compare_{sys.argv[1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
