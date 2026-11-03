"""Build the Mind the Query splits and the schema file used by every model.

Splits (by graph, so the unseen-schema tests never share a schema with training):
  train        TRAIN_GRAPHS, 90% of automatically validated pairs
  test_seen    TRAIN_GRAPHS, held-out 10% (i.i.d.: same schemas as training)
  val          VAL_GRAPHS, all pairs (unseen schema; for model selection)
  test_unseen  TEST_GRAPHS, all pairs (unseen schemas; main zero-shot result)
Pairs a human marked logically wrong are dropped; pairs a human marked correct get verified=True.
Duplicate questions are removed, and seen-test pairs whose Cypher also appears in train are dropped.

Outputs: data/processed/mtq/{split}.jsonl and data/processed/schemas.json
Usage: .venv\\Scripts\\python.exe src\\prepare_mtq.py   (Neo4j must be running, for the aml schema)
"""
import glob
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from neo4j_utils import DB_NAMES, extract_schema, get_driver

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/Mind-the-Query"
OUT = ROOT / "data/processed"

TRAIN_GRAPHS = ["legis_graph", "covid", "pole", "twitter_trolls", "star_wars", "gdsc", "osm"]
VAL_GRAPHS = ["er"]
TEST_GRAPHS = ["wwc", "healthcare", "bloom50"]
SEEN_TEST_FRACTION = 0.10
SEED = 42


def norm_cypher(q):
    return q.strip().rstrip(";").strip()


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def load_json(pattern):
    return [r for f in sorted(glob.glob(str(RAW / pattern))) for r in json.load(open(f, encoding="utf-8"))]


def drop_unparseable(items):
    """Keep pairs whose Cypher passes EXPLAIN (parsed and planned, not run). Graphs without a database are kept."""
    driver = get_driver()

    def parses(p):
        if not p["database"]:
            return True
        try:
            with driver.session(database=p["database"]) as s:
                s.run("EXPLAIN " + p["cypher"]).consume()
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(8) as ex:
        keep = list(ex.map(parses, items))
    driver.close()
    return [p for p, k in zip(items, keep) if k]


def main():
    auto = load_json("Automated_Validated_Datasets/*/*_all_passed.json")
    manual = load_json("Manually_Validated_Datasets/*/*.json")
    wrong = {r["unique_id"] for r in manual if str(r.get("logical [0/1]")) == "0"}
    verified = {r["unique_id"] for r in manual if str(r.get("logical [0/1]")) == "1"}

    # human-verified pairs that never went through the automated pipeline are added too
    auto_ids = {r["unique_id"] for r in auto}
    extra = [r for r in manual if r["unique_id"] in verified and r["unique_id"] not in auto_ids]

    seen, pairs, dropped = set(), [], Counter()
    for r in auto + extra:
        if r["unique_id"] in wrong:
            dropped["human-marked wrong"] += 1
            continue
        key = (r["source_dataset"], norm_text(r["NL Question"]))
        if key in seen:
            dropped["duplicate question"] += 1
            continue
        seen.add(key)
        pairs.append({
            "id": r["unique_id"],
            "graph": r["source_dataset"],
            "database": DB_NAMES[r["source_dataset"]],
            "category": r["original_file"].replace(".json", ""),
            "question": r["NL Question"].strip(),
            "cypher": norm_cypher(r["Cypher"]),
            "verified": r["unique_id"] in verified,
        })

    rng = random.Random(SEED)
    splits = {"train": [], "test_seen": [], "val": [], "test_unseen": []}
    for g in TRAIN_GRAPHS:
        by_cat = {}
        for p in pairs:
            if p["graph"] == g:
                by_cat.setdefault(p["category"], []).append(p)
        for items in by_cat.values():  # stratify the seen-test sample by category
            rng.shuffle(items)
            # graphs without a Neo4j database cannot be executed, so they are training-only
            k = round(len(items) * SEEN_TEST_FRACTION) if DB_NAMES[g] else 0
            splits["test_seen"] += items[:k]
            splits["train"] += items[k:]
    train_cyphers = {(p["graph"], norm_text(p["cypher"])) for p in splits["train"]}
    before = len(splits["test_seen"])
    splits["test_seen"] = [p for p in splits["test_seen"] if (p["graph"], norm_text(p["cypher"])) not in train_cyphers]
    dropped["seen-test Cypher also in train"] = before - len(splits["test_seen"])
    # training on Cypher that does not even parse teaches the model invalid syntax
    before = len(splits["train"])
    splits["train"] = drop_unparseable(splits["train"])
    dropped["train Cypher fails to parse"] = before - len(splits["train"])
    splits["val"] = [p for p in pairs if p["graph"] in VAL_GRAPHS]
    splits["test_unseen"] = [p for p in pairs if p["graph"] in TEST_GRAPHS]

    (OUT / "mtq").mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        for p in items:
            p["split"] = name
        with open(OUT / "mtq" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in items:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    schemas = {k: v for line in open(RAW / "db_schema.jsonl") for k, v in json.loads(line).items()}
    with get_driver() as driver:
        schemas["aml"] = extract_schema(driver, "aml")
    json.dump(schemas, open(OUT / "schemas.json", "w", encoding="utf-8"), indent=1)

    print("dropped:", dict(dropped))
    for name, items in splits.items():
        print(f"{name:12s} {len(items):6d} pairs, verified {sum(p['verified'] for p in items):5d}, "
              f"graphs {sorted({p['graph'] for p in items})}")
    print("categories (test_unseen):", dict(Counter(p["category"] for p in splits["test_unseen"])))


if __name__ == "__main__":
    main()
