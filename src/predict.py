"""Generate Cypher for a split with a trained model and score it.

Writes results/<run>/<split>.predictions.jsonl, <split>.details.jsonl and <split>.report.json.
Progress is saved after every batch: if the run is stopped, the same command resumes it.
Usage: .venv\\Scripts\\python.exe src\\predict.py --run baseline_t5 --split test_unseen [--beams 4] [--sample N]
The GAT model (Phase 4) is detected from models/<run>/best/gat.pt.

--exec_guided: execution-guided decoding. All beams are kept; each is run on Neo4j (best beam first) and the
first that runs and returns rows is chosen, else the first that runs, else the top beam. Gold queries are never used.
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration

from evaluate import run, score_predictions
from gat_model import GATText2Cypher
from neo4j_utils import get_driver
from model_utils import load_tokenizer
from text2cypher_data import encode, generate, load_split

ROOT = Path(__file__).resolve().parent.parent


def choose_by_execution(candidates, databases, workers=8):
    """{id: [beams]} -> {id: chosen query}."""
    driver = get_driver()

    def pick(i):
        first_valid = None
        for q in candidates[i]:
            rows, err = run(driver, q, databases[i])
            if err is None:
                if rows:
                    return i, q
                first_valid = first_valid or q
        return i, first_valid or candidates[i][0]

    with ThreadPoolExecutor(workers) as ex:
        chosen = dict(ex.map(pick, candidates))
    driver.close()
    changed = sum(chosen[i] != candidates[i][0] for i in chosen)
    print(f"execution-guided: changed the top beam for {changed} of {len(chosen)} questions")
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline_t5")
    ap.add_argument("--split", default="test_unseen")
    ap.add_argument("--beams", type=int, default=4)
    # batch 16 x 4 beams on 1,024-token inputs overflows 8 GB and spills into slow shared memory
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="fixed random sample of N questions (seed 0)")
    ap.add_argument("--exec_guided", action="store_true", help="pick among beams by executing them")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer()
    best = ROOT / "models" / args.run / "best"
    gat = (best / "gat.pt").exists()
    if gat:
        model = GATText2Cypher.load(best, tok).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(best, dtype=torch.float32).to(device)
    pairs = [p for p in load_split(args.split) if p["database"]]  # only graphs we can execute
    if args.limit:
        pairs = pairs[:args.limit]
    if args.sample:  # same ids for every model, so runs stay comparable
        pairs = random.Random(0).sample(pairs, min(args.sample, len(pairs)))

    # predictions are appended to a progress file after every batch, so a stopped run resumes
    out = ROOT / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    name = args.split + (f".sample{args.sample}" if args.sample else "") + (".eg" if args.exec_guided else "")
    progress = out / f"{name}.predictions.partial.jsonl"
    preds = {}
    if progress.exists():
        for line in open(progress, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                preds[r["id"]] = r["prediction"]
        print(f"resuming: {len(preds)} of {len(pairs)} already generated")
    todo = [p for p in pairs if p["id"] not in preds]

    with open(progress, "a", encoding="utf-8") as f:
        def save(batch):
            for i, q in batch.items():
                f.write(json.dumps({"id": i, "prediction": q}, ensure_ascii=False) + "\n")
            f.flush()
        preds.update(generate(model, tok, encode(tok, todo, with_labels=False, with_graph=gat), device,
                              batch_size=args.batch_size, num_beams=args.beams, on_batch=save,
                              num_return=args.beams if args.exec_guided else 1))
    if args.exec_guided:
        json.dump(preds, open(out / f"{name}.candidates.json", "w", encoding="utf-8"), ensure_ascii=False)
        preds = choose_by_execution(preds, {p["id"]: p["database"] for p in pairs})

    with open(out / f"{name}.predictions.jsonl", "w", encoding="utf-8") as f:
        for i, q in preds.items():
            f.write(json.dumps({"id": i, "prediction": q}, ensure_ascii=False) + "\n")
    report, details = score_predictions(args.split, preds)
    json.dump(report, open(out / f"{name}.report.json", "w"), indent=1)
    with open(out / f"{name}.details.jsonl", "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    progress.unlink()  # finished; the full predictions file replaces it


if __name__ == "__main__":
    main()
