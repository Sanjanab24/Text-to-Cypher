"""Score the baseline and GAT models on the financial test set in src/aml_questions.py.

For each question and model it records the top beam and the execution-guided choice, and whether each
returns the same result as the reference query. Writes results/aml_check.json (read by the app's Results page).
Usage (Neo4j running): .venv\\Scripts\\python.exe src\\check_aml.py [--verify] [--baseline RUN --gat RUN --out NAME]
  --verify  only run the reference queries and report rows and time (no models)
"""
import argparse
import json
import time
from pathlib import Path

import torch

from aml_questions import AML_QUESTIONS
from evaluate import fingerprint, is_ordered, run
from inference import candidates, choose, load_model
from neo4j_utils import get_driver

ROOT = Path(__file__).resolve().parent.parent


def gold_results(driver):
    """Reference fingerprint per question; each reference runs once (retried once, for a cold cache)."""
    out = []
    for item in AML_QUESTIONS:
        t0 = time.time()
        rows, err = run(driver, item["cypher"], "aml")
        if err:
            rows, err = run(driver, item["cypher"], "aml")
        out.append({"rows": rows, "err": err, "seconds": round(time.time() - t0, 1),
                    "hash": None if err else fingerprint(rows, is_ordered(item["cypher"]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--baseline", default="baseline_t5")
    ap.add_argument("--gat", default="gat_t5")
    ap.add_argument("--out", default="aml_check", help="writes results/<out>.json")
    args = ap.parse_args()
    driver = get_driver()
    gold = gold_results(driver)
    if args.verify:
        for item, g in zip(AML_QUESTIONS, gold):
            print(f"{g['seconds']:5.1f}s {'ERROR ' + g['err'] if g['err'] else len(g['rows'])} rows | "
                  f"{item['question']} | {g['rows'][:3] if g['rows'] else ''}")
        return

    def correct(cypher, item, g):
        rows, err = run(driver, cypher, "aml")
        return err is None and fingerprint(rows, is_ordered(item["cypher"])) == g["hash"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = [{"question": i["question"], "gold": i["cypher"], "category": i["category"],
            "paraphrase_of": i["paraphrase_of"]} for i in AML_QUESTIONS]
    for key, run_name in [("baseline", args.baseline), ("gat", args.gat)]:
        model, tok, gat = load_model(run_name, device)
        for r, item, g in zip(out, AML_QUESTIONS, gold):
            cands = candidates(model, tok, gat, r["question"], "aml", device)
            k, _ = choose(driver, cands, "aml")
            exec_top = correct(cands[0], item, g)
            exec_guided = exec_top if k == 0 else correct(cands[k], item, g)
            r[key] = {"top": cands[0], "exec_top": exec_top, "guided": cands[k], "exec_guided": exec_guided}
            print(key, exec_top, exec_guided, "|", r["question"], flush=True)
        del model
        torch.cuda.empty_cache()
    json.dump(out, open(ROOT / "results" / f"{args.out}.json", "w", encoding="utf-8"), indent=1)
    for key in ["baseline", "gat"]:
        for mode in ["exec_top", "exec_guided"]:
            print(key, mode, sum(r[key][mode] for r in out), "/", len(out))


if __name__ == "__main__":
    main()
