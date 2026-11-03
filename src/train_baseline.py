"""Phase 3 baseline: fine-tune CodeT5+ 220M on question + serialized schema -> Cypher (no GAT).

Each epoch it scores execution accuracy on a fixed 400-question sample of val (unseen schema, er)
and keeps the best epoch in models/<run>/best. A resumable checkpoint is written every epoch,
so rerunning the same command continues where it stopped.

Usage: .venv\\Scripts\\python.exe src\\train_baseline.py [--run baseline_t5] [--epochs 10] [--limit N]
       .venv\\Scripts\\python.exe src\\train_baseline.py --gat --run gat_t5 --epochs 2   (Phase 4 GAT model)
Domain adaptation (continue from a trained model on extra data plus a replay sample of the original training set):
       .venv\\Scripts\\python.exe src\\train_baseline.py --gat --run gat_aml --init_from gat_t5 \\
           --extra data/processed/aml_train.jsonl --replay 3000 --epochs 2 --lr 5e-5
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup

from evaluate import score_predictions
from gat_model import GATText2Cypher
from model_utils import BACKBONE, load_tokenizer
from text2cypher_data import collate, encode, generate, length_batches, load_split

ROOT = Path(__file__).resolve().parent.parent
VAL_SAMPLE = 400


def val_sample(seed=0):
    """Fixed sample of val questions whose gold query runs and returns rows."""
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/mtq/val.gold_results.jsonl")}
    pool = [p for p in load_split("val") if gold.get(p["id"], {}).get("ok") and gold[p["id"]]["n_rows"] > 0]
    random.Random(seed).shuffle(pool)
    return pool[:VAL_SAMPLE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline_t5")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="stop after this many epochs without val gain")
    ap.add_argument("--limit", type=int, default=0, help="use only N training pairs (smoke test)")
    ap.add_argument("--gat", action="store_true", help="add the GAT schema encoder (Phase 4)")
    ap.add_argument("--init_from", help="continue from models/<run>/best instead of the pretrained backbone")
    ap.add_argument("--extra", help="extra training pairs (jsonl), e.g. synthetic financial data")
    ap.add_argument("--replay", type=int, default=0,
                    help="with --extra: also train on this many original pairs, so earlier skills are kept")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "models" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "results" / args.run / "train_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()
    train_pairs = load_split("train")
    if args.limit:
        train_pairs = random.Random(0).sample(train_pairs, args.limit)
    if args.extra:
        extra = [json.loads(l) for l in open(ROOT / args.extra, encoding="utf-8")]
        train_pairs = random.Random(0).sample(train_pairs, min(args.replay, len(train_pairs))) + extra
        print(f"training on {len(extra)} extra + {len(train_pairs) - len(extra)} replayed pairs")
    train = encode(tok, train_pairs, with_graph=args.gat)
    val_pairs = val_sample()[: (40 if args.limit else VAL_SAMPLE)]
    val = encode(tok, val_pairs, with_labels=False, with_graph=args.gat)

    # the checkpoint is stored in fp16; train fp32 master weights under bf16 autocast
    start = ROOT / "models" / args.init_from / "best" if args.init_from else BACKBONE
    if args.gat:
        model = (GATText2Cypher.load(start, tok) if args.init_from else GATText2Cypher(BACKBONE, tok)).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(start, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps_per_epoch * args.epochs),
                                            steps_per_epoch * args.epochs)

    state_path = out_dir / "last.pt"
    start_epoch, best, stale = 0, -1.0, 0
    if state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best, stale = st["epoch"] + 1, st["best"], st["stale"]
        print(f"resuming after epoch {st['epoch']} (best val exec {best:.2f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        batches = length_batches(train, args.batch_size, shuffle=True, seed=epoch)
        for step, idx in enumerate(batches):
            batch, _ = collate(train, idx, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total, n = total + loss.item() * args.grad_accum, n + 1
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(batches)} loss {total / n:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        preds = generate(model, tok, val, device)
        report, _ = score_predictions("val", preds, quiet=True)
        val_exec = report["overall"]["exec_acc"]
        entry = {"epoch": epoch, "train_loss": round(total / n, 4), "val_exec_acc": val_exec,
                 "val_validity": report["overall"]["validity"], "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(entry), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if val_exec > best:
            best, stale = val_exec, 0
            model.save_pretrained(out_dir / "best")
            tok.save_pretrained(out_dir / "best")
            print(f"new best val exec acc {best:.2f}, saved to {out_dir / 'best'}")
        else:
            stale += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best": best, "stale": stale}, state_path)
        if stale >= args.patience:
            print(f"no val improvement for {stale} epochs, stopping")
            break

    print(f"done. best val exec acc {best:.2f}")


if __name__ == "__main__":
    main()
