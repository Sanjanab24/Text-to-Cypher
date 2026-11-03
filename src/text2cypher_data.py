"""Datasets, batching and generation shared by the baseline and the GAT model."""
import json
import random
from pathlib import Path

import torch

from serialize import model_input

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/mtq"
MAX_SOURCE = 1024  # T5 uses relative positions; long schemas (gdsc, bloom50) are not truncated
MAX_TARGET = 384


def load_split(split):
    return [json.loads(l) for l in open(DATA / f"{split}.jsonl", encoding="utf-8")]


def encode(tok, pairs, with_labels=True, with_graph=False):
    out = []
    for p in pairs:
        ex = {"id": p["id"], "input_ids": tok(model_input(p["question"], p["graph"]),
                                              truncation=True, max_length=MAX_SOURCE).input_ids}
        if with_graph:  # the GAT model also encodes the schema graph
            ex["graph"] = p["graph"]
        if with_labels:
            ex["labels"] = tok(p["cypher"], truncation=True, max_length=MAX_TARGET).input_ids
        out.append(ex)
    return out


def length_batches(examples, batch_size, shuffle, seed=0):
    """Batches of similar input length (less padding); batch order shuffled for training."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    if shuffle:  # sort within random chunks so batches differ between epochs
        rng = random.Random(seed)
        order = list(range(len(examples)))
        rng.shuffle(order)
        chunk = batch_size * 50
        order = [i for c in range(0, len(order), chunk)
                 for i in sorted(order[c:c + chunk], key=lambda i: len(examples[i]["input_ids"]))]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(examples, idx, pad_id, device):
    batch = [examples[i] for i in idx]
    L = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), L), pad_id)
    for j, e in enumerate(batch):
        ids[j, :len(e["input_ids"])] = torch.tensor(e["input_ids"])
    out = {"input_ids": ids.to(device), "attention_mask": (ids != pad_id).long().to(device)}
    if "graph" in batch[0]:
        out["graphs"] = [e["graph"] for e in batch]
    if "labels" in batch[0]:
        T = max(len(e["labels"]) for e in batch)
        lab = torch.full((len(batch), T), -100)  # -100 = ignored by the loss
        for j, e in enumerate(batch):
            lab[j, :len(e["labels"])] = torch.tensor(e["labels"])
        out["labels"] = lab.to(device)
    return out, [e["id"] for e in batch]


@torch.no_grad()
def generate(model, tok, examples, device, batch_size=16, num_beams=1, max_new_tokens=MAX_TARGET,
             on_batch=None, num_return=1):
    """Returns {id: predicted cypher}, or {id: [best beams first]} when num_return > 1.
    on_batch(new) is called after every batch (for saving progress)."""
    model.eval()
    preds = {}
    batches = length_batches(examples, batch_size, shuffle=False)
    for b, idx in enumerate(batches):
        if b % 20 == 0 and len(batches) > 20:
            print(f"generating {len(preds)}/{len(examples)}", flush=True)
        batch, ids = collate([{k: v for k, v in examples[i].items() if k != "labels"} for i in idx],
                             range(len(idx)), tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model.generate(**batch, num_beams=num_beams, num_return_sequences=num_return,
                                 max_new_tokens=max_new_tokens)
        texts = [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
        if num_return > 1:  # returned grouped per input, best beam first
            new = {i: texts[k * num_return:(k + 1) * num_return] for k, i in enumerate(ids)}
        else:
            new = dict(zip(ids, texts))
        preds.update(new)
        if on_batch:
            on_batch(new)
    return preds
