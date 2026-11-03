"""Backbone loading shared by the baseline and the GAT model."""
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
BACKBONE = "Salesforce/codet5p-220m"
TOKENIZER_DIR = ROOT / "models/codet5p-220m-tokenizer"

SPECIAL = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
           "cls_token": "<s>", "pad_token": "<pad>", "mask_token": "<mask>"}


def load_tokenizer():
    """CodeT5+'s tokenizer_config uses an old AddedToken format that transformers 5 cannot parse,
    so we keep a cleaned copy (same vocab and merges, special tokens as plain strings)."""
    if not (TOKENIZER_DIR / "vocab.json").exists():
        src = Path(snapshot_download(BACKBONE, allow_patterns=["vocab.json", "merges.txt"]))
        TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        for f in ["vocab.json", "merges.txt"]:
            shutil.copy(src / f, TOKENIZER_DIR / f)
        json.dump({"tokenizer_class": "RobertaTokenizer", "model_max_length": 512,
                   "add_prefix_space": False, "errors": "replace", **SPECIAL},
                  open(TOKENIZER_DIR / "tokenizer_config.json", "w"), indent=1)
        json.dump(SPECIAL, open(TOKENIZER_DIR / "special_tokens_map.json", "w"), indent=1)
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)
