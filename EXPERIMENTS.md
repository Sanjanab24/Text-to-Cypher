# Experiments

## Phase 3: text-only baseline (no GAT)

**Model.** CodeT5+ 220M (`Salesforce/codet5p-220m`), an encoder-decoder pretrained on code. The same backbone is
used by the GAT model in Phase 4, so the comparison isolates the effect of the graph encoder.

**Input.** `question: <q> | nodes: Label(prop type, ...) | rel props: TYPE(prop type, ...) | edges: (A|B)-[:TYPE]->(C)`
(`src/serialize.py`).
- Edges are grouped by relationship type (multi-label graphs such as gdsc otherwise list every label combination).
- Labels with more than 12 properties keep the question-matched properties plus others up to 12. This uses only the
  question, never the answer; 0.4% of training pairs and 0% of val/test pairs lose a property the gold query uses.
- Inputs up to 1,024 tokens, so no schema is truncated (T5 uses relative positions).

**Training** (`src/train_baseline.py`). AdamW, lr 1e-4, linear schedule with 5% warmup, weight decay 0.01, batch 8 x
gradient accumulation 2, bf16 autocast over fp32 weights, gradient checkpointing, up to 10 epochs. After each epoch,
execution accuracy on a fixed sample of 400 val questions (er, unseen schema, non-empty gold) selects the best epoch;
training stops after 3 epochs without improvement.

Speed on the RTX 5050 laptop GPU: about 22 minutes per epoch including validation.

**Run** (in your own PowerShell terminal, from `C:\Projects\NLP_SEE`):

```powershell
.venv\Scripts\python.exe src\train_baseline.py --run baseline_t5
```

If it stops (closed window, restart), run the same command again: it resumes from the last finished epoch.
Progress: `results\baseline_t5\train_log.jsonl`. Best model: `models\baseline_t5\best`.

**Evaluate** after training:

```powershell
foreach ($s in "test_seen", "test_unseen") { .venv\Scripts\python.exe src\predict.py --run baseline_t5 --split $s }
```

Reports: `results\baseline_t5\<split>.report.json`; per-question details in `<split>.details.jsonl`.

### Results

| Split | Exec acc | Exec acc (non-empty gold) | Verified only | Exact match | Validity |
|---|---|---|---|---|---|
| test_seen | | | | | |
| test_unseen | | | | | |

## Phase 4: GAT schema encoder (`src/gat_model.py`)

**Idea (schema linking).** The schema becomes a small graph: one node per label, relationship type and property, with
typed edges (label has property, start label -> relationship type -> end label). Each schema node starts from the
embedding of its name and attends over the encoded question, so the elements the question mentions are highlighted.
Two GATv2 layers with edge-type embeddings (relation-aware) pass this along the schema structure. The node vectors
are appended to the T5 encoder output, so the decoder attends to both the question text and the schema graph.
Same backbone, input text, optimiser and selection as the baseline; only the graph encoder is added.

**Train** (2 epochs, the baseline's best epoch was its 2nd; about 25 minutes per epoch):

```powershell
.venv\Scripts\python.exe src\train_baseline.py --gat --run gat_t5 --epochs 2
```

**Evaluate** (test_unseen on a fixed 1,000-question sample to save time; the baseline is compared on the same ids):

```powershell
.venv\Scripts\python.exe src\predict.py --run gat_t5 --split test_seen
.venv\Scripts\python.exe src\predict.py --run gat_t5 --split test_unseen --sample 1000
.venv\Scripts\python.exe src\compare.py test_unseen_1000 baseline_t5:test_unseen gat_t5:test_unseen.sample1000
.venv\Scripts\python.exe src\compare.py test_seen baseline_t5:test_seen gat_t5:test_seen
```

## Execution-guided decoding (`--exec_guided` in `src/predict.py`)

All 4 beams are kept; each is run on Neo4j (best beam first) and the first that runs and returns rows is chosen,
else the first that runs, else the top beam (Wang et al., 2018). Only the database is used, never the gold query.
Applied to both models on the same 1,000 unseen-schema questions:

```powershell
.venv\Scripts\python.exe src\predict.py --run gat_t5 --split test_unseen --sample 1000 --exec_guided
.venv\Scripts\python.exe src\predict.py --run baseline_t5 --split test_unseen --sample 1000 --exec_guided
.venv\Scripts\python.exe src\compare.py test_unseen_1000_eg baseline_t5:test_unseen gat_t5:test_unseen.sample1000 baseline_t5:test_unseen.sample1000.eg gat_t5:test_unseen.sample1000.eg
```

## Financial test set (IBM AML graph, `src/aml_questions.py`, scored by `src/check_aml.py`)

55 hand-written financial questions with reference Cypher (written before looking at model output): 10 simple
retrieval, 11 simple aggregation, 8 evaluation, 10 complex retrieval, 11 complex aggregation, plus 5 paraphrases.
Every reference was run on the graph (non-empty results, no ties at top-N cut-offs). The schema is never seen
in training (zero-shot domain transfer).

| Setup | Correct (of 55) | Nodes and paths (30) | Transfer properties (25) |
|---|---|---|---|
| Baseline | 14 (25.5%) | 14 | 0 |
| Baseline + exec-guided | 15 (27.3%) | 15 | 0 |
| GAT | 17 (30.9%) | 17 | 0 |
| GAT + exec-guided | 19 (34.5%) | 19 | 0 |

GAT + exec-guided vs baseline + exec-guided: 4 questions only GAT gets right, 0 only the baseline.
Paraphrase robustness: same outcome for a question and its rewording in 4 of 5 pairs for both models.

Every question about transfer properties (amount, currency, format, laundering flag on the TRANSFERRED_TO
relationship) fails for all setups: the models write `(t:TRANSFERRED_TO)` or `(t:Transaction)` as a node, or put
the property on an Account. Training graphs mostly keep such facts on nodes, so this is a domain-transfer gap;
fine-tuning on synthetic financial pairs or representing transfers as nodes are the obvious next steps.

### Domain adaptation (`src/make_aml_train.py`, `--init_from` / `--extra` / `--replay` in `src/train_baseline.py`)

981 synthetic financial question-Cypher pairs from 63 templates (several wordings each), filled with values
sampled from the graph. Held out: every named value the test set uses (Germany, Portugal, Switzerland, the
test banks, entities, accounts and transaction IDs), and any pair whose question or query (ignoring aliases)
equals a test item. Each model continued from its best checkpoint for 2 epochs (lr 5e-5) on these pairs plus
3,000 replayed Mind the Query training pairs; validation accuracy on the benchmark graph did not drop
(GAT 63.5 -> 64.75, baseline 65.5 -> 67.0), so the earlier skills were kept.

| Setup | Zero-shot | After domain adaptation | Transfer-property questions (25), after |
|---|---|---|---|
| Baseline | 14 / 55 (25.5%) | 43 / 55 (78.2%) | 17 |
| Baseline + exec-guided | 15 / 55 (27.3%) | **44 / 55 (80.0%)** | 17 |
| GAT | 17 / 55 (30.9%) | 39 / 55 (70.9%) | 16 |
| GAT + exec-guided | 19 / 55 (34.5%) | 40 / 55 (72.7%) | 16 |

After adaptation the baseline is slightly ahead (7 questions only the baseline gets right, 3 only GAT; McNemar
p = 0.34, not significant). Reading: the GAT encoder helps most when the schema is new (zero-shot and unseen
benchmark graphs); once the model is trained on the target schema, schema linking can be learned from the data
directly and the advantage disappears. Remaining errors are mostly multi-hop retrieval (3-4 of 10 correct).
