# Phase 2: data and evaluation

## Build the data

```powershell
.venv\Scripts\python.exe src\prepare_mtq.py                 # splits + schemas (Neo4j must be running)
foreach ($s in "val", "test_unseen", "test_seen") { .venv\Scripts\python.exe src\evaluate.py gold $s }
```

Outputs in `data/processed/`:

| File | Contents |
|---|---|
| `mtq/{train,test_seen,val,test_unseen}.jsonl` | one pair per line: `id, graph, database, category, question, cypher, verified, split` |
| `mtq/{split}.gold_results.jsonl` | cached gold results (fingerprint, row count, error) used by scoring |
| `schemas.json` | schema per graph (`node_props`, `rel_props`, `relationships`), Mind the Query graphs plus `aml` |

## Splits

Split by graph, so the unseen-schema sets never share a schema with training.

| Split | Graphs | Pairs | Human-verified | Purpose |
|---|---|---|---|---|
| train | legis_graph, covid, pole, twitter_trolls, star_wars, gdsc, osm (90%) | 15,895 | 17 | training |
| test_seen | same graphs except star_wars (10%) | 1,250 | 3 | seen-schema (i.i.d.) score |
| val | er | 3,137 | 1,391 | unseen schema; model selection |
| test_unseen | wwc, healthcare, bloom50 | 7,851 | 2,964 | unseen schemas; main zero-shot score |

Cleaning, in order:
- 31 pairs a human marked logically wrong are dropped; 2,964 + 1,391 human-verified pairs are flagged `verified`
  (including human-verified pairs that were not in the automated set).
- 277 duplicate questions are removed (same graph, same normalised question).
- The seen-test sample is stratified by category; 250 seen-test pairs whose Cypher also appears in train are dropped.
- Every training query passes `EXPLAIN` on its database (0 failed). Star Wars has no database: training only.
- Category coverage of test_unseen: Simple Retrieval 1,566, Simple Aggregation 1,147, Complex Retrieval 1,820,
  Complex Aggregation 1,622, Evaluation query 1,696.

Schemas: Mind the Query graphs use the benchmark's `db_schema.jsonl` (the questions were generated from it).
`aml` is extracted from the live database with `src/neo4j_utils.py:extract_schema`, which produces the same format;
on WWC it matches `db_schema.jsonl` exactly, and on other graphs the live schema is a superset (rare properties).

## Gold query check (Neo4j 2026.10.1)

| Split | Gold queries that run | Return no rows |
|---|---|---|
| val | 3,085 / 3,137 | 244 |
| test_unseen | 7,849 / 7,851 | 692 |
| test_seen | 1,246 / 1,250 | 1 |

All 52 failing val golds are human-reviewed pairs that never passed the benchmark's runtime check (invalid Cypher such
as SQL-style `FROM`); the 4 test_seen failures are 30-second timeouts. Failing golds are excluded from scoring.

## Metrics (`src/evaluate.py`)

- **Execution accuracy** (main): predicted and gold queries return the same rows. Rows are a multiset, or an ordered
  list when the gold has `ORDER BY`; column order and aliases are ignored; floats are rounded to 4 decimals.
- **Exact match**: identical after whitespace and case normalisation.
- **Validity**: the predicted query runs without error (30 s timeout, read-only sessions, at most 10,000 rows).

Each report gives `overall`, `nonempty_gold` (excludes golds with empty results, which any empty wrong answer would
match), `verified_only`, and breakdowns by category and graph.

```powershell
.venv\Scripts\python.exe src\evaluate.py score test_unseen results\predictions.jsonl
```

Predictions file: one `{"id": ..., "prediction": "<cypher>"}` per line.
