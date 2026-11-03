# Phase 1 setup

## Project layout

```
NLP_SEE/
  data/raw/          downloaded datasets (Mind-the-Query, ieee-cis)
  data/processed/    cleaned splits, extracted schemas
  src/               Python code
  notebooks/         Colab / Jupyter notebooks
  models/            saved checkpoints
  results/           metrics, error analysis
  .venv/             Python 3.11 environment (local)
```

## 1. Python environment (local)

Created with `uv` (Python 3.11; the system Python 3.14 is too new for some ML libraries).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

The cu128 build of PyTorch is needed for RTX 50-series GPUs.

## 2. Neo4j (each teammate does this once)

1. Install **Neo4j Desktop** from https://neo4j.com/download/ (it bundles its own Java, so the system Java version does not matter).
2. Create a local instance, set a password, and start it.
3. Copy `.env.example` to `.env` and put that password in `NEO4J_PASSWORD`.

## 3. Datasets

**Mind the Query**: cloned from https://github.com/endeavorXx/Mind-the-Query into `data/raw/Mind-the-Query`.
Splits: 18,470 train / 2,000 validation / 7,059 test. The test set uses the WWC, HCA and Bloom graphs only;
the other 8 graphs are train/validation, so test results measure transfer to unseen schemas.

**IBM Transactions for Anti-Money Laundering** (financial graph; ordinary Kaggle dataset, no competition;
licence CDLA-Sharing 1.0; https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
We use the HI-Small version (~5M transactions, ~515k accounts).

1. Kaggle (free account) > Settings > API > create a **legacy** API token. Save the downloaded `kaggle.json`
   to `C:\Users\<you>\.kaggle\kaggle.json`.
2. Run (PowerShell):

```powershell
foreach ($f in "HI-Small_Trans.csv", "HI-Small_accounts.csv", "HI-Small_Patterns.txt") {
  .venv\Scripts\kaggle.exe datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f $f -p data\raw\ibm-aml --unzip
}
```

**Load the Mind the Query graphs into Neo4j** (the instance must be running; set `NEO4J_HOME` and
`NEO4J_JAVA_HOME` in `.env` first):

```powershell
.venv\Scripts\python.exe src\load_mtq_graphs.py
```

This creates one database per graph: `wwc`, `twitter-trolls`, `covid`, `healthcare`, `pole`, `bloom50`, `osm`,
`gdsc`, `entity-resolution`, `legis-graph`. Star Wars ships as CSV files and is not loaded by this script yet.

If pandas fails with "An Application Control policy has blocked this file", Windows Smart App Control is
blocking pandas 3.x; `requirements.txt` pins pandas 2.2.3, which loads fine.

**Build the financial graph** (database `aml`, about 30 seconds of import after the CSVs are written):

```powershell
.venv\Scripts\python.exe src\build_aml_graph.py
```

Schema:

```
(:Entity {entity_id, name, entity_type})-[:OWNS]->(:Account {account_id, account_number})-[:HELD_AT]->(:Bank {bank_id, name, country, is_crypto})
(:Account)-[:TRANSFERRED_TO {transaction_id, timestamp, amount_paid, payment_currency, amount_received,
                             receiving_currency, payment_format, is_laundering, laundering_pattern, attempt_id}]->(:Account)
```

Result: 30,470 banks, 166,207 entities, 518,581 accounts, 5,078,345 transfers (5,177 laundering, of which 3,209
belong to one of the 8 named patterns in `HI-Small_Patterns.txt`).

## 4. Check everything

```bash
.venv/Scripts/python src/check_setup.py
```

All five lines should say `[ OK ]`.

## 5. Google Colab (for training)

Open `notebooks/00_colab_setup.ipynb` in Colab, set Runtime > Change runtime type > T4 GPU, and run all cells.
Colab cannot reach a Neo4j instance on your laptop; use it for training only, and run query execution locally.
