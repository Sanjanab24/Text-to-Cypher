"""Build the financial transaction graph from IBM AMLworld (HI-Small) and bulk-import it into Neo4j.

Schema:
  (:Entity {entity_id, name, entity_type})-[:OWNS]->(:Account {account_id, account_number})
  (:Account)-[:HELD_AT]->(:Bank {bank_id, name, country, is_crypto})
  (:Account)-[:TRANSFERRED_TO {transaction_id, timestamp, amount_paid, payment_currency,
                               amount_received, receiving_currency, payment_format,
                               is_laundering, laundering_pattern, attempt_id}]->(:Account)

account_id is "<bank_id>:<account_number>" because a few account numbers repeat across banks.
Bank IDs are compared as integers: the transactions file keeps leading zeros, the accounts file does not.

Usage: .venv\\Scripts\\python.exe src\\build_aml_graph.py   (Neo4j instance must be running)
"""
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/ibm-aml"
OUT = ROOT / "data/processed/aml_import"
DB = "aml"

TRANS_COLS = ["timestamp", "from_bank", "from_acct", "to_bank", "to_acct", "amount_received",
              "receiving_currency", "amount_paid", "payment_currency", "payment_format", "is_laundering"]
# columns that identify a transaction row (used to match the patterns file)
ROW_KEY = TRANS_COLS[:-1]


def bank_int(s):
    return s.astype(int).astype(str)


def parse_patterns():
    """Map each laundering transaction row to its pattern type and attempt number."""
    rows, attempt, pattern = [], 0, None
    for line in open(RAW / "HI-Small_Patterns.txt"):
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            attempt += 1
            pattern = re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z-]+)", line).group(1)
        elif line.startswith("END") or not line:
            pattern = None
        elif pattern:
            rows.append(line.split(",")[:10] + [pattern, attempt])
    p = pd.DataFrame(rows, columns=ROW_KEY + ["laundering_pattern", "attempt_id"])
    p["from_bank"], p["to_bank"] = bank_int(p["from_bank"]), bank_int(p["to_bank"])
    return p.drop_duplicates(ROW_KEY)


def build_csvs():
    OUT.mkdir(parents=True, exist_ok=True)
    acc = pd.read_csv(RAW / "HI-Small_accounts.csv", dtype=str)
    acc.columns = ["bank_name", "bank_id", "account_number", "entity_id", "entity_name"]
    acc["bank_id"] = bank_int(acc["bank_id"])
    acc["account_id"] = acc["bank_id"] + ":" + acc["account_number"]

    # Banks: "<Country> Bank #n" or a US bank named by city
    banks = acc[["bank_id", "bank_name"]].drop_duplicates("bank_id")
    country = banks["bank_name"].str.extract(r"^(.*) Bank #\d+$")[0]
    banks["is_crypto"] = (country == "Crytpo").map({True: "true", False: "false"})  # sic, as in the data
    banks["country"] = country.where(country.notna(), "US").where(country != "Crytpo")
    banks.rename(columns={"bank_id": "bank_id:ID(Bank)", "bank_name": "name"}) \
        .assign(**{":LABEL": "Bank"}).to_csv(OUT / "banks.csv", index=False)

    ent = acc[["entity_id", "entity_name"]].drop_duplicates("entity_id")
    ent["entity_type"] = ent["entity_name"].str.replace(r" #\d+$", "", regex=True)
    ent.rename(columns={"entity_id": "entity_id:ID(Entity)", "entity_name": "name"}) \
        .assign(**{":LABEL": "Entity"}).to_csv(OUT / "entities.csv", index=False)

    acc[["account_id", "account_number"]].rename(columns={"account_id": "account_id:ID(Account)"}) \
        .assign(**{":LABEL": "Account"}).to_csv(OUT / "accounts.csv", index=False)
    pd.DataFrame({":START_ID(Entity)": acc["entity_id"], ":END_ID(Account)": acc["account_id"],
                  ":TYPE": "OWNS"}).to_csv(OUT / "owns.csv", index=False)
    pd.DataFrame({":START_ID(Account)": acc["account_id"], ":END_ID(Bank)": acc["bank_id"],
                  ":TYPE": "HELD_AT"}).to_csv(OUT / "held_at.csv", index=False)

    t = pd.read_csv(RAW / "HI-Small_Trans.csv", dtype=str)
    t.columns = TRANS_COLS
    t["from_bank"], t["to_bank"] = bank_int(t["from_bank"]), bank_int(t["to_bank"])
    t = t.merge(parse_patterns(), on=ROW_KEY, how="left")
    t["attempt_id"] = t["attempt_id"].astype("Int64")
    rel = pd.DataFrame({
        ":START_ID(Account)": t["from_bank"] + ":" + t["from_acct"],
        ":END_ID(Account)": t["to_bank"] + ":" + t["to_acct"],
        ":TYPE": "TRANSFERRED_TO",
        "transaction_id:long": range(1, len(t) + 1),
        "timestamp:localdatetime": pd.to_datetime(t["timestamp"], format="%Y/%m/%d %H:%M").dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "amount_paid:double": t["amount_paid"],
        "payment_currency": t["payment_currency"],
        "amount_received:double": t["amount_received"],
        "receiving_currency": t["receiving_currency"],
        "payment_format": t["payment_format"],
        "is_laundering:boolean": t["is_laundering"].map({"1": "true", "0": "false"}),
        "laundering_pattern": t["laundering_pattern"],
        "attempt_id:int": t["attempt_id"],
    })
    rel.to_csv(OUT / "transfers.csv", index=False)
    print(f"banks {len(banks)}, entities {len(ent)}, accounts {len(acc)}, transfers {len(rel)}, "
          f"laundering {int((t['is_laundering'] == '1').sum())}, matched to a pattern {int(t['laundering_pattern'].notna().sum())}")


def neo4j_import(driver):
    exists = driver.execute_query("SHOW DATABASES YIELD name WHERE name = $n RETURN name",
                                  n=DB, database_="system").records
    if exists:
        driver.execute_query(f"DROP DATABASE `{DB}` IF EXISTS WAIT", database_="system")
    nodes = ["banks.csv", "entities.csv", "accounts.csv"]
    rels = ["owns.csv", "held_at.csv", "transfers.csv"]
    cmd = [str(Path(os.environ["NEO4J_HOME"]) / "bin/neo4j-admin.bat"), "database", "import", "full", DB,
           "--overwrite-destination=true", "--skip-bad-relationships=true", "--ignore-empty-strings=true","--report-file=" + str(OUT / "import.report")]
    cmd += [f"--nodes={OUT / f}" for f in nodes] + [f"--relationships={OUT / f}" for f in rels]
    subprocess.run(cmd, check=True)
    driver.execute_query(f"CREATE DATABASE `{DB}` IF NOT EXISTS WAIT", database_="system")
    for q in ["CREATE INDEX account_id IF NOT EXISTS FOR (a:Account) ON (a.account_id)",
              "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.entity_id)",
              "CREATE INDEX bank_id IF NOT EXISTS FOR (b:Bank) ON (b.bank_id)",
              "CREATE INDEX bank_country IF NOT EXISTS FOR (b:Bank) ON (b.country)",
              "CREATE INDEX transfer_laundering IF NOT EXISTS FOR ()-[t:TRANSFERRED_TO]-() ON (t.is_laundering)"]:
        driver.execute_query(q, database_=DB)


def main():
    load_dotenv(ROOT / ".env")
    if os.getenv("NEO4J_JAVA_HOME"):  # neo4j-admin needs Java 21, not the system Java
        os.environ["JAVA_HOME"] = os.environ["NEO4J_JAVA_HOME"]
    build_csvs()
    with GraphDatabase.driver(os.getenv("NEO4J_URI"),
                              auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        neo4j_import(driver)
        for label in ["Bank", "Entity", "Account"]:
            n = driver.execute_query(f"MATCH (n:{label}) RETURN count(n) AS n", database_=DB).records[0]["n"]
            print(f"[ OK ] {label}: {n}")
        for rel in ["OWNS", "HELD_AT", "TRANSFERRED_TO"]:
            n = driver.execute_query(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n", database_=DB).records[0]["n"]
            print(f"[ OK ] {rel}: {n}")


if __name__ == "__main__":
    main()
