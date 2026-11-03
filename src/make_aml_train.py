"""Synthetic financial training pairs for domain adaptation on the IBM AML graph.

Templates (several wordings each) are filled with values sampled from the graph. To keep the financial test set
(src/aml_questions.py) fair:
- the named values it uses (countries, banks, entities, accounts, transaction IDs) are never sampled;
- a pair is dropped if its question or its query (ignoring aliases and spacing) equals a test question or query.
Writes data/processed/aml_train.jsonl in the same format as the Mind the Query splits.
Usage (Neo4j running): .venv\\Scripts\\python.exe src\\make_aml_train.py [--per_template 30]
"""
import argparse
import json
import random
import re
from pathlib import Path

from aml_questions import AML_QUESTIONS
from evaluate import run
from neo4j_utils import get_driver

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/processed/aml_train.jsonl"
SR, SA, EV, CR, CA = ("Simple_Retrieval", "Simple_Aggregation", "Evaluation_query", "Complex_Retrieval",
                      "Complex_Aggregation")

# named values used by the test set
HELD_OUT = {"countries": {"Portugal", "Switzerland", "Germany"},
            "banks": {"Canada Bank #27", "Portugal Bank #4507"},
            "entities": {"Sole Proprietorship #50438", "Corporation #34854"},
            "accounts": {"80B779D80", "100428660"},
            "transactions": {2}}

AMOUNTS = [1000, 5000, 10000, 50000, 100000, 250000, 500000, 1000000, 5000000, 20000000, 50000000]
DAYS = [f"2022-09-{d:02d}" for d in range(1, 19)]
T = "MATCH ()-[t:TRANSFERRED_TO]->()"


def fmt_amount(x):
    words = {1000: "1,000", 5000: "5,000", 10000: "10,000", 50000: "50,000", 100000: "100,000",
             250000: "250,000", 500000: "500,000", 1000000: "1 million", 5000000: "5 million",
             20000000: "20 million", 50000000: "50 million"}
    return words[x]


def templates(v):
    """Each template: (category, [wordings], cypher, slot sampler). Wordings use {slots}."""
    c, f, p, cur, rc = v["currencies"], v["formats"], v["patterns"], v["currencies"], v["currencies"]
    pick = random.choice
    return [
        # --- transfer properties (relationship) ---
        (SA, ["How many transfers were made with the {fmt} payment format?", "Count the transfers paid by {fmt}.",
              "What is the number of {fmt} transfers?"],
         T + " WHERE t.payment_format = '{fmt}' RETURN count(t) AS numTransfers", lambda: {"fmt": pick(f)}),
        (SA, ["How many transfers were paid in {cur}?", "Count the transfers whose payment currency is {cur}.",
              "What is the number of transfers paid using {cur}?"],
         T + " WHERE t.payment_currency = '{cur}' RETURN count(t) AS numTransfers", lambda: {"cur": pick(c)}),
        (SA, ["How many transfers were received in {cur}?", "Count the transfers with receiving currency {cur}."],
         T + " WHERE t.receiving_currency = '{cur}' RETURN count(t) AS numTransfers", lambda: {"cur": pick(rc)}),
        (SA, ["What is the total amount paid in {cur}?", "Sum the amount paid over all transfers paid in {cur}.",
              "How much money was paid in {cur} in total?"],
         T + " WHERE t.payment_currency = '{cur}' RETURN sum(t.amount_paid) AS totalPaid", lambda: {"cur": pick(c)}),
        (SA, ["What is the average amount received in transfers received in {cur}?",
              "On average, how much is received per transfer in {cur}?"],
         T + " WHERE t.receiving_currency = '{cur}' RETURN avg(t.amount_received) AS avgReceived",
         lambda: {"cur": pick(rc)}),
        (SA, ["What is the largest amount paid in a {fmt} transfer?", "Find the maximum amount paid by {fmt}."],
         T + " WHERE t.payment_format = '{fmt}' RETURN max(t.amount_paid) AS maxPaid", lambda: {"fmt": pick(f)}),
        (SA, ["What is the smallest amount paid in {cur}?", "Find the minimum amount paid in a {cur} transfer."],
         T + " WHERE t.payment_currency = '{cur}' RETURN min(t.amount_paid) AS minPaid", lambda: {"cur": pick(c)}),
        (SA, ["How many laundering transfers were paid in {cur}?",
              "Count the money laundering transfers whose payment currency is {cur}."],
         T + " WHERE t.is_laundering AND t.payment_currency = '{cur}' RETURN count(t) AS numLaundering",
         lambda: {"cur": pick(c)}),
        (SA, ["How many laundering transfers used the {fmt} payment format?",
              "Count the {fmt} transfers that are flagged as laundering."],
         T + " WHERE t.is_laundering AND t.payment_format = '{fmt}' RETURN count(t) AS numLaundering",
         lambda: {"fmt": pick(f)}),
        (SA, ["How many transfers belong to the '{pat}' laundering pattern?",
              "Count the transfers in the {pat} laundering pattern."],
         T + " WHERE t.laundering_pattern = '{pat}' RETURN count(t) AS numTransfers", lambda: {"pat": pick(p)}),
        (SA, ["What is the total amount paid in '{pat}' laundering transfers?",
              "How much was paid in total in transfers of the {pat} pattern?"],
         T + " WHERE t.laundering_pattern = '{pat}' RETURN sum(t.amount_paid) AS totalPaid", lambda: {"pat": pick(p)}),
        (SA, ["How many transfers were made on {day}?", "Count the transfers with a timestamp on {day}."],
         T + " WHERE date(t.timestamp) = date('{day}') RETURN count(t) AS numTransfers", lambda: {"day": pick(DAYS)}),
        (SA, ["How many laundering transfers happened on {day}?"],
         T + " WHERE t.is_laundering AND date(t.timestamp) = date('{day}') RETURN count(t) AS numLaundering",
         lambda: {"day": pick(DAYS)}),
        (SA, ["How many transfers paid more than {amt_w}?", "Count the transfers with an amount paid above {amt_w}."],
         T + " WHERE t.amount_paid > {amt} RETURN count(t) AS numTransfers",
         lambda: (lambda a: {"amt": a, "amt_w": fmt_amount(a)})(pick(AMOUNTS))),
        (SA, ["How many transfers were sent by an account to itself?", "Count the self-transfers.",
              "How many transfers have the same sender and receiver account?"],
         "MATCH (a:Account)-[t:TRANSFERRED_TO]->(a) RETURN count(t) AS numSelfTransfers", lambda: {}),
        (SA, ["How many transfers were paid in one currency and received in another?",
              "Count the transfers whose payment and receiving currencies differ."],
         T + " WHERE t.payment_currency <> t.receiving_currency RETURN count(t) AS numTransfers", lambda: {}),
        (SA, ["How many laundering attempts are there?", "Count the distinct laundering attempts."],
         T + " WHERE t.attempt_id IS NOT NULL RETURN count(DISTINCT t.attempt_id) AS numAttempts", lambda: {}),
        (SR, ["List the transaction IDs of '{pat}' laundering transfers paid in {cur}.",
              "Which {pat} transfers were paid in {cur}? Return the transaction IDs."],
         T + " WHERE t.laundering_pattern = '{pat}' AND t.payment_currency = '{cur}' RETURN t.transaction_id",
         lambda: {"pat": pick(p), "cur": pick(["US Dollar", "Euro", "UK Pound", "Yuan", "Rupee", "Yen"])}),
        (SR, ["Find the transaction IDs of transfers paid by {fmt} with an amount paid above {amt_w}.",
              "Which {fmt} transfers paid more than {amt_w}? Return the transaction IDs."],
         T + " WHERE t.payment_format = '{fmt}' AND t.amount_paid > {amt} RETURN t.transaction_id",
         lambda: (lambda a: {"fmt": pick(f), "amt": a, "amt_w": fmt_amount(a)})(pick(AMOUNTS[6:]))),
        (SR, ["Which payment formats are used by laundering transfers?",
              "List the distinct payment formats of transfers flagged as laundering."],
         T + " WHERE t.is_laundering RETURN DISTINCT t.payment_format", lambda: {}),
        (SR, ["Which currencies are transfers received in?", "List the distinct receiving currencies."],
         T + " RETURN DISTINCT t.receiving_currency", lambda: {}),
        (SR, ["Which laundering patterns use the {fmt} payment format?",
              "List the distinct laundering patterns of {fmt} transfers."],
         T + " WHERE t.payment_format = '{fmt}' AND t.laundering_pattern IS NOT NULL "
             "RETURN DISTINCT t.laundering_pattern", lambda: {"fmt": pick(f)}),
        (EV, ["What payment currency was used in the transfer with transaction ID {tid}?",
              "Which currency was transfer {tid} paid in?"],
         "MATCH ()-[t:TRANSFERRED_TO {{transaction_id: {tid}}}]->() RETURN t.payment_currency",
         lambda: {"tid": pick(v["tids"])}),
        (EV, ["How much was received in the transfer with transaction ID {tid}?",
              "What amount was received in transfer {tid}?"],
         "MATCH ()-[t:TRANSFERRED_TO {{transaction_id: {tid}}}]->() RETURN t.amount_received",
         lambda: {"tid": pick(v["tids"])}),
        (EV, ["Is the transfer with transaction ID {tid} marked as laundering?",
              "Was transfer {tid} a money laundering transfer?"],
         "MATCH ()-[t:TRANSFERRED_TO {{transaction_id: {tid}}}]->() RETURN t.is_laundering",
         lambda: {"tid": pick(v["tids"])}),
        (EV, ["Which laundering pattern does the transfer with transaction ID {tid} belong to?",
              "What is the laundering pattern of transfer {tid}?"],
         "MATCH ()-[t:TRANSFERRED_TO {{transaction_id: {tid}}}]->() RETURN t.laundering_pattern",
         lambda: {"tid": pick(v["pattern_tids"])}),
        (EV, ["When was the transfer with transaction ID {tid} made?", "What is the timestamp of transfer {tid}?"],
         "MATCH ()-[t:TRANSFERRED_TO {{transaction_id: {tid}}}]->() RETURN t.timestamp",
         lambda: {"tid": pick(v["tids"])}),
        # --- transfer properties along paths ---
        (CR, ["Which accounts sent a '{pat}' laundering transfer? Return the distinct account numbers.",
              "Find the account numbers of senders of {pat} transfers."],
         "MATCH (a:Account)-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern = '{pat}' "
         "RETURN DISTINCT a.account_number", lambda: {"pat": pick(p)}),
        (CR, ["Which accounts received a '{pat}' laundering transfer? Return the distinct account numbers."],
         "MATCH ()-[t:TRANSFERRED_TO]->(b:Account) WHERE t.laundering_pattern = '{pat}' "
         "RETURN DISTINCT b.account_number", lambda: {"pat": pick(p)}),
        (CR, ["Which banks hold accounts that received '{pat}' laundering transfers? Return the distinct bank names."],
         "MATCH ()-[t:TRANSFERRED_TO]->(:Account)-[:HELD_AT]->(b:Bank) WHERE t.laundering_pattern = '{pat}' "
         "RETURN DISTINCT b.name", lambda: {"pat": pick(p)}),
        (CR, ["Which entities own an account that received a '{pat}' laundering transfer? Return the distinct names."],
         "MATCH (e:Entity)-[:OWNS]->(:Account)<-[t:TRANSFERRED_TO]-() WHERE t.laundering_pattern = '{pat}' "
         "RETURN DISTINCT e.name", lambda: {"pat": pick(p)}),
        (CR, ["Which accounts held at banks in {country} sent laundering transfers? Return the distinct account numbers.",
              "Find accounts at {country} banks that sent money flagged as laundering."],
         "MATCH (:Bank {{country: '{country}'}})<-[:HELD_AT]-(a:Account)-[t:TRANSFERRED_TO]->() WHERE t.is_laundering "
         "RETURN DISTINCT a.account_number", lambda: {"country": pick(v["countries"])}),
        (CR, ["Which accounts did the account '{acct}' send money to by {fmt}? Return the distinct account numbers."],
         "MATCH (:Account {{account_number: '{acct}'}})-[t:TRANSFERRED_TO]->(b:Account) WHERE t.payment_format = '{fmt}' "
         "RETURN DISTINCT b.account_number", lambda: {"acct": pick(v["accounts"]), "fmt": pick(f)}),
        (CR, ["Which accounts sent money to the account '{acct}'? Return the distinct account numbers.",
              "List the senders of transfers to account '{acct}'."],
         "MATCH (a:Account)-[:TRANSFERRED_TO]->(:Account {{account_number: '{acct}'}}) RETURN DISTINCT a.account_number",
         lambda: {"acct": pick(v["accounts"])}),
        (CR, ["List the transaction IDs of laundering transfers sent from banks in {country}.",
              "Which laundering transfers were sent by accounts at {country} banks? Return transaction IDs."],
         "MATCH (:Bank {{country: '{country}'}})<-[:HELD_AT]-(:Account)-[t:TRANSFERRED_TO]->() WHERE t.is_laundering "
         "RETURN t.transaction_id", lambda: {"country": pick(v["countries"])}),
        (SA, ["How much did the account '{acct}' pay in total?", "What is the total amount paid by account '{acct}'?"],
         "MATCH (:Account {{account_number: '{acct}'}})-[t:TRANSFERRED_TO]->() RETURN sum(t.amount_paid) AS totalPaid",
         lambda: {"acct": pick(v["accounts"])}),
        (SA, ["How many transfers did the account '{acct}' receive?", "Count the incoming transfers of account '{acct}'."],
         "MATCH ()-[t:TRANSFERRED_TO]->(:Account {{account_number: '{acct}'}}) RETURN count(t) AS numReceived",
         lambda: {"acct": pick(v["accounts"])}),
        (SA, ["How many laundering transfers were received by accounts at the bank '{bank}'?"],
         "MATCH ()-[t:TRANSFERRED_TO]->(:Account)-[:HELD_AT]->(:Bank {{name: '{bank}'}}) WHERE t.is_laundering "
         "RETURN count(t) AS numLaundering", lambda: {"bank": pick(v["banks"])}),
        # --- grouping over transfer properties ---
        (CA, ["For each payment currency, how many transfers are there?", "Count transfers per payment currency.",
              "Break down the number of transfers by payment currency."],
         T + " RETURN t.payment_currency AS currency, count(t) AS numTransfers", lambda: {}),
        (CA, ["For each payment format, what is the total amount paid?",
              "What is the total amount paid per payment format?"],
         T + " RETURN t.payment_format AS paymentFormat, sum(t.amount_paid) AS totalPaid", lambda: {}),
        (CA, ["For each receiving currency, what is the average amount received?",
              "Average amount received per receiving currency."],
         T + " RETURN t.receiving_currency AS currency, avg(t.amount_received) AS avgReceived", lambda: {}),
        (CA, ["How many laundering transfers were paid in each currency?",
              "Break down laundering transfers by payment currency."],
         T + " WHERE t.is_laundering RETURN t.payment_currency AS currency, count(t) AS numLaundering", lambda: {}),
        (CA, ["For {fmt} transfers, how many are there per payment currency?"],
         T + " WHERE t.payment_format = '{fmt}' RETURN t.payment_currency AS currency, count(t) AS numTransfers",
         lambda: {"fmt": pick(f)}),
        (CA, ["What is the total amount paid in each laundering pattern?",
              "Sum the amount paid per laundering pattern."],
         T + " WHERE t.laundering_pattern IS NOT NULL RETURN t.laundering_pattern AS pattern, "
             "sum(t.amount_paid) AS totalPaid", lambda: {}),
        (CA, ["How many laundering transfers happened on each day? Order by date.",
              "Count laundering transfers per day, ordered by day."],
         T + " WHERE t.is_laundering RETURN date(t.timestamp) AS day, count(t) AS numLaundering ORDER BY day",
         lambda: {}),
        (CA, ["Which {n} payment currencies have the highest total amount paid?",
              "Top {n} currencies by total amount paid."],
         T + " RETURN t.payment_currency AS currency, sum(t.amount_paid) AS totalPaid ORDER BY totalPaid DESC LIMIT {n}",
         lambda: {"n": pick([2, 3, 4])}),
        (CA, ["Which payment format has the most laundering transfers?",
              "What is the most common payment format among laundering transfers?"],
         T + " WHERE t.is_laundering RETURN t.payment_format AS paymentFormat, count(t) AS n ORDER BY n DESC LIMIT 1",
         lambda: {}),
        (CA, ["Which {n} accounts received the most transfers?", "Top {n} accounts by number of incoming transfers."],
         "MATCH ()-[t:TRANSFERRED_TO]->(b:Account) WITH b, count(t) AS n ORDER BY n DESC LIMIT {n} "
         "RETURN b.account_number, n", lambda: {"n": pick([3, 5, 10])}),
        (CA, ["For each bank country, how many laundering transfers were sent from its banks?"],
         "MATCH (b:Bank)<-[:HELD_AT]-(:Account)-[t:TRANSFERRED_TO]->() WHERE t.is_laundering AND b.country IS NOT NULL "
         "RETURN b.country AS country, count(t) AS numLaundering", lambda: {}),
        (CA, ["For each entity type, how many laundering transfers did their accounts send?"],
         "MATCH (e:Entity)-[:OWNS]->(:Account)-[t:TRANSFERRED_TO]->() WHERE t.is_laundering "
         "RETURN e.entity_type AS entityType, count(t) AS numLaundering", lambda: {}),
        # --- nodes and paths (keeps these skills) ---
        (SR, ["List the names of banks in {country}.", "Which banks are located in {country}? Return their names."],
         "MATCH (b:Bank {{country: '{country}'}}) RETURN b.name", lambda: {"country": pick(v["countries"])}),
        (SA, ["How many banks are there in {country}?", "Count the banks located in {country}."],
         "MATCH (b:Bank {{country: '{country}'}}) RETURN count(b) AS numBanks",
         lambda: {"country": pick(v["countries"])}),
        (SA, ["How many entities are of type '{etype}'?", "Count the {etype} entities."],
         "MATCH (e:Entity {{entity_type: '{etype}'}}) RETURN count(e) AS numEntities",
         lambda: {"etype": pick(["Sole Proprietorship", "Partnership", "Individual"])}),
        (SA, ["How many accounts are held at banks in {country}?"],
         "MATCH (a:Account)-[:HELD_AT]->(:Bank {{country: '{country}'}}) RETURN count(a) AS numAccounts",
         lambda: {"country": pick(v["countries"])}),
        (SA, ["How many accounts are held at crypto banks?", "Count the accounts at crypto banks."],
         "MATCH (a:Account)-[:HELD_AT]->(b:Bank) WHERE b.is_crypto = 'true' RETURN count(a) AS numAccounts",
         lambda: {}),
        (EV, ["Who owns the account '{acct}'?", "Which entity owns account number '{acct}'?"],
         "MATCH (e:Entity)-[:OWNS]->(:Account {{account_number: '{acct}'}}) RETURN e.name",
         lambda: {"acct": pick(v["accounts"])}),
        (EV, ["Which bank holds the account '{acct}'?", "At which bank is account '{acct}' held?"],
         "MATCH (:Account {{account_number: '{acct}'}})-[:HELD_AT]->(b:Bank) RETURN b.name",
         lambda: {"acct": pick(v["accounts"])}),
        (EV, ["In which country is the bank '{bank}'?", "What country is '{bank}' located in?"],
         "MATCH (b:Bank {{name: '{bank}'}}) RETURN b.country", lambda: {"bank": pick(v["banks"])}),
        (EV, ["What type of entity is '{ent}'?"], "MATCH (e:Entity {{name: '{ent}'}}) RETURN e.entity_type",
         lambda: {"ent": pick(v["entities"])}),
        (SR, ["Which accounts does '{ent}' own? Return the account numbers.", "List the accounts owned by '{ent}'."],
         "MATCH (:Entity {{name: '{ent}'}})-[:OWNS]->(a:Account) RETURN a.account_number",
         lambda: {"ent": pick(v["entities"])}),
        (CR, ["Which entities own accounts at banks in {country}? Return the distinct entity names."],
         "MATCH (e:Entity)-[:OWNS]->(:Account)-[:HELD_AT]->(:Bank {{country: '{country}'}}) RETURN DISTINCT e.name",
         lambda: {"country": pick(v["countries"])}),
        (CA, ["For each country, how many banks are there? Show the top {n}.", "Top {n} countries by number of banks."],
         "MATCH (b:Bank) WHERE b.country IS NOT NULL RETURN b.country AS country, count(b) AS numBanks "
         "ORDER BY numBanks DESC LIMIT {n}", lambda: {"n": pick([3, 5, 10])}),
        (CA, ["How many entities of each type are there?", "Count entities per entity type."],
         "MATCH (e:Entity) RETURN e.entity_type AS entityType, count(e) AS numEntities", lambda: {}),
    ]


def norm(s):
    s = re.sub(r"\s+as\s+\w+", "", s.lower())
    return re.sub(r"\s+", " ", s).strip().rstrip(";")


def sample_values(driver):
    def col(q):
        rows, err = run(driver, q, "aml")
        assert not err, err
        return [r[0] for r in rows]

    v = {"countries": [c for c in col("MATCH (b:Bank) WHERE b.country IS NOT NULL RETURN DISTINCT b.country")
                       if c not in HELD_OUT["countries"]],
         "currencies": col("MATCH ()-[t:TRANSFERRED_TO]->() RETURN DISTINCT t.payment_currency"),
         "formats": col("MATCH ()-[t:TRANSFERRED_TO]->() RETURN DISTINCT t.payment_format"),
         "patterns": col("MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL "
                         "RETURN DISTINCT t.laundering_pattern"),
         "banks": [b for b in col("MATCH (b:Bank) WITH b SKIP 50 LIMIT 400 RETURN b.name")
                   if b not in HELD_OUT["banks"]],
         "entities": [e for e in col("MATCH (e:Entity) WITH e SKIP 1000 LIMIT 400 RETURN e.name")
                      if e not in HELD_OUT["entities"]],
         "accounts": [a for a in col("MATCH (a:Account)-[:TRANSFERRED_TO]->() WITH DISTINCT a SKIP 2000 LIMIT 400 "
                                     "RETURN a.account_number") if a not in HELD_OUT["accounts"]],
         "pattern_tids": col("MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL "
                             "RETURN t.transaction_id LIMIT 400")}
    v["tids"] = [t for t in random.sample(range(1, 5_000_000), 400) if t not in HELD_OUT["transactions"]]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_template", type=int, default=30)
    args = ap.parse_args()
    random.seed(7)
    v = sample_values(get_driver())
    test_q = {norm(i["question"]) for i in AML_QUESTIONS}
    test_c = {norm(i["cypher"]) for i in AML_QUESTIONS}
    pairs, seen, dropped = [], set(), 0
    for k, (cat, wordings, cypher, slots) in enumerate(templates(v)):
        for _ in range(args.per_template):
            s = slots()
            question, query = random.choice(wordings).format(**s), cypher.format(**s)
            if norm(question) in test_q or norm(query) in test_c:
                dropped += 1
                continue
            if (question, query) in seen:
                continue
            seen.add((question, query))
            pairs.append({"id": f"aml_train_{k}_{len(pairs)}", "graph": "aml", "database": "aml", "category": cat,
                          "question": question, "cypher": query, "verified": False, "split": "aml_train"})
    with open(OUT, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} pairs from {len(templates(v))} templates written to {OUT} "
          f"({dropped} dropped for matching the test set)")


if __name__ == "__main__":
    main()
