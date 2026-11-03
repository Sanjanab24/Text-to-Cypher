"""Financial test set: questions on the IBM AML graph with hand-written reference Cypher.

Written before looking at any model output on them. Question types follow Mind the Query; paraphrases
reword an earlier question (same reference query) to test robustness to wording.
Used by the demo app (DEMO questions) and scored by src/check_aml.py.

Schema: (:Entity {entity_id, name, entity_type})-[:OWNS]->(:Account {account_id, account_number})
        (:Account)-[:HELD_AT]->(:Bank {bank_id, name, country, is_crypto})
        (:Account)-[:TRANSFERRED_TO {transaction_id, timestamp, amount_paid, payment_currency, amount_received,
                                     receiving_currency, payment_format, is_laundering, laundering_pattern,
                                     attempt_id}]->(:Account)
"""

SR, SA, EV, CR, CA = ("Simple_Retrieval", "Simple_Aggregation", "Evaluation_query", "Complex_Retrieval",
                      "Complex_Aggregation")


def q(question, cypher, category, paraphrase_of=None):
    return {"question": question, "cypher": cypher, "category": category, "paraphrase_of": paraphrase_of}


AML_QUESTIONS = [
    # simple retrieval: one label or one hop, a filter, return attributes
    q("List the names of all crypto banks.", "MATCH (b:Bank) WHERE b.is_crypto = 'true' RETURN b.name", SR),
    q("Show the bank IDs of all banks located in Portugal.",
      "MATCH (b:Bank {country: 'Portugal'}) RETURN b.bank_id", SR),
    q("Which entities are of type 'Individual'? Return their names.",
      "MATCH (e:Entity {entity_type: 'Individual'}) RETURN e.name", SR),
    q("List the names of all banks in Switzerland.", "MATCH (b:Bank {country: 'Switzerland'}) RETURN b.name", SR),
    q("What are the distinct payment formats used in transfers?",
      "MATCH ()-[t:TRANSFERRED_TO]->() RETURN DISTINCT t.payment_format", SR),
    q("Which currencies are used to pay for transfers?",
      "MATCH ()-[t:TRANSFERRED_TO]->() RETURN DISTINCT t.payment_currency", SR),
    q("List the transaction IDs of transfers that belong to the 'CYCLE' laundering pattern.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern = 'CYCLE' RETURN t.transaction_id", SR),
    q("Which account numbers are held at the bank named 'Canada Bank #27'?",
      "MATCH (a:Account)-[:HELD_AT]->(b:Bank {name: 'Canada Bank #27'}) RETURN a.account_number", SR),
    q("Find the transaction IDs of transfers paid in Bitcoin with an amount paid greater than 100.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.payment_currency = 'Bitcoin' AND t.amount_paid > 100 "
      "RETURN t.transaction_id", SR),
    q("What are the distinct laundering patterns in the data?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL RETURN DISTINCT t.laundering_pattern",
      SR),

    # simple aggregation: one count / sum / average / max over a filter
    q("How many banks are there?", "MATCH (b:Bank) RETURN count(b)", SA),
    q("How many entities are there?", "MATCH (e:Entity) RETURN count(e)", SA),
    q("How many accounts are there?", "MATCH (a:Account) RETURN count(a)", SA),
    q("How many transfers are marked as laundering?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering RETURN count(t)", SA),
    q("How many crypto banks are there?", "MATCH (b:Bank) WHERE b.is_crypto = 'true' RETURN count(b)", SA),
    q("What is the total amount paid in transfers marked as laundering?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering RETURN sum(t.amount_paid)", SA),
    q("What is the largest amount paid in a single transfer?",
      "MATCH ()-[t:TRANSFERRED_TO]->() RETURN max(t.amount_paid)", SA),
    q("How many transfers used the 'ACH' payment format?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.payment_format = 'ACH' RETURN count(t)", SA),
    q("What is the average amount paid in transfers paid in Euro?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.payment_currency = 'Euro' RETURN avg(t.amount_paid)", SA),
    q("How many banks are located in Germany?", "MATCH (b:Bank {country: 'Germany'}) RETURN count(b)", SA),
    q("How many entities are corporations?",
      "MATCH (e:Entity {entity_type: 'Corporation'}) RETURN count(e)", SA),

    # evaluation: look up one attribute of one named item
    q("What is the country of the bank named 'Portugal Bank #4507'?",
      "MATCH (b:Bank {name: 'Portugal Bank #4507'}) RETURN b.country", EV),
    q("What type of entity is 'Sole Proprietorship #50438'?",
      "MATCH (e:Entity {name: 'Sole Proprietorship #50438'}) RETURN e.entity_type", EV),
    q("What is the bank ID of 'Canada Bank #27'?", "MATCH (b:Bank {name: 'Canada Bank #27'}) RETURN b.bank_id", EV),
    q("Is the bank 'Canada Bank #27' a crypto bank?",
      "MATCH (b:Bank {name: 'Canada Bank #27'}) RETURN b.is_crypto", EV),
    q("Which entity owns the account with account number '80B779D80'?",
      "MATCH (e:Entity)-[:OWNS]->(a:Account {account_number: '80B779D80'}) RETURN e.name", EV),
    q("At which bank is the account with account number '80B779D80' held?",
      "MATCH (a:Account {account_number: '80B779D80'})-[:HELD_AT]->(b:Bank) RETURN b.name", EV),
    q("What payment format was used in the transfer with transaction ID 2?",
      "MATCH ()-[t:TRANSFERRED_TO {transaction_id: 2}]->() RETURN t.payment_format", EV),
    q("How much was paid in the transfer with transaction ID 2?",
      "MATCH ()-[t:TRANSFERRED_TO {transaction_id: 2}]->() RETURN t.amount_paid", EV),

    # complex retrieval: several hops or conditions
    q("Which entities own accounts at crypto banks? Return the distinct entity names.",
      "MATCH (e:Entity)-[:OWNS]->(:Account)-[:HELD_AT]->(b:Bank) WHERE b.is_crypto = 'true' RETURN DISTINCT e.name",
      CR),
    q("List the transaction IDs of laundering transfers paid in Bitcoin.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering AND t.payment_currency = 'Bitcoin' "
      "RETURN t.transaction_id", CR),
    q("Which banks hold accounts that sent 'FAN-OUT' laundering transfers? Return the distinct bank names.",
      "MATCH (b:Bank)<-[:HELD_AT]-(:Account)-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern = 'FAN-OUT' "
      "RETURN DISTINCT b.name", CR),
    q("Find transfers where the payment currency differs from the receiving currency and more than 10 million "
      "was paid. Return the transaction IDs.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.payment_currency <> t.receiving_currency "
      "AND t.amount_paid > 10000000 RETURN t.transaction_id", CR),
    q("Which accounts held at German banks received a laundering transfer? Return the distinct account numbers.",
      "MATCH ()-[t:TRANSFERRED_TO]->(a:Account)-[:HELD_AT]->(:Bank {country: 'Germany'}) WHERE t.is_laundering "
      "RETURN DISTINCT a.account_number", CR),
    q("Which accounts are held at a bank in Switzerland and owned by an entity of type 'Individual'? "
      "Return the account numbers.",
      "MATCH (:Entity {entity_type: 'Individual'})-[:OWNS]->(a:Account)-[:HELD_AT]->(:Bank {country: 'Switzerland'}) "
      "RETURN a.account_number", CR),
    q("Find the names of entities that own an account which sent a 'CYCLE' laundering transfer.",
      "MATCH (e:Entity)-[:OWNS]->(:Account)-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern = 'CYCLE' "
      "RETURN DISTINCT e.name", CR),
    q("Which laundering transfers went between accounts held at banks in different countries? "
      "Return the transaction IDs.",
      "MATCH (b1:Bank)<-[:HELD_AT]-(:Account)-[t:TRANSFERRED_TO]->(:Account)-[:HELD_AT]->(b2:Bank) "
      "WHERE t.is_laundering AND b1.country <> b2.country RETURN t.transaction_id", CR),
    q("List the transaction IDs of transfers of more than 100 million paid in US Dollar.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.payment_currency = 'US Dollar' AND t.amount_paid > 100000000 "
      "RETURN t.transaction_id", CR),
    q("Which accounts did the account '80B779D80' send money to? Return the distinct account numbers.",
      "MATCH (:Account {account_number: '80B779D80'})-[:TRANSFERRED_TO]->(b:Account) RETURN DISTINCT b.account_number",
      CR),

    # complex aggregation: grouping, ordering, several hops
    q("Which entity owns the most accounts?",
      "MATCH (e:Entity)-[:OWNS]->(a:Account) WITH e, count(a) AS n ORDER BY n DESC LIMIT 1 RETURN e.name, n", CA),
    q("How many accounts does each bank country have? Show the top 5.",
      "MATCH (a:Account)-[:HELD_AT]->(b:Bank) WHERE b.country IS NOT NULL "
      "RETURN b.country, count(a) AS n ORDER BY n DESC LIMIT 5", CA),
    q("Find the top 5 accounts by number of outgoing transfers.",
      "MATCH (a:Account)-[:TRANSFERRED_TO]->(:Account) WITH a, count(*) AS n ORDER BY n DESC LIMIT 5 "
      "RETURN a.account_number, n", CA),
    q("Show the top 5 banks by number of accounts held there.",
      "MATCH (a:Account)-[:HELD_AT]->(b:Bank) WITH b, count(a) AS n ORDER BY n DESC LIMIT 5 RETURN b.name, n", CA),
    q("For each laundering pattern, how many transfers are there?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL "
      "RETURN t.laundering_pattern, count(t)", CA),
    q("How many laundering transfers are there for each payment format?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering RETURN t.payment_format, count(t)", CA),
    q("For each entity type, how many accounts do entities of that type own?",
      "MATCH (e:Entity)-[:OWNS]->(a:Account) RETURN e.entity_type, count(a)", CA),
    q("What is the average amount paid for each payment format, ordered from highest to lowest?",
      "MATCH ()-[t:TRANSFERRED_TO]->() RETURN t.payment_format, avg(t.amount_paid) AS avgPaid ORDER BY avgPaid DESC",
      CA),
    q("How many laundering attempts are there for each laundering pattern?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL "
      "RETURN t.laundering_pattern, count(DISTINCT t.attempt_id)", CA),
    q("Which payment currency has the highest total amount paid in laundering transfers?",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering "
      "RETURN t.payment_currency, sum(t.amount_paid) AS total ORDER BY total DESC LIMIT 1", CA),
    q("How many transfers were made on each day? Return the date and the count, ordered by date.",
      "MATCH ()-[t:TRANSFERRED_TO]->() RETURN date(t.timestamp) AS day, count(t) ORDER BY day", CA),

    # paraphrases of earlier questions (same reference query)
    q("What's the total number of banks in the dataset?", "MATCH (b:Bank) RETURN count(b)", SA,
      "How many banks are there?"),
    q("Count the transfers that are flagged as money laundering.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.is_laundering RETURN count(t)", SA,
      "How many transfers are marked as laundering?"),
    q("Give me the 5 accounts that made the most outgoing transfers.",
      "MATCH (a:Account)-[:TRANSFERRED_TO]->(:Account) WITH a, count(*) AS n ORDER BY n DESC LIMIT 5 "
      "RETURN a.account_number, n", CA, "Find the top 5 accounts by number of outgoing transfers."),
    q("Who is the entity with the largest number of accounts?",
      "MATCH (e:Entity)-[:OWNS]->(a:Account) WITH e, count(a) AS n ORDER BY n DESC LIMIT 1 RETURN e.name, n", CA,
      "Which entity owns the most accounts?"),
    q("Break down the number of laundering transfers by laundering pattern.",
      "MATCH ()-[t:TRANSFERRED_TO]->() WHERE t.laundering_pattern IS NOT NULL "
      "RETURN t.laundering_pattern, count(t)", CA, "For each laundering pattern, how many transfers are there?"),
]

# sample questions in the demo app (answered correctly by the fine-tuned GAT model)
DEMO = [
    "How many transfers are marked as laundering?",
    "Which payment currency has the highest total amount paid in laundering transfers?",
    "Who is the entity with the largest number of accounts?",
    "How many laundering transfers are there for each payment format?",
]
