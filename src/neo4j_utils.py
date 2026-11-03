"""Shared Neo4j helpers: connection, schema extraction, and database names for each graph."""
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent

# schema key (as in db_schema.jsonl) -> Neo4j database name
DB_NAMES = {
    "wwc": "wwc", "twitter_trolls": "twitter-trolls", "covid": "covid", "healthcare": "healthcare",
    "pole": "pole", "bloom50": "bloom50", "osm": "osm", "gdsc": "gdsc", "er": "entity-resolution",
    "legis_graph": "legis-graph", "aml": "aml",
    "star_wars": None,  # not loaded (CSV only); used for training, never executed
}

# Neo4j schema-procedure type names -> the names used in Mind the Query's db_schema.jsonl
TYPE_MAP = {
    "String": "STRING", "Long": "INTEGER", "Integer": "INTEGER", "Double": "FLOAT", "Float": "FLOAT",
    "Boolean": "BOOLEAN", "Date": "DATE", "DateTime": "DATE_TIME", "LocalDateTime": "LOCAL_DATE_TIME",
    "Point": "POINT", "Duration": "DURATION",
}


def get_driver():
    load_dotenv(ROOT / ".env")
    return GraphDatabase.driver(os.getenv("NEO4J_URI"),
                                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))


# newer Neo4j versions report Cypher type names ("STRING NOT NULL", "LOCAL DATETIME", "LIST<STRING>")
CYPHER_TYPE_MAP = {"LOCAL DATETIME": "LOCAL_DATE_TIME", "ZONED DATETIME": "DATE_TIME", "DATETIME": "DATE_TIME"}


def _dtype(types):
    t = (types or ["String"])[0]
    if t in TYPE_MAP:
        return TYPE_MAP[t]
    if t.endswith("Array"):
        return "LIST"
    t = t.replace(" NOT NULL", "").strip()
    if t.startswith("LIST"):
        return "LIST"
    return CYPHER_TYPE_MAP.get(t, t.replace(" ", "_"))


def extract_schema(driver, database):
    """Return {node_props, rel_props, relationships} in the same format as db_schema.jsonl."""
    node_props, rel_props = {}, {}
    for r in driver.execute_query("CALL db.schema.nodeTypeProperties()", database_=database).records:
        for label in r["nodeLabels"]:
            node_props.setdefault(label, [])
            if r["propertyName"]:
                node_props[label].append({"property": r["propertyName"], "datatype": _dtype(r["propertyTypes"])})
    for r in driver.execute_query("CALL db.schema.relTypeProperties()", database_=database).records:
        rel = r["relType"].strip(":`")
        if r["propertyName"]:
            rel_props.setdefault(rel, []).append({"property": r["propertyName"], "datatype": _dtype(r["propertyTypes"])})
    rels = set()
    for r in driver.execute_query(
            "MATCH (a)-[r]->(b) WITH DISTINCT labels(a) AS la, type(r) AS t, labels(b) AS lb "
            "UNWIND la AS s UNWIND lb AS e RETURN DISTINCT s, t, e", database_=database).records:
        rels.add((r["s"], r["t"], r["e"]))
    return {"node_props": node_props, "rel_props": rel_props,
            "relationships": [{"start": s, "type": t, "end": e} for s, t, e in sorted(rels)]}
