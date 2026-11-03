"""Load the Mind the Query graph dumps into Neo4j, one database per graph.

Database names follow the schema keys in db_schema.jsonl, with "_" replaced by "-"
(Neo4j database names cannot contain underscores); "er" becomes "entity-resolution"
because names need at least 3 characters. Star Wars ships as CSVs, not a dump,
and is loaded separately.

Usage: .venv\\Scripts\\python.exe src\\load_mtq_graphs.py
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import DatabaseUnavailable

ROOT = Path(__file__).resolve().parent.parent
DUMPS = ROOT / "data/raw/Mind-the-Query/Datasets"

# schema key -> dump file
GRAPHS = {
    "wwc": "wwc2019-50.dump",
    "twitter_trolls": "twitter-trolls-50.dump",
    "covid": "contact-tracing-50.dump",
    "healthcare": "healthcare-analytics-50.dump",
    "pole": "pole-50.dump",
    "bloom50": "bloom-50.dump",
    "osm": "openstreetmap-50.dump",
    "gdsc": "graph-data-science-50.dump",
    "er": "entity-resolution-50.dump",
    "legis_graph": "legis-graph-50.dump",
}


def db_name(schema_key):
    # Neo4j names: 3-63 chars, no underscores
    return {"er": "entity-resolution"}.get(schema_key, schema_key.replace("_", "-"))


def load_dump(neo4j_home, schema_key, dump_file):
    name = db_name(schema_key)
    with tempfile.TemporaryDirectory() as tmp:
        # neo4j-admin expects <from-path>/<database>.dump
        shutil.copy(DUMPS / dump_file, Path(tmp) / f"{name}.dump")
        subprocess.run(
            [str(Path(neo4j_home) / "bin/neo4j-admin.bat"), "database", "load", name,
             f"--from-path={tmp}", "--overwrite-destination=true"],
            check=True, capture_output=True, text=True,
        )
    return name


def count_nodes(driver, name, tries=30):
    # a just-created database can report online before it accepts queries
    for _ in range(tries):
        try:
            return driver.execute_query("MATCH (n) RETURN count(n) AS n", database_=name).records[0]["n"]
        except DatabaseUnavailable:
            time.sleep(2)
    raise RuntimeError(f"{name} did not come online")


def main():
    load_dotenv(ROOT / ".env")
    neo4j_home = os.environ["NEO4J_HOME"]
    if os.getenv("NEO4J_JAVA_HOME"):  # neo4j-admin needs Java 21, not the system Java
        os.environ["JAVA_HOME"] = os.environ["NEO4J_JAVA_HOME"]
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )

    for key, dump in GRAPHS.items():
        name = db_name(key)
        # a loaded database must be stopped before it can be overwritten
        exists = driver.execute_query("SHOW DATABASES YIELD name WHERE name = $n RETURN name",
                                      n=name, database_="system").records
        if exists:
            driver.execute_query(f"STOP DATABASE `{name}` WAIT", database_="system")
        try:
            load_dump(neo4j_home, key, dump)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {name}: {e.stderr.strip().splitlines()[-1] if e.stderr else e}")
            continue
        if exists:
            driver.execute_query(f"START DATABASE `{name}` WAIT", database_="system")
        else:
            driver.execute_query(f"CREATE DATABASE `{name}` IF NOT EXISTS WAIT", database_="system")
        print(f"[ OK ] {name}: {count_nodes(driver, name)} nodes")

    driver.close()


if __name__ == "__main__":
    main()
