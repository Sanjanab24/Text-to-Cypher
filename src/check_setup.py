"""Phase 1 sanity check: libraries, GPU, Neo4j connection, datasets on disk."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(name, fn):
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")


def torch_info():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    if torch.cuda.is_available():
        torch.ones(1, device="cuda")  # fails here if the build doesn't support this GPU
    return f"torch {torch.__version__}, {gpu}"


def pyg_info():
    import torch
    import torch_geometric
    from torch_geometric.nn import GATConv
    out = GATConv(4, 8, heads=2)(torch.randn(3, 4), torch.tensor([[0, 1], [1, 2]]))
    return f"torch_geometric {torch_geometric.__version__}, GATConv output {tuple(out.shape)}"


def transformers_info():
    import transformers
    return f"transformers {transformers.__version__}"


def neo4j_info():
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
    load_dotenv(ROOT / ".env")
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        n = driver.execute_query("MATCH (n) RETURN count(n) AS n").records[0]["n"]
    return f"connected to {uri}, {n} nodes"


def data_info():
    found = {
        "Mind the Query": (ROOT / "data/raw/Mind-the-Query").exists(),
        "IBM AML (HI-Small)": (ROOT / "data/raw/ibm-aml/HI-Small_Trans.csv").exists(),
    }
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise FileNotFoundError(f"missing: {', '.join(missing)}")
    return "all datasets present"


if __name__ == "__main__":
    check("PyTorch + GPU", torch_info)
    check("PyTorch Geometric", pyg_info)
    check("Transformers", transformers_info)
    check("Neo4j", neo4j_info)
    check("Datasets", data_info)
