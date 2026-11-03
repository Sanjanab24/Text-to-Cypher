"""Phase 4: CodeT5+ with a relation-aware graph attention (GAT) schema encoder.

The schema of the target graph becomes a small graph: one node per label, relationship type and property,
joined by typed edges (label has property, start label -> relationship type -> end label, ...).
- Schema linking: each schema node starts from the embedding of its name and attends over the encoded
  question + schema text, so nodes mentioned in the question light up.
- Two GATv2 layers with edge-type embeddings pass this along the schema structure (relation-aware).
- The node vectors are appended to the T5 encoder output, so the decoder attends to the text and the schema graph.
"""
import re
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from serialize import SCHEMAS, internal

KINDS = ["label", "rel_type", "property"]
EDGE_TYPES = ["label_has_prop", "rel_has_prop", "start_of", "end_of"]  # plus the reverse of each


def words(name):
    """'primarySubstance' / 'SCORED_GOAL' -> 'primary substance' / 'scored goal'"""
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return " ".join(t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t)


def schema_graph(graph):
    """Returns (names, kinds, edges) for a schema; edges are (src, dst, edge type index)."""
    s = SCHEMAS[graph]
    names, kinds, edges, index = [], [], [], {}

    def node(key, name, kind):
        if key not in index:
            index[key] = len(names)
            names.append(words(name))
            kinds.append(KINDS.index(kind))
        return index[key]

    def link(a, b, t):
        edges.append((a, b, EDGE_TYPES.index(t)))
        edges.append((b, a, EDGE_TYPES.index(t) + len(EDGE_TYPES)))

    for label, props in s["node_props"].items():
        if internal(label):
            continue
        n = node(("L", label), label, "label")
        for p in props:
            link(n, node(("P", label, p["property"]), p["property"], "property"), "label_has_prop")
    for t, props in s["rel_props"].items():
        r = node(("R", t), t, "rel_type")
        for p in props:
            link(r, node(("RP", t, p["property"]), p["property"], "property"), "rel_has_prop")
    for e in s["relationships"]:
        if internal(e["start"]) or internal(e["end"]):
            continue
        r = node(("R", e["type"]), e["type"], "rel_type")
        link(node(("L", e["start"]), e["start"], "label"), r, "start_of")
        link(r, node(("L", e["end"]), e["end"], "label"), "end_of")
    return names, kinds, sorted(set(edges))


class GATText2Cypher(nn.Module):
    def __init__(self, backbone, tok, layers=2, heads=4):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(backbone, dtype=torch.float32)
        self.config = self.t5.config
        d = self.t5.config.d_model
        self.tok = tok
        self.kind = nn.Embedding(len(KINDS), d)
        self.link = nn.MultiheadAttention(d, 8, batch_first=True)
        self.edge = nn.Embedding(2 * len(EDGE_TYPES), d)
        self.gats = nn.ModuleList(GATv2Conv(d, d // heads, heads=heads, edge_dim=d) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers + 1))
        self.out = nn.Linear(d, d)
        self._graphs = {}

    @classmethod
    def load(cls, path, tok):
        model = cls(path, tok)
        model.load_state_dict(torch.load(Path(path) / "gat.pt", weights_only=True), strict=False)
        return model

    def save_pretrained(self, path):
        self.t5.save_pretrained(path)
        torch.save({k: v for k, v in self.state_dict().items() if not k.startswith("t5.")}, Path(path) / "gat.pt")

    def gradient_checkpointing_enable(self):
        self.t5.gradient_checkpointing_enable()

    def graph_tensors(self, graph, device):
        if graph not in self._graphs:
            names, kinds, edges = schema_graph(graph)
            ids = self.tok(names, add_special_tokens=False, padding=True, return_tensors="pt").input_ids
            self._graphs[graph] = (ids, torch.tensor(kinds), torch.tensor([e[:2] for e in edges]).T,
                                   torch.tensor([e[2] for e in edges]))
        return [t.to(device) for t in self._graphs[graph]]

    def encode(self, input_ids, attention_mask, graphs):
        text = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        xs, eis, ets, batch, offset = [], [], [], [], 0
        emb = self.t5.get_input_embeddings()
        for b, g in enumerate(graphs):
            ids, kinds, ei, et = self.graph_tensors(g, input_ids.device)
            mask = (ids != self.tok.pad_token_id).unsqueeze(-1).float()
            xs.append((emb(ids) * mask).sum(1) / mask.sum(1).clamp(min=1) + self.kind(kinds))
            eis.append(ei + offset)
            ets.append(et)
            batch.append(torch.full((len(ids),), b, device=ids.device))
            offset += len(ids)
        x, batch = torch.cat(xs), torch.cat(batch)
        # scatter softmax in GATv2 is safer in fp32
        with torch.autocast("cuda", enabled=False):
            x, text32 = x.float(), text.float()
            dense, node_mask = to_dense_batch(x, batch)
            linked, _ = self.link(dense, text32, text32, key_padding_mask=attention_mask == 0)
            x = self.norms[0](x + linked[node_mask])
            ei, ea = torch.cat(eis, 1), self.edge(torch.cat(ets))
            for gat, norm in zip(self.gats, self.norms[1:]):
                x = norm(x + gat(x, ei, ea))
            nodes, node_mask = to_dense_batch(self.out(x), batch)
        memory = torch.cat([text, nodes.to(text.dtype)], 1)
        return BaseModelOutput(last_hidden_state=memory), torch.cat([attention_mask, node_mask.long()], 1)

    def forward(self, input_ids, attention_mask, graphs, labels=None):
        enc, mask = self.encode(input_ids, attention_mask, graphs)
        return self.t5(encoder_outputs=enc, attention_mask=mask, labels=labels)

    def generate(self, input_ids, attention_mask, graphs, **kwargs):
        enc, mask = self.encode(input_ids, attention_mask, graphs)
        return self.t5.generate(encoder_outputs=enc, attention_mask=mask, **kwargs)
