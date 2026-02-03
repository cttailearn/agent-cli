from __future__ import annotations

import math
import re
from pathlib import Path

from .storage import KnowledgeGraphStore


_TOKEN_RE = re.compile(r"[\\w\\u4e00-\\u9fff]+", re.UNICODE)


def _tokens(s: str) -> list[str]:
    raw = (s or "").lower()
    return [m.group(0) for m in _TOKEN_RE.finditer(raw) if m.group(0)]


def _node_text(node: dict[str, object]) -> str:
    name = node.get("name")
    typ = node.get("type")
    aliases = node.get("aliases")
    parts: list[str] = []
    if isinstance(name, str) and name:
        parts.append(name)
    if isinstance(typ, str) and typ:
        parts.append(typ)
    if isinstance(aliases, list):
        for a in aliases:
            if isinstance(a, str) and a:
                parts.append(a)
    return " ".join(parts).lower()


def search_graph(store: KnowledgeGraphStore, query: str, limit: int = 12) -> str:
    q = (query or "").strip()
    if not q:
        return ""
    toks = _tokens(q)
    if not toks:
        return ""
    with store.with_lock():
        g = store.get_graph_mut()
        nodes = g.get("nodes")
        edges = g.get("edges")
        if not isinstance(nodes, dict) or not isinstance(edges, list):
            return ""

        node_scores: list[tuple[float, str, dict[str, object]]] = []
        for nid, n in nodes.items():
            if not isinstance(nid, str) or not isinstance(n, dict):
                continue
            text = _node_text(n)
            if not text:
                continue
            hit = 0
            for t in toks:
                if t in text:
                    hit += 1
            if hit <= 0:
                continue
            score = hit / max(1.0, math.sqrt(len(text)))
            node_scores.append((score, nid, n))

        node_scores.sort(key=lambda x: x[0], reverse=True)
        top_nodes = {nid for _, nid, _ in node_scores[: max(10, limit)]}

        out: list[str] = []
        for score, nid, n in node_scores[: min(limit, len(node_scores))]:
            name = n.get("name")
            typ = n.get("type")
            if isinstance(name, str) and name:
                out.append(f"- node {nid} [{typ or 'unknown'}] {name}")

        edge_lines: list[str] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            s = e.get("source")
            t = e.get("target")
            rel = e.get("relation")
            ts = e.get("ts")
            doc = e.get("doc")
            if not isinstance(s, str) or not isinstance(t, str) or not isinstance(rel, str):
                continue
            if s not in top_nodes and t not in top_nodes:
                continue
            sn = nodes.get(s)
            tn = nodes.get(t)
            if not isinstance(sn, dict) or not isinstance(tn, dict):
                continue
            sname = sn.get("name")
            tname = tn.get("name")
            if not isinstance(sname, str) or not isinstance(tname, str):
                continue
            meta = []
            if isinstance(ts, str) and ts:
                meta.append(ts)
            if isinstance(doc, str) and doc:
                meta.append(Path(doc).name)
            suffix = f" ({', '.join(meta)})" if meta else ""
            edge_lines.append(f"- edge {sname} -[{rel}]-> {tname}{suffix}")

        if edge_lines:
            out.append("")
            out.extend(edge_lines[:limit])
        return "\n".join(out).strip()


def graph_stats(store: KnowledgeGraphStore) -> str:
    with store.with_lock():
        g = store.get_graph_mut()
        nodes = g.get("nodes")
        edges = g.get("edges")
        docs = g.get("documents")
        n = len(nodes) if isinstance(nodes, dict) else 0
        e = len(edges) if isinstance(edges, list) else 0
        d = len(docs) if isinstance(docs, dict) else 0
    return f"nodes={n} edges={e} documents={d}"

