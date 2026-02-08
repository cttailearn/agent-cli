from __future__ import annotations

import json
import queue
import re
import threading
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .model import init_model
from .storage import KnowledgeGraphStore


_FRONT_MATTER_RE = re.compile(r"^---\s*$")
_TURN_RE = re.compile(r"^##\s+Turn\s+(\d+)\s*$", re.MULTILINE)


def _extract_text(msg: object) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _parse_ts_from_md(md: str) -> str:
    lines = (md or "").splitlines()
    idxs: list[int] = [i for i, line in enumerate(lines[:40]) if _FRONT_MATTER_RE.match(line.strip())]
    if len(idxs) >= 2 and idxs[0] == 0 and idxs[1] > 0:
        for line in lines[idxs[0] + 1 : idxs[1]]:
            s = line.strip()
            if s.lower().startswith("ts:"):
                v = s.split(":", 1)[1].strip()
                if v:
                    return v
    return ""


def _split_turns(md: str) -> list[tuple[int, str]]:
    text = md or ""
    matches = list(_TURN_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [(0, body)] if body else []
    out: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        raw_n = m.group(1)
        try:
            n = int(raw_n)
        except Exception:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            out.append((n, chunk))
    return out


def _safe_json_loads(text: str) -> dict[str, object] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            return None
    return obj if isinstance(obj, dict) else None


def _node_key(name: str, typ: str) -> str:
    return f"{(name or '').strip().lower()}::{(typ or 'unknown').strip().lower()}"


def _upsert_graph(
    *,
    store: KnowledgeGraphStore,
    doc_id: str,
    doc_ts: str,
    doc_hash: str,
    extracted: dict[str, object],
) -> None:
    rel_doc = (doc_id or "").strip()
    if not rel_doc:
        return
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with store.with_lock():
        g = store.get_graph_mut()
        docs = g.get("documents")
        if not isinstance(docs, dict):
            docs = {}
            g["documents"] = docs
        docs[rel_doc] = {"ts": doc_ts, "hash": doc_hash, "processed_at": now_iso}

        nodes = g.get("nodes")
        if not isinstance(nodes, dict):
            nodes = {}
            g["nodes"] = nodes
        edges = g.get("edges")
        if not isinstance(edges, list):
            edges = []
            g["edges"] = edges
        else:
            edges[:] = [e for e in edges if not (isinstance(e, dict) and e.get("doc") == rel_doc)]
        for n in nodes.values():
            if not isinstance(n, dict):
                continue
            mentions = n.get("mentions")
            if isinstance(mentions, list):
                n["mentions"] = [m for m in mentions if not (isinstance(m, dict) and m.get("doc") == rel_doc)]

        entities = extracted.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        rels = extracted.get("relations", [])
        if not isinstance(rels, list):
            rels = []

        key_to_id: dict[str, str] = {}
        for nid, n in nodes.items():
            if not isinstance(nid, str) or not isinstance(n, dict):
                continue
            name = n.get("name")
            typ = n.get("type")
            if isinstance(name, str) and name.strip():
                key_to_id[_node_key(name, str(typ or "unknown"))] = nid

        next_id = 1
        if nodes:
            for nid in nodes.keys():
                if isinstance(nid, str) and nid.startswith("n"):
                    try:
                        next_id = max(next_id, int(nid[1:]) + 1)
                    except Exception:
                        pass

        def ensure_node(name: str, typ: str, aliases: list[str] | None = None) -> str:
            nonlocal next_id
            k = _node_key(name, typ)
            existing = key_to_id.get(k)
            if existing:
                n = nodes.get(existing)
                if isinstance(n, dict):
                    mentions = n.get("mentions")
                    if not isinstance(mentions, list):
                        mentions = []
                        n["mentions"] = mentions
                    mentions.append({"ts": doc_ts, "doc": rel_doc})
                    if aliases:
                        a = n.get("aliases")
                        if not isinstance(a, list):
                            a = []
                            n["aliases"] = a
                        for x in aliases:
                            if isinstance(x, str) and x and x not in a:
                                a.append(x)
                return existing
            nid = f"n{next_id}"
            next_id += 1
            node_obj: dict[str, object] = {
                "id": nid,
                "name": name,
                "type": typ or "unknown",
                "created_at": now_iso,
                "mentions": [{"ts": doc_ts, "doc": rel_doc}],
            }
            if aliases:
                node_obj["aliases"] = [x for x in aliases if isinstance(x, str) and x]
            nodes[nid] = node_obj
            key_to_id[k] = nid
            return nid

        for e in entities:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            typ = e.get("type", "unknown")
            if not isinstance(name, str) or not name.strip():
                continue
            aliases = e.get("aliases")
            aliases_list = aliases if isinstance(aliases, list) else None
            ensure_node(name.strip(), str(typ or "unknown").strip(), aliases=aliases_list)

        for r in rels:
            if not isinstance(r, dict):
                continue
            s = r.get("source")
            t = r.get("target")
            rel = r.get("relation")
            if not isinstance(s, str) or not isinstance(t, str) or not isinstance(rel, str):
                continue
            if not s.strip() or not t.strip() or not rel.strip():
                continue
            st = ensure_node(s.strip(), str(r.get("source_type") or "unknown").strip())
            tt = ensure_node(t.strip(), str(r.get("target_type") or "unknown").strip())
            edge = {
                "source": st,
                "target": tt,
                "relation": rel.strip(),
                "ts": doc_ts,
                "doc": rel_doc,
            }
            attrs = r.get("attributes")
            if isinstance(attrs, dict) and attrs:
                edge["attributes"] = attrs
            edges.append(edge)
        store.save()


def ingest_chat_markdown(*, model, store: KnowledgeGraphStore, path: Path) -> tuple[bool, str]:
    p = path.resolve()
    if not p.exists() or not p.is_file():
        return False, "not_found"
    md = p.read_text(encoding="utf-8", errors="replace")
    doc_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    turns = _split_turns(md)
    if not turns:
        return True, "empty"

    doc_prefix = p.as_posix()
    to_process: list[tuple[int, str, str]] = []
    with store.with_lock():
        g = store.get_graph_mut()
        docs = g.get("documents")
        docs_dict = docs if isinstance(docs, dict) else {}
        for turn_n, chunk in turns:
            doc_id = doc_prefix if turn_n == 0 else f"{doc_prefix}#t{turn_n}"
            doc_hash = store.document_hash(chunk)
            meta = docs_dict.get(doc_id)
            if isinstance(meta, dict) and meta.get("hash") == doc_hash:
                continue
            to_process.append((turn_n, doc_id, chunk))

    if not to_process:
        return True, "skipped"

    sys_text = "\n".join(
        [
            "你是一个信息抽取器。输入是一段 Markdown 聊天记录，你需要把它转为知识图谱片段。",
            "输出必须是严格 JSON（不要 Markdown、不要注释、不要额外文本）。",
            "JSON schema：",
            "{",
            '  "entities": [{"name": "实体名", "type": "Person|Org|Project|File|Concept|Task|Other", "aliases": ["别名"]}],',
            '  "relations": [{"source": "实体名", "relation": "关系", "target": "实体名", "attributes": {"key": "value"}}]',
            "}",
            "要求：",
            "- 不要输出空实体名",
            "- 关系用简短动词短语（中文或英文均可）",
            "- 只抽取对未来有用、可复用的稳定事实与偏好",
        ]
    )
    ok_any = False
    for turn_n, doc_id, chunk in to_process:
        user_text = "\n".join(
            [
                f"doc_id={doc_id}",
                f"doc_path={p.as_posix()}",
                f"doc_ts={doc_ts}",
                f"turn={turn_n}",
                "doc_markdown:",
                chunk,
            ]
        )
        try:
            resp = model.invoke([SystemMessage(content=sys_text), HumanMessage(content=user_text)])
        except Exception:
            continue
        extracted = _safe_json_loads(_extract_text(resp))
        if extracted is None:
            extracted = {"entities": [], "relations": []}
        doc_hash = store.document_hash(chunk)
        _upsert_graph(store=store, doc_id=doc_id, doc_ts=doc_ts or "", doc_hash=doc_hash, extracted=extracted)
        ok_any = True
    return (True, "ok") if ok_any else (False, "invoke_failed")


class KnowledgeGraphWorker:
    def __init__(self, *, model_name: str, kg_store: KnowledgeGraphStore) -> None:
        self.model_name = model_name
        self.kg_store = kg_store
        self._q: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="KnowledgeGraphWorker", daemon=True)
        self._thread.start()

    def stop(self, *, flush: bool) -> None:
        if flush:
            deadline = time.time() + 8.0
            while not self._q.empty() and time.time() < deadline:
                time.sleep(0.05)
        self._stop.set()
        try:
            self._q.put_nowait(Path())
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def enqueue(self, path: Path) -> None:
        try:
            self._q.put_nowait(path)
        except Exception:
            pass

    def _run(self) -> None:
        model = None
        while not self._stop.is_set():
            try:
                p = self._q.get(timeout=0.25)
            except Exception:
                continue
            if not isinstance(p, Path) or not p.as_posix():
                continue
            if model is None:
                try:
                    model = init_model(self.model_name)
                except Exception:
                    model = None
            if model is None:
                continue
            try:
                ingest_chat_markdown(model=model, store=self.kg_store, path=p)
            except Exception:
                continue
