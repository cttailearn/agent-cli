from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_iso(ts: float | None = None) -> str:
    t = time.gmtime(ts if ts is not None else time.time())
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", t)


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class ChatTurn:
    ts: str
    session_id: str
    turn: int
    user_text: str
    assistant_text: str

    def to_markdown_entry(self) -> str:
        return "\n".join(
            [
                f"## Turn {self.turn}",
                "",
                f"- ts: {self.ts}",
                "",
                "### User",
                "",
                self.user_text,
                "",
                "### Assistant",
                "",
                self.assistant_text,
            ]
        )


class ChatLogStore:
    def __init__(self, *, chats_dir: Path, session_id: str) -> None:
        self.chats_dir = chats_dir.resolve()
        self.session_id = session_id
        self._lock = threading.Lock()
        self.chats_dir.mkdir(parents=True, exist_ok=True)

    def write_turn(self, *, turn: int, user_text: str, assistant_text: str) -> Path:
        ts = _utc_iso()
        t = ChatTurn(ts=ts, session_id=self.session_id, turn=turn, user_text=user_text, assistant_text=assistant_text)
        date_dir = self.chats_dir / ts.split("T", 1)[0]
        date_dir.mkdir(parents=True, exist_ok=True)
        path = (date_dir / f"s{self.session_id}.md").resolve()

        entry = t.to_markdown_entry().rstrip() + "\n"
        with self._lock:
            if not path.exists():
                header = "\n".join(
                    [
                        "---",
                        f"ts: {ts}",
                        f"session: {self.session_id}",
                        "---",
                        "",
                        "# Chat Log",
                        "",
                    ]
                )
                path.write_text(header, encoding="utf-8", errors="replace")
            with path.open("a", encoding="utf-8", errors="replace", newline="\n") as f:
                f.write("\n")
                f.write(entry)
        return path


class KnowledgeGraphStore:
    def __init__(self, *, graph_path: Path) -> None:
        self.graph_path = graph_path.resolve()
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._graph: dict[str, object] | None = None

    def _load_unlocked(self) -> dict[str, object]:
        if self._graph is not None:
            return self._graph
        if not self.graph_path.exists():
            self._graph = {"version": 1, "nodes": {}, "edges": [], "documents": {}}
            return self._graph
        try:
            raw = json.loads(self.graph_path.read_text(encoding="utf-8", errors="replace") or "{}")
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("version", 1)
        raw.setdefault("nodes", {})
        raw.setdefault("edges", [])
        raw.setdefault("documents", {})
        if not isinstance(raw.get("nodes"), dict):
            raw["nodes"] = {}
        if not isinstance(raw.get("edges"), list):
            raw["edges"] = []
        if not isinstance(raw.get("documents"), dict):
            raw["documents"] = {}
        self._graph = raw
        return raw

    def load(self) -> dict[str, object]:
        with self._lock:
            return dict(self._load_unlocked())

    def get_graph_mut(self) -> dict[str, object]:
        return self._load_unlocked()

    def save(self) -> None:
        with self._lock:
            g = self._load_unlocked()
            tmp = self.graph_path.with_suffix(self.graph_path.suffix + ".tmp")
            tmp.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
            os.replace(tmp, self.graph_path)

    def with_lock(self):
        return self._lock

    def document_hash(self, text: str) -> str:
        return _sha256_text(text)


class PersistentInMemoryStore:
    def __init__(self, *, path: Path, index: dict[str, object] | None = None) -> None:
        from langgraph.store.memory import InMemoryStore

        self._path = path.resolve()
        self._store = InMemoryStore(index=index) if index else InMemoryStore()
        self._lock = threading.RLock()
        self._load_unlocked()

    @property
    def supports_ttl(self) -> bool:
        return bool(getattr(self._store, "supports_ttl", False))

    def _load_unlocked(self) -> None:
        from langgraph.store.base import Item

        if not self._path.exists() or not self._path.is_file():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8", errors="replace") or "{}")
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        items = raw.get("items")
        vectors = raw.get("vectors")

        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                ns = it.get("namespace")
                key = it.get("key")
                val = it.get("value")
                created_at = it.get("created_at")
                updated_at = it.get("updated_at")
                if not isinstance(ns, list) or not all(isinstance(x, str) for x in ns):
                    continue
                if not isinstance(key, str) or not key:
                    continue
                if not isinstance(val, dict):
                    continue
                ca = created_at if isinstance(created_at, str) else datetime.now(timezone.utc).isoformat()
                ua = updated_at if isinstance(updated_at, str) else ca
                item = Item(
                    value=val,
                    key=key,
                    namespace=tuple(ns),
                    created_at=ca,
                    updated_at=ua,
                )
                data = getattr(self._store, "_data", None)
                if isinstance(data, dict):
                    data[tuple(ns)][key] = item

        if isinstance(vectors, list):
            vecs = getattr(self._store, "_vectors", None)
            if isinstance(vecs, dict):
                for entry in vectors:
                    if not isinstance(entry, dict):
                        continue
                    ns = entry.get("namespace")
                    key = entry.get("key")
                    paths = entry.get("paths")
                    if not isinstance(ns, list) or not all(isinstance(x, str) for x in ns):
                        continue
                    if not isinstance(key, str) or not key:
                        continue
                    if not isinstance(paths, dict):
                        continue
                    for p, v in paths.items():
                        if not isinstance(p, str) or not p:
                            continue
                        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                            continue
                        vecs[tuple(ns)][key][p] = [float(x) for x in v]

    def _dump_unlocked(self) -> dict[str, object]:
        data = getattr(self._store, "_data", None)
        vecs = getattr(self._store, "_vectors", None)
        items_out: list[dict[str, object]] = []
        vectors_out: list[dict[str, object]] = []

        if isinstance(data, dict):
            for ns, ns_items in data.items():
                if not isinstance(ns, tuple) or not isinstance(ns_items, dict):
                    continue
                for item in ns_items.values():
                    dict_fn = getattr(item, "dict", None)
                    if callable(dict_fn):
                        try:
                            items_out.append(dict_fn())
                        except Exception:
                            continue

        if isinstance(vecs, dict):
            for ns, ns_items in vecs.items():
                if not isinstance(ns, tuple) or not isinstance(ns_items, dict):
                    continue
                for key, paths in ns_items.items():
                    if not isinstance(key, str) or not isinstance(paths, dict):
                        continue
                    out_paths: dict[str, list[float]] = {}
                    for p, v in paths.items():
                        if not isinstance(p, str) or not isinstance(v, list):
                            continue
                        out_paths[p] = [float(x) for x in v if isinstance(x, (int, float))]
                    if out_paths:
                        vectors_out.append({"namespace": list(ns), "key": key, "paths": out_paths})

        return {"version": 1, "items": items_out, "vectors": vectors_out}

    def _flush_unlocked(self) -> None:
        payload = self._dump_unlocked()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
        os.replace(tmp, self._path)

    def batch(self, ops) -> list[Any]:
        from langgraph.store.base import PutOp

        with self._lock:
            has_put = any(isinstance(op, PutOp) for op in ops)
            res = self._store.batch(ops)
            if has_put:
                self._flush_unlocked()
            return res

    async def abatch(self, ops) -> list[Any]:
        from langgraph.store.base import PutOp

        with self._lock:
            has_put = any(isinstance(op, PutOp) for op in ops)
            res = await self._store.abatch(ops)
            if has_put:
                self._flush_unlocked()
            return res

    def __getattr__(self, name: str):
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            with self._lock:
                res = attr(*args, **kwargs)
                if name in {"put", "update", "delete", "aput", "aupdate", "adelete"}:
                    try:
                        self._flush_unlocked()
                    except Exception:
                        pass
                return res

        return _wrapped

    def snapshot_items(self, namespace: tuple[str, ...]) -> list[dict[str, object]]:
        from langgraph.store.base import Item

        ns = namespace if isinstance(namespace, tuple) else tuple(namespace or ())
        out: list[dict[str, object]] = []
        with self._lock:
            data = getattr(self._store, "_data", None)
            if not isinstance(data, dict):
                return out
            ns_items = data.get(ns)
            if not isinstance(ns_items, dict):
                return out
            for key, item in ns_items.items():
                if not isinstance(key, str) or not key:
                    continue
                if not isinstance(item, Item):
                    continue
                val = getattr(item, "value", None)
                if not isinstance(val, dict):
                    continue
                out.append(
                    {
                        "namespace": list(getattr(item, "namespace", ()) or ()),
                        "key": getattr(item, "key", ""),
                        "created_at": getattr(item, "created_at", ""),
                        "updated_at": getattr(item, "updated_at", ""),
                        "value": val,
                    }
                )
        return out


def _safe_component(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return "default"
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("._-")
    return s[:160] if s else "default"


class PageIndexStore:
    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self._lock = threading.RLock()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _ns_dir(self, namespace: tuple[str, ...]) -> Path:
        ns = namespace if isinstance(namespace, tuple) else tuple(namespace or ())
        p = self.root_dir
        for part in ns:
            p = p / _safe_component(str(part))
        return p.resolve()

    def _doc_path(self, namespace: tuple[str, ...], key: str) -> Path:
        k = _safe_component(key)
        return (self._ns_dir(namespace) / f"{k}.json").resolve()

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, object]) -> None:
        k = (key or "").strip()
        if not k:
            raise ValueError("Missing key.")
        if not isinstance(value, dict):
            raise TypeError("value must be a dict")
        p = self._doc_path(namespace, k)
        payload = dict(value)
        with self._lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
            os.replace(tmp, p)

    def get(self, namespace: tuple[str, ...], key: str) -> dict[str, object] | None:
        k = (key or "").strip()
        if not k:
            return None
        p = self._doc_path(namespace, k)
        with self._lock:
            if not p.exists() or not p.is_file():
                return None
            try:
                raw = json.loads(p.read_text(encoding="utf-8", errors="replace") or "{}")
            except Exception:
                return None
        return raw if isinstance(raw, dict) else None

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        k = (key or "").strip()
        if not k:
            return
        p = self._doc_path(namespace, k)
        with self._lock:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    def iter_docs(self, namespace: tuple[str, ...]) -> list[tuple[str, Path]]:
        d = self._ns_dir(namespace)
        with self._lock:
            if not d.exists() or not d.is_dir():
                return []
            out: list[tuple[str, Path]] = []
            for p in sorted(d.glob("*.json")):
                if not p.is_file():
                    continue
                out.append((p.stem, p.resolve()))
            return out

    def list_namespaces(self, prefix: tuple[str, ...] | None = None, limit: int = 200) -> list[tuple[str, ...]]:
        pre = prefix if isinstance(prefix, tuple) else tuple(prefix or ()) if prefix is not None else ()
        base = self._ns_dir(pre) if pre else self.root_dir
        out: list[tuple[str, ...]] = []
        with self._lock:
            if not base.exists() or not base.is_dir():
                return out
            for dirpath, dirnames, filenames in os.walk(base):
                if len(out) >= max(1, int(limit or 200)):
                    break
                if any(f.endswith(".json") for f in filenames):
                    rel = Path(dirpath).resolve().relative_to(self.root_dir)
                    parts = tuple([p for p in rel.as_posix().split("/") if p])
                    ns = tuple(pre + parts) if pre else parts
                    out.append(ns)
            out.sort()
        return out

    def stats(self, prefix: tuple[str, ...] | None = None) -> dict[str, int]:
        namespaces = self.list_namespaces(prefix=prefix, limit=10_000)
        doc_count = 0
        node_count = 0
        for ns in namespaces:
            for _, p in self.iter_docs(ns):
                doc_count += 1
                try:
                    raw = json.loads(p.read_text(encoding="utf-8", errors="replace") or "{}")
                except Exception:
                    continue
                if not isinstance(raw, dict):
                    continue
                structure = raw.get("structure")
                if isinstance(structure, list):
                    node_count += _count_pageindex_nodes(structure)
        return {"namespaces": len(namespaces), "docs": doc_count, "nodes": node_count}


def _count_pageindex_nodes(structure: object) -> int:
    if isinstance(structure, dict):
        n = 1
        nodes = structure.get("nodes")
        if isinstance(nodes, list):
            for c in nodes:
                n += _count_pageindex_nodes(c)
        return n
    if isinstance(structure, list):
        return sum(_count_pageindex_nodes(x) for x in structure)
    return 0
