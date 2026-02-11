from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
