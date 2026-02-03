from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


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

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "---",
                f"ts: {self.ts}",
                f"session: {self.session_id}",
                f"turn: {self.turn}",
                "---",
                "",
                "## User",
                "",
                self.user_text,
                "",
                "## Assistant",
                "",
                self.assistant_text,
                "",
            ]
        )


class ChatLogStore:
    def __init__(self, *, chats_dir: Path, session_id: str) -> None:
        self.chats_dir = chats_dir.resolve()
        self.session_id = session_id
        self.chats_dir.mkdir(parents=True, exist_ok=True)

    def write_turn(self, *, turn: int, user_text: str, assistant_text: str) -> Path:
        ts = _utc_iso()
        t = ChatTurn(ts=ts, session_id=self.session_id, turn=turn, user_text=user_text, assistant_text=assistant_text)
        date_dir = self.chats_dir / ts.split("T", 1)[0]
        date_dir.mkdir(parents=True, exist_ok=True)
        safe_ts = ts.replace(":", "").replace("Z", "Z")
        path = (date_dir / f"{safe_ts}_s{self.session_id}_t{turn:04d}.md").resolve()
        path.write_text(t.to_markdown(), encoding="utf-8", errors="replace")
        return path


class KnowledgeGraphStore:
    def __init__(self, *, graph_path: Path) -> None:
        self.graph_path = graph_path.resolve()
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
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

