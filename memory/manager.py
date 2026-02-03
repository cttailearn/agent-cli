from __future__ import annotations

import threading
import uuid
from pathlib import Path

from . import paths
from .storage import ChatLogStore, KnowledgeGraphStore
from .worker import KnowledgeGraphWorker


def ensure_memory_scaffold(project_root: Path) -> dict[str, str]:
    root = paths.memory_root(project_root)
    core = paths.core_dir(project_root)
    chats = paths.chats_dir(project_root)
    kg = paths.kg_dir(project_root)

    root.mkdir(parents=True, exist_ok=True)
    core.mkdir(parents=True, exist_ok=True)
    chats.mkdir(parents=True, exist_ok=True)
    kg.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}
    defaults: dict[str, str] = {
        "soul.md": "\n".join(
            [
                "# 灵魂（Soul）",
                "",
                "这里记录不可妥协的核心价值、长期目标、底线与原则。",
                "这些内容应当在长期对话中逐步沉淀，并保持稳定性。",
                "",
            ]
        ),
        "traits.md": "\n".join(
            [
                "# 特性（Traits）",
                "",
                "这里记录稳定的行为风格、偏好、表达习惯与工作方式。",
                "这些内容应当在长期对话中逐步沉淀，并可缓慢演化。",
                "",
            ]
        ),
        "identity.md": "\n".join(
            [
                "# 身份（Identity）",
                "",
                "这里记录身份设定、职责边界、对外承诺与不做的事情。",
                "这些内容应当在长期对话中逐步沉淀，并保持一致性。",
                "",
            ]
        ),
        "user.md": "\n".join(
            [
                "# 用户记忆（User）",
                "",
                "这里记录与你对应“用户”的稳定信息，用于长期协作与决策一致性。内容允许更新与修订。",
                "",
                "## 偏好（Preferences）",
                "",
                "- ",
                "",
                "## 重要目标（Goals）",
                "",
                "- ",
                "",
                "## 关键决策（Decisions）",
                "",
                "- ",
                "",
                "## 关键数据（Data）",
                "",
                "- ",
                "",
            ]
        ),
    }
    for name, text in defaults.items():
        if name == "soul.md":
            p = paths.soul_path(project_root)
        elif name == "traits.md":
            p = paths.traits_path(project_root)
        elif name == "identity.md":
            p = paths.identity_path(project_root)
        elif name == "user.md":
            p = paths.user_path(project_root)
        else:
            continue
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", errors="replace")
            created[name] = p.as_posix()
    return created


def load_core_prompt(project_root: Path) -> str:
    parts: list[str] = []
    mapping = [
        ("灵魂（Soul）", paths.soul_path(project_root)),
        ("特性（Traits）", paths.traits_path(project_root)),
        ("身份（Identity）", paths.identity_path(project_root)),
        ("用户（User）", paths.user_path(project_root)),
    ]
    for title, p in mapping:
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if not text:
            continue
        parts.append(f"## {title}\n\n{text}".strip())
    return "\n\n".join(parts).strip()


class MemoryManager:
    def __init__(self, *, project_root: Path, model_name: str) -> None:
        self.project_root = project_root.resolve()
        self.model_name = model_name
        self.session_id = uuid.uuid4().hex[:12]
        self._turn = 0
        self._lock = threading.Lock()
        self.chat_store = ChatLogStore(chats_dir=paths.chats_dir(self.project_root), session_id=self.session_id)
        self.kg_store = KnowledgeGraphStore(graph_path=paths.graph_path(self.project_root))
        self.worker = KnowledgeGraphWorker(
            model_name=self.model_name,
            kg_store=self.kg_store,
        )

    def start(self) -> None:
        self.worker.start()

    def stop(self, *, flush: bool = True) -> None:
        self.worker.stop(flush=flush)

    def record_turn(self, *, user_text: str, assistant_text: str) -> Path | None:
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if not u and not a:
            return None
        with self._lock:
            self._turn += 1
            turn = self._turn
        path = self.chat_store.write_turn(turn=turn, user_text=u, assistant_text=a)
        self.worker.enqueue(path)
        return path
