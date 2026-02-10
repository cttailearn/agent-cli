from __future__ import annotations

import threading
import time
import uuid
import os
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from . import paths
from .storage import ChatLogStore, PageIndexStore
from .worker import PageIndexWorker


def ensure_memory_scaffold(project_root: Path) -> dict[str, str]:
    root = paths.memory_root(project_root)
    core = paths.core_dir(project_root)
    chats = paths.chats_dir(project_root)
    pi = paths.pageindex_dir(project_root)
    pi_chats = paths.pageindex_chats_dir(project_root)
    pi_ltm = paths.pageindex_ltm_dir(project_root)
    pi_docs = paths.pageindex_docs_dir(project_root)
    episodic_dir = paths.langgraph_store_path(project_root).parent

    root.mkdir(parents=True, exist_ok=True)
    core.mkdir(parents=True, exist_ok=True)
    chats.mkdir(parents=True, exist_ok=True)
    pi.mkdir(parents=True, exist_ok=True)
    pi_chats.mkdir(parents=True, exist_ok=True)
    pi_ltm.mkdir(parents=True, exist_ok=True)
    pi_docs.mkdir(parents=True, exist_ok=True)
    episodic_dir.mkdir(parents=True, exist_ok=True)

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

def _extract_system_text(msg) -> str:
    if msg is None:
        return ""
    text_attr = getattr(msg, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _strip_block(text: str, *, start: str, end: str) -> str:
    t = text or ""
    if not t.strip():
        return ""
    i = t.find(start)
    if i < 0:
        return t
    j = t.find(end, i + len(start))
    if j < 0:
        return t[:i].rstrip()
    return f"{t[:i]}{t[j + len(end):]}".strip()


def _core_files_signature(project_root: Path) -> tuple[tuple[str, int, int], ...]:
    files = [
        paths.soul_path(project_root),
        paths.traits_path(project_root),
        paths.identity_path(project_root),
        paths.user_path(project_root),
    ]
    sig: list[tuple[str, int, int]] = []
    for p in files:
        try:
            st = p.stat()
            sig.append((p.as_posix(), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            sig.append((p.as_posix(), 0, 0))
    return tuple(sig)


class CoreMemoryMiddleware(AgentMiddleware):
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._core_sig: tuple[tuple[str, int, int], ...] | None = None
        self._core_text: str = ""

    def _load_core_prompt_cached(self) -> str:
        sig = _core_files_signature(self.project_root)
        if sig == self._core_sig:
            return self._core_text
        core_text = load_core_prompt(self.project_root).strip()
        self._core_sig = sig
        self._core_text = core_text
        return core_text

    def wrap_model_call(self, request: ModelRequest, handler):
        base_text = _extract_system_text(request.system_message)
        core_text = self._load_core_prompt_cached()
        if not core_text:
            return handler(request)

        start = "<CORE_MEMORIES priority=highest>"
        end = "</CORE_MEMORIES>"
        core_block = "\n".join([start, "# Core Memories（最高优先级）", "", core_text, end]).strip()
        rest = _strip_block(base_text, start=start, end=end)
        new_text = f"{core_block}\n\n{rest}".strip() if rest else core_block
        if new_text == base_text:
            return handler(request)
        return handler(request.override(system_message=SystemMessage(content=new_text)))


def create_memory_middleware(project_root: Path) -> CoreMemoryMiddleware:
    return CoreMemoryMiddleware(project_root=project_root)


class MemoryManager:
    def __init__(self, *, project_root: Path, model_name: str) -> None:
        self.project_root = project_root.resolve()
        self.model_name = model_name
        self.session_id = uuid.uuid4().hex[:12]
        self._turn = 0
        self._lock = threading.Lock()
        self.chat_store = ChatLogStore(chats_dir=paths.chats_dir(self.project_root), session_id=self.session_id)
        self.pageindex_store = PageIndexStore(root_dir=paths.pageindex_chats_dir(self.project_root))
        self.worker = PageIndexWorker(store=self.pageindex_store, default_namespace=())

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
        realtime = (os.environ.get("AGENT_PAGEINDEX_REALTIME") or "").strip().lower()
        if realtime in {"1", "true", "yes", "on"}:
            self.worker.enqueue(path)
        return path
