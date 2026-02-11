from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from . import paths
from .model import init_model


def ensure_memory_scaffold(project_root: Path) -> dict[str, str]:
    root = paths.memory_root(project_root)
    core = paths.core_dir(project_root)
    episodic_dir = paths.langgraph_store_path(project_root).parent

    root.mkdir(parents=True, exist_ok=True)
    core.mkdir(parents=True, exist_ok=True)
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
        self._lock = threading.Lock()
        self._model = None

    def start(self) -> None:
        return

    def stop(self, *, flush: bool = True) -> None:
        return

    def _session_input_tokens_threshold(self) -> int:
        raw = (os.environ.get("AGENT_SESSION_MEMORY_INPUT_TOKENS_THRESHOLD") or "").strip()
        if not raw:
            return 0
        try:
            v = int(raw)
        except ValueError:
            return 0
        return max(0, v)

    def _session_md_path(self, *, dt: datetime) -> Path:
        root = paths.memory_root(self.project_root)
        d = (root / "sessions").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return (d / f"{dt.date().isoformat()}.md").resolve()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        self._model = init_model(self.model_name)
        return self._model

    def _extract_json_object(self, text: str) -> dict[str, object] | None:
        s = (text or "").strip()
        if not s:
            return None
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        i = s.find("{")
        j = s.rfind("}")
        if i < 0 or j < 0 or j <= i:
            return None
        try:
            parsed = json.loads(s[i : j + 1])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_session_items(self, *, user_text: str, assistant_text: str) -> list[dict[str, str]]:
        from langchain_core.messages import HumanMessage

        model = self._ensure_model()
        system = SystemMessage(
            content=(
                "你是一个会话记忆提取器。给定一轮对话（用户+助手），提取可复用的信息用于“会话记忆（Episodic）”。"
                "会话记忆必须以“场景”组织，便于之后用自然语言回忆（例如：昨天你做了什么？上次我们讨论 XX 得出什么结论？）。"
                "仅输出 JSON，不要输出任何多余文本。\n"
                'JSON 格式：{"items":[{"scene":"<场景>","content":"<关键信息>","kind":"<类型>"}]}。\n'
                "要求：\n"
                "- scene：4~24 个字，场景标题，尽量具体（围绕任务/问题/决策/进展/协作），去重，不要包含日期时间。\n"
                "- content：1~4 句，写清在该场景里发生了什么（事实/约束/决定/偏好/计划/结论），允许包含关键对象名。\n"
                "- content 必须尽量保留可复用的精确信息：数字/版本号/端口/路径/文件名/函数名/命令/参数/URL/错误码/配置键/阈值/ID 等，不要把这些信息泛化成“若干/一些/大概”。\n"
                "- 若出现对比或变更：写清“从什么变成什么”。\n"
                "- 不要编造对话中不存在的具体值；若对话未给出具体数字/路径/报错信息，则不要补全。\n"
                "- kind：可选，取值之一：fact|constraint|decision|preference|plan|result。\n"
                "- items 数量 3~12，信息不足时可少于 3。\n"
            )
        )
        human = HumanMessage(content=f"用户：\n{(user_text or '').strip()}\n\n助手：\n{(assistant_text or '').strip()}\n")
        try:
            resp = model.invoke([system, human])
        except Exception:
            return []
        text = getattr(resp, "content", "") if resp is not None else ""
        if not isinstance(text, str):
            text = str(text)
        obj = self._extract_json_object(text)
        if obj is None:
            return []
        items = obj.get("items")
        if not isinstance(items, list):
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            k = it.get("scene")
            if not isinstance(k, str) or not k.strip():
                k = it.get("keyword")
            c = it.get("content")
            kd = it.get("kind")
            if not isinstance(k, str) or not isinstance(c, str):
                continue
            k = k.strip()
            c = c.strip()
            if not k or not c:
                continue
            lk = k.lower()
            if lk in seen:
                continue
            seen.add(lk)
            kind = kd.strip() if isinstance(kd, str) else ""
            if kind and kind not in {"fact", "constraint", "decision", "preference", "plan", "result"}:
                kind = ""
            out.append({"keyword": k, "content": c, "kind": kind})
        return out

    def _append_session_md(self, *, dt: datetime, items: list[dict[str, str]]) -> Path | None:
        if not items:
            return None
        p = self._session_md_path(dt=dt)
        now_str = dt.strftime("%H:%M:%S")

        def _split_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
            lines = (text or "").splitlines()
            preamble: list[str] = []
            sections: list[tuple[str, list[str]]] = []
            key_to_index: dict[str, int] = {}
            current_index: int | None = None
            in_sections = False
            for line in lines:
                if line.startswith("## "):
                    in_sections = True
                    k = line[3:].strip()
                    lk = k.lower()
                    idx = key_to_index.get(lk)
                    if idx is None:
                        sections.append((k, []))
                        idx = len(sections) - 1
                        key_to_index[lk] = idx
                    current_index = idx
                    continue
                if not in_sections:
                    preamble.append(line)
                    continue
                if current_index is None:
                    continue
                sec_key, sec_lines = sections[current_index]
                sec_lines.append(line)
                sections[current_index] = (sec_key, sec_lines)
            return preamble, sections

        def _render(preamble: list[str], sections: list[tuple[str, list[str]]]) -> str:
            out: list[str] = []
            if preamble:
                while preamble and not preamble[-1].strip():
                    preamble.pop()
                out.extend(preamble)
                out.append("")
            for key, body_lines in sections:
                k = (key or "").strip()
                if not k:
                    continue
                out.append(f"## {k}")
                out.append("")
                cleaned = list(body_lines or [])
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                if cleaned:
                    out.extend(cleaned)
                    out.append("")
                out.append("")
            while out and not out[-1].strip():
                out.pop()
            return "\n".join(out).rstrip() + "\n"

        def _find_section_index(sections: list[tuple[str, list[str]]], key: str) -> int | None:
            lk = key.strip().lower()
            for i, (k, _) in enumerate(sections):
                if k.strip().lower() == lk:
                    return i
            return None

        try:
            existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            preamble, sections = _split_sections(existing)
            if not existing.strip():
                preamble = [f"# 会话记忆 {dt.date().isoformat()}"]
                sections = []
            else:
                if not any((ln or "").strip() for ln in preamble):
                    preamble = [f"# 会话记忆 {dt.date().isoformat()}"]

            for it in items:
                k = (it.get("keyword") or "").strip()
                c = (it.get("content") or "").strip()
                kind = (it.get("kind") or "").strip()
                if not k or not c:
                    continue
                prefix = f"[{kind}] " if kind else ""
                bullet = f"- ({now_str}) {prefix}{c}".rstrip()

                idx = _find_section_index(sections, k)
                if idx is None:
                    sections.append((k, [bullet]))
                    continue
                sec_key, body = sections[idx]
                body_lines = list(body or [])
                if bullet in body_lines:
                    continue
                body_lines.append(bullet)
                sections[idx] = (sec_key, body_lines)

            new_text = _render(preamble, sections)
            if not new_text.strip():
                return None
            p.write_text(new_text, encoding="utf-8", errors="replace")
        except OSError:
            return None
        return p

    def _truncate_carryover(self, text: str, *, max_chars: int) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        if max_chars <= 0:
            return ""
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 1].rstrip() + "…"

    def _rotate_thread_with_carryover(self, *, user_text: str, assistant_text: str, dt: datetime) -> None:
        new_tid = uuid.uuid4().hex[:12]
        os.environ["AGENT_THREAD_ID"] = new_tid
        payload = {
            "ts": dt.isoformat(),
            "user": self._truncate_carryover(user_text, max_chars=4000),
            "assistant": self._truncate_carryover(assistant_text, max_chars=4000),
        }
        try:
            os.environ["AGENT_THREAD_CARRYOVER"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            os.environ["AGENT_THREAD_CARRYOVER"] = ""

    def record_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        token_usage: dict[str, int] | None = None,
    ) -> Path | None:
        threshold = self._session_input_tokens_threshold()
        if threshold <= 0:
            return None
        input_tokens = int((token_usage or {}).get("input_tokens") or 0)
        dt = datetime.now().astimezone()
        with self._lock:
            items = self._extract_session_items(user_text=user_text, assistant_text=assistant_text)
            p = self._append_session_md(dt=dt, items=items)
            if p is not None and input_tokens > threshold:
                self._rotate_thread_with_carryover(user_text=user_text, assistant_text=assistant_text, dt=dt)
            return p
