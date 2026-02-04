from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from . import paths
from .storage import ChatLogStore, KnowledgeGraphStore
from .worker import KnowledgeGraphWorker


def ensure_memory_scaffold(project_root: Path) -> dict[str, str]:
    root = paths.memory_root(project_root)
    core = paths.core_dir(project_root)
    chats = paths.chats_dir(project_root)
    episodic = paths.episodic_dir(project_root)
    kg = paths.kg_dir(project_root)

    root.mkdir(parents=True, exist_ok=True)
    core.mkdir(parents=True, exist_ok=True)
    chats.mkdir(parents=True, exist_ok=True)
    episodic.mkdir(parents=True, exist_ok=True)
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


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    lines = (text or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, (text or "")
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, (text or "")

    meta: dict[str, str] = {}
    for raw in lines[1:end_idx]:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip("'\"")
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body


def _iso_utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def _render_front_matter(meta: dict[str, str]) -> str:
    ordered = ["summary", "keywords", "count", "created_at", "updated_at"]
    lines = ["---"]
    for k in ordered:
        v = (meta.get(k) or "").strip()
        if v:
            lines.append(f"{k}: {v}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _count_entries(body: str) -> int:
    n = 0
    for line in (body or "").splitlines():
        if line.strip().startswith("#### "):
            n += 1
    return n


def _generate_episodic_meta(*, model_name: str, body: str) -> tuple[str, str]:
    text = (body or "").strip()
    if not text:
        return "", ""
    try:
        from .model import init_model
    except Exception:
        init_model = None
    if init_model is None:
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        return first[:80], ""
    model = init_model(model_name)
    sys_text = "\n".join(
        [
            "你是记忆索引生成器。输入是一段 Markdown，包含多条长期记忆条目。",
            "请输出严格 JSON（不要 Markdown、不要注释、不要额外文本）：",
            '{ "summary": "一句话概括（<=40字）", "keywords": ["关键词1","关键词2"] }',
            "要求：summary 必须是单句；keywords 3~10 个，尽量短，去重，不要空字符串。",
        ]
    )
    try:
        resp = model.invoke([SystemMessage(content=sys_text), HumanMessage(content=text)])
    except Exception:
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        return first[:80], ""
    raw = getattr(resp, "content", "") or ""
    obj = _safe_json_loads(str(raw))
    if not obj:
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        return first[:80], ""
    summary = obj.get("summary")
    keywords = obj.get("keywords")
    s = summary.strip() if isinstance(summary, str) else ""
    kws: list[str] = []
    if isinstance(keywords, list):
        for k in keywords:
            if isinstance(k, str):
                kk = k.strip()
                if kk and kk not in kws:
                    kws.append(kk)
    kw_text = ", ".join(kws[:10])
    if len(s) > 120:
        s = s[:120]
    return s, kw_text


def append_episodic_memory(*, project_root: Path, model_name: str, content: str) -> tuple[bool, str]:
    text = (content or "").strip()
    if not text:
        return False, "Empty content."

    root = paths.episodic_dir(project_root)
    root.mkdir(parents=True, exist_ok=True)

    latest: Path | None = None
    for p in sorted(root.glob("*.md"), key=lambda x: x.stat().st_mtime_ns if x.exists() else 0, reverse=True):
        latest = p
        break

    target = latest
    if target is not None:
        try:
            existing_text = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            existing_text = ""
        meta, body = _parse_front_matter(existing_text)
        cnt_raw = (meta.get("count") or "").strip()
        try:
            cnt = int(cnt_raw) if cnt_raw else _count_entries(body)
        except Exception:
            cnt = _count_entries(body)
        if cnt >= 5:
            target = None

    now = _iso_utc_now()
    if target is None:
        target = root / f"ep-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}.md"
        meta = {"created_at": now, "count": "0"}
        body = ""
    else:
        try:
            existing_text = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            existing_text = ""
        meta, body = _parse_front_matter(existing_text)

    entry_header = f"#### {now}"
    new_body = (body or "").rstrip()
    if new_body:
        new_body = f"{new_body}\n\n{entry_header}\n\n{text}\n"
    else:
        new_body = f"{entry_header}\n\n{text}\n"

    cnt_raw = (meta.get("count") or "").strip()
    try:
        cnt = int(cnt_raw) if cnt_raw else _count_entries(new_body)
    except Exception:
        cnt = _count_entries(new_body)
    if cnt_raw:
        cnt = max(cnt, int(cnt_raw) + 1)
    meta["count"] = str(min(cnt, 5))
    meta["updated_at"] = now
    if not (meta.get("created_at") or "").strip():
        meta["created_at"] = now

    summary, keywords = _generate_episodic_meta(model_name=model_name, body=new_body)
    if summary:
        meta["summary"] = summary
    if keywords:
        meta["keywords"] = keywords

    out_text = f"{_render_front_matter(meta)}{new_body}".rstrip() + "\n"
    try:
        target.write_text(out_text, encoding="utf-8", errors="replace")
    except OSError as e:
        return False, f"Write failed: {e}"
    return True, target.as_posix()


def _read_episodic_index(project_root: Path) -> list[dict[str, str]]:
    root = paths.episodic_dir(project_root)
    if not root.exists():
        return []
    items: list[dict[str, str]] = []
    for p in sorted(root.glob("*.md"), key=lambda x: x.stat().st_mtime_ns if x.exists() else 0, reverse=True):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        meta, _ = _parse_front_matter(text)
        items.append(
            {
                "file": p.name,
                "path": p.as_posix(),
                "summary": (meta.get("summary") or "").strip(),
                "keywords": (meta.get("keywords") or "").strip(),
                "count": (meta.get("count") or "").strip(),
                "created_at": (meta.get("created_at") or "").strip(),
                "updated_at": (meta.get("updated_at") or "").strip(),
            }
        )
    return items


def _extract_query_terms(text: str) -> set[str]:
    t = (text or "").strip().lower()
    if not t:
        return set()
    parts = re.findall(r"[a-z0-9_./\\-]+|[\u4e00-\u9fff]+", t)
    out: set[str] = set()
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if len(s) == 1 and not re.match(r"[\u4e00-\u9fff]", s):
            continue
        out.add(s)
    return out


def _match_episodic_meta(query: str, meta: dict[str, str]) -> int:
    q = (query or "").strip()
    if not q:
        return 0
    terms = _extract_query_terms(q)
    if not terms:
        return 0
    hay = " ".join([(meta.get("summary") or ""), (meta.get("keywords") or "")]).lower()
    score = 0
    for term in terms:
        if term in hay:
            score += 2
        else:
            for kw in (meta.get("keywords") or "").lower().split(","):
                k = kw.strip()
                if k and term == k:
                    score += 3
                    break
    return score


def build_episodic_prompt(project_root: Path, user_query: str) -> str:
    index = _read_episodic_index(project_root)
    if not index:
        return "暂无长期分片记忆。"

    latest = index[0]
    older = index[1:]

    lines: list[str] = []
    latest_path = Path(latest["path"])
    try:
        latest_text = latest_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        latest_text = ""
    _, latest_body = _parse_front_matter(latest_text)
    if latest_body.strip():
        lines.append("### 最新分片（完整加载）")
        lines.append(latest_body.strip())

    if older:
        lines.append("")
        lines.append("### 历史分片（仅元数据）")
        max_meta = 80
        for item in older[:max_meta]:
            f = item.get("file") or ""
            summary = item.get("summary") or ""
            keywords = item.get("keywords") or ""
            count = item.get("count") or ""
            meta_bits = []
            if summary:
                meta_bits.append(summary)
            if keywords:
                meta_bits.append(f"keywords: {keywords}")
            if count:
                meta_bits.append(f"count: {count}")
            meta_text = " | ".join(meta_bits).strip()
            lines.append(f"- {f}{(': ' + meta_text) if meta_text else ''}")
        if len(older) > max_meta:
            lines.append(f"- ...(truncated, total={len(older)})")

    q = (user_query or "").strip()
    if q and older:
        scored: list[tuple[int, dict[str, str]]] = []
        for item in older:
            s = _match_episodic_meta(q, item)
            if s > 0:
                scored.append((s, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = int((os.environ.get("AGENT_MEMORY_EPISODIC_MATCH_LIMIT") or "2").strip() or "2")
        limit = max(0, min(5, limit))
        picked = [it for _, it in scored[:limit]]
        if picked:
            lines.append("")
            lines.append("### 按相关性补充加载（完整）")
            for item in picked:
                p = Path(item["path"])
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                _, body = _parse_front_matter(text)
                body = body.strip()
                if not body:
                    continue
                lines.append(f"#### {item.get('file')}")
                lines.append(body)

    return "\n".join(lines).strip()


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


def _extract_last_user_text(request: ModelRequest) -> str:
    msgs = getattr(request, "messages", None)
    if not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        mtype = getattr(m, "type", None)
        if isinstance(mtype, str) and mtype.lower() in {"human", "user"}:
            c = getattr(m, "content", None)
            return c if isinstance(c, str) else ""
    return ""


class EpisodicMemoryMiddleware(AgentMiddleware):
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def wrap_model_call(self, request: ModelRequest, handler):
        base_text = _extract_system_text(request.system_message)
        marker = "## Available Memories"
        query = _extract_last_user_text(request)
        mem_text = build_episodic_prompt(self.project_root, query)

        mem_addendum = "\n".join(
            [
                "",
                marker,
                "",
                mem_text,
                "",
                "说明：默认完整加载最新分片；历史分片仅提供元数据，并按当前问题相关性补充加载。",
            ]
        ).strip("\n")

        if marker in base_text:
            prefix = base_text.split(marker, 1)[0].rstrip()
            new_text = f"{prefix}\n\n{mem_addendum}".strip()
        else:
            base = base_text.strip()
            new_text = f"{base}\n\n{mem_addendum}".strip() if base else mem_addendum

        if new_text == base_text:
            return handler(request)
        modified_request = request.override(system_message=SystemMessage(content=new_text))
        return handler(modified_request)


def create_memory_middleware(project_root: Path) -> EpisodicMemoryMiddleware:
    return EpisodicMemoryMiddleware(project_root=project_root)


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
