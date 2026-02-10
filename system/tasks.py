from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import asyncio
import os
import time
from pathlib import Path
from typing import Callable

from agents.runtime import _run_agent_to_text, _summarize_tool_output_for_terminal, stream_nested_agent_reply
from memory import paths
from memory.manager import MemoryManager
from memory.storage import PageIndexStore

from .schedules import Schedule


@dataclass(frozen=True, slots=True)
class SystemContext:
    project_root: Path
    output_dir: Path
    work_dir: Path
    model_name: str
    observer_agent: object
    memory_manager: MemoryManager


class SystemTask:
    id: str
    schedule: Schedule

    def run(self, ctx: SystemContext) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ObserverPromptTask(SystemTask):
    id: str
    schedule: Schedule
    prompt: str | Callable[[SystemContext], str]

    def run(self, ctx: SystemContext) -> str:
        prompt = self.prompt(ctx) if callable(self.prompt) else self.prompt
        user_text = f"[SYSTEM_TASK id={self.id}] {prompt}".strip()
        assistant_text, tool_output = _run_agent_to_text(
            ctx.observer_agent,
            [{"role": "user", "content": user_text}],
            checkpoint_ns="observer_system",
            thread_id=f"system_{self.id}",
        )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        final_text = assistant_text
        if summarized:
            final_text = "\n".join([summarized, assistant_text]).strip()
        ctx.memory_manager.record_turn(user_text=user_text, assistant_text=final_text)
        return final_text


@dataclass(frozen=True, slots=True)
class AgentReminderTask(SystemTask):
    id: str
    schedule: Schedule
    message: str
    on_complete: Callable[[str], None] | None = None

    def run(self, ctx: SystemContext) -> str:
        user_text = "\n".join(
            [
                f"[REMINDER id={self.id}]",
                "这是到期提醒事件，不是闲聊。",
                "要求：直接执行提醒内容或给出可执行步骤；不要寒暄；不要使用表情符号；不要提及“倒计时/还在计时”。",
                "提醒内容：",
                self.message.strip(),
            ]
        ).strip()
        assistant_text, tool_output = stream_nested_agent_reply(
            ctx.observer_agent,
            [{"role": "user", "content": user_text}],
            label="REMINDER",
            thread_id=f"reminder_{self.id}",
        )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        final_text = assistant_text
        if summarized:
            final_text = "\n".join([summarized, assistant_text]).strip()
        ctx.memory_manager.record_turn(user_text=user_text, assistant_text=final_text)
        if self.on_complete is not None:
            try:
                self.on_complete(final_text)
            except Exception:
                pass
        return final_text


@dataclass(frozen=True, slots=True)
class KnowledgeGraphBackfillTask(SystemTask):
    id: str
    schedule: Schedule
    target_day: Callable[[], date] | None = None

    def run(self, ctx: SystemContext) -> str:
        d = self.target_day() if self.target_day is not None else (date.today() - timedelta(days=1))
        text = _extract_pageindex_for_day(ctx, d)
        ctx.memory_manager.record_turn(user_text=f"[SYSTEM_TASK id={self.id}] backfill", assistant_text=text)
        return text


@dataclass(frozen=True, slots=True)
class PageIndexDailyExtractTask(SystemTask):
    id: str
    schedule: Schedule
    target_day: Callable[[], date] | None = None

    def run(self, ctx: SystemContext) -> str:
        d = self.target_day() if self.target_day is not None else date.today()
        text = _extract_pageindex_for_day(ctx, d)
        ctx.memory_manager.record_turn(user_text=f"[SYSTEM_TASK id={self.id}] daily_extract", assistant_text=text)
        return text


def _uploads_day_dir(project_root: Path, day: str) -> Path:
    raw = (os.environ.get("AGENT_UPLOADS_DIR") or "").strip() or "memory/uploads"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return (p / day).resolve()


def _extract_pageindex_for_day(ctx: SystemContext, d: date) -> str:
    day = d.strftime("%Y-%m-%d")
    chat_day_dir = (paths.chats_dir(ctx.project_root) / day).resolve()
    uploads_dir = _uploads_day_dir(ctx.project_root, day)

    chat_files = sorted([p for p in chat_day_dir.rglob("*.md") if p.is_file()]) if chat_day_dir.exists() else []
    upload_files = sorted([p for p in uploads_dir.rglob("*") if p.is_file()]) if uploads_dir.exists() else []

    combined_dir = (ctx.project_root / "memory" / "pageindex_daily").resolve()
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_md = (combined_dir / f"{day}.md").resolve()

    parts: list[str] = [f"# Daily Memory {day}", ""]
    for p in chat_files:
        rel = ""
        try:
            rel = p.resolve().relative_to(ctx.project_root).as_posix()
        except Exception:
            rel = p.as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        parts.append(f"## Chat File: {rel}")
        parts.append("")
        parts.append(text.strip())
        parts.append("")

    for p in upload_files:
        suf = p.suffix.lower()
        rel = ""
        try:
            rel = p.resolve().relative_to(ctx.project_root).as_posix()
        except Exception:
            rel = p.as_posix()
        parts.append(f"## Upload File: {rel}")
        parts.append("")
        if suf in {".md", ".markdown", ".txt"}:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            parts.append(text.strip())
        elif suf == ".pdf":
            max_chars_raw = (os.environ.get("AGENT_PAGEINDEX_PDF_TEXT_MAX_CHARS") or "").strip()
            try:
                max_chars = int(max_chars_raw) if max_chars_raw else 200000
            except Exception:
                max_chars = 200000
            max_chars = max(1000, min(2_000_000, max_chars))
            try:
                import pymupdf

                doc = pymupdf.open(p.as_posix())
                buf: list[str] = []
                total = 0
                for page in doc:
                    if total >= max_chars:
                        break
                    t = (page.get_text() or "").strip()
                    if not t:
                        continue
                    if total + len(t) > max_chars:
                        t = t[: max_chars - total]
                    buf.append(t)
                    total += len(t)
                parts.append("\n\n".join(buf).strip())
            except Exception:
                parts.append("[binary:.pdf]")
        else:
            parts.append(f"[binary:{suf or 'unknown'}]")
        parts.append("")

    combined_md.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8", errors="replace")

    try:
        from memory.pageindex.page_index_md import md_to_tree
        from memory.pageindex.utils import ConfigLoader
    except Exception as e:
        return f"[system] pageindex daily extract failed: import_error={e}"

    opt = ConfigLoader().load()
    try:
        tree = asyncio.run(
            md_to_tree(
                md_path=combined_md.as_posix(),
                if_thinning=False,
                min_token_threshold=None,
                if_add_node_summary=str(getattr(opt, "if_add_node_summary", "yes")),
                summary_token_threshold=200,
                model=str(getattr(opt, "model", "")),
                if_add_doc_description=str(getattr(opt, "if_add_doc_description", "no")),
                if_add_node_text=str(getattr(opt, "if_add_node_text", "no")),
                if_add_node_id=str(getattr(opt, "if_add_node_id", "yes")),
            )
        )
    except Exception:
        try:
            tree = asyncio.run(
                md_to_tree(
                    md_path=combined_md.as_posix(),
                    if_thinning=False,
                    min_token_threshold=None,
                    if_add_node_summary="no",
                    summary_token_threshold=200,
                    model=str(getattr(opt, "model", "")),
                    if_add_doc_description="no",
                    if_add_node_text="yes",
                    if_add_node_id=str(getattr(opt, "if_add_node_id", "yes")),
                )
            )
        except Exception as e:
            return f"[system] pageindex daily extract failed: error={e}"

    if not isinstance(tree, dict):
        tree = {"doc_name": day, "structure": []}
    meta = tree.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    t = time.time()
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)))
    meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    meta["source_day"] = day
    meta["combined_source_path"] = combined_md.as_posix()
    meta["chat_files"] = [p.as_posix() for p in chat_files]
    meta["upload_files"] = [p.as_posix() for p in upload_files]
    tree["meta"] = meta

    store = PageIndexStore(root_dir=paths.pageindex_chats_dir(ctx.project_root))
    store.put(("daily", day), "all", tree)

    return f"[system] pageindex daily extracted: day={day} chats={len(chat_files)} uploads={len(upload_files)}"
