from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from agents.runtime import _run_agent_to_text, _summarize_tool_output_for_terminal
from memory import paths
from memory.manager import MemoryManager

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
        assistant_text, tool_output = _run_agent_to_text(ctx.observer_agent, [{"role": "user", "content": user_text}])
        summarized = _summarize_tool_output_for_terminal(tool_output)
        final_text = assistant_text
        if summarized:
            final_text = "\n".join([summarized, assistant_text]).strip()
        ctx.memory_manager.record_turn(user_text=user_text, assistant_text=final_text)
        return final_text


@dataclass(frozen=True, slots=True)
class KnowledgeGraphBackfillTask(SystemTask):
    id: str
    schedule: Schedule
    target_day: Callable[[], date] | None = None

    def run(self, ctx: SystemContext) -> str:
        d = self.target_day() if self.target_day is not None else (date.today() - timedelta(days=1))
        day_dir = paths.chats_dir(ctx.project_root) / d.strftime("%Y-%m-%d")
        if not day_dir.exists() or not day_dir.is_dir():
            text = f"[system] knowledge graph backfill: no chat dir for {d.isoformat()}"
            ctx.memory_manager.record_turn(user_text=f"[SYSTEM_TASK id={self.id}] backfill", assistant_text=text)
            return text

        md_files = sorted([p for p in day_dir.rglob("*.md") if p.is_file()])
        for p in md_files:
            try:
                ctx.memory_manager.worker.enqueue(p)
            except Exception:
                continue
        text = f"[system] knowledge graph backfill queued: day={d.isoformat()} files={len(md_files)}"
        ctx.memory_manager.record_turn(user_text=f"[SYSTEM_TASK id={self.id}] backfill", assistant_text=text)
        return text
