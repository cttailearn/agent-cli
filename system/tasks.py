from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from agents.runtime import _run_agent_to_text, _summarize_tool_output_for_terminal, stream_nested_agent_reply
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
        if self.on_complete is not None:
            try:
                self.on_complete(final_text)
            except Exception:
                pass
        return final_text


@dataclass(frozen=True, slots=True)
class MemoryRollupTask(SystemTask):
    id: str
    schedule: Schedule

    def run(self, ctx: SystemContext) -> str:
        ctx.memory_manager.build_due_rollups(now_dt=datetime.now().astimezone())
        return "OK"
