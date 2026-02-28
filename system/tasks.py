from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from agents.runtime import _run_agent_to_text, _summarize_tool_output_for_terminal
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
    notify_user: Callable[[dict[str, object]], None] | None = None


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
    user_id: str = "default"
    thread_id: str = "default"

    def run(self, ctx: SystemContext) -> str:
        prompt = self.prompt(ctx) if callable(self.prompt) else self.prompt
        user_text = f"[SYSTEM_TASK id={self.id}] {prompt}".strip()
        messages = [{"role": "user", "content": user_text}]
        assistant_text, tool_output = _run_agent_to_text(
            ctx.observer_agent,
            messages,
            checkpoint_ns="observer_system",
            thread_id=(self.thread_id or "").strip() or None,
            user_id=(self.user_id or "").strip() or None,
        )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        final_text = "\n".join([summarized, assistant_text]).strip() if summarized else assistant_text
        try:
            ctx.memory_manager.record_turn(user_text=user_text, assistant_text=final_text, token_usage=None)
        except Exception:
            pass
        if ctx.notify_user is not None:
            try:
                ctx.notify_user(
                    {
                        "type": "push",
                        "event": "system_task",
                        "assistant": final_text,
                        "user_id": (self.user_id or "").strip() or "default",
                    }
                )
            except Exception:
                pass
        return final_text


@dataclass(frozen=True, slots=True)
class AgentReminderTask(SystemTask):
    id: str
    schedule: Schedule
    message: str
    on_complete: Callable[[str], None] | None = None
    user_id: str = "default"
    thread_id: str = "default"

    def run(self, ctx: SystemContext) -> str:
        user_text = "\n".join(
            [
                f"[REMINDER id={self.id}]",
                "这是系统定时任务到期事件，不是闲聊。",
                "系统会把你的回复以“对话消息”的方式主动发给用户。",
                "你必须先判断这是：A) 提醒类  B) 执行类（到点要把一件事做掉）。",
                "判断规则：若内容主要是“提醒我/记得/别忘/到点提醒/提醒一下”，则为提醒类；否则默认按执行类处理。",
                "提醒类要求：只输出提醒正文（可包含可执行步骤）；不要询问用户称呼/主用户关系/身份验证；不要寒暄；不要使用表情符号；不要提及“倒计时/还在计时”。",
                "执行类要求：把内容当作用户此刻发来的任务指令，直接自动执行；优先调用工具产出可验证结果（创建/修改文件、运行命令等）；信息不足时做最合理的默认假设继续推进（例如生成通用模板/骨架），不要向用户反问需求；不要虚构已执行动作。",
                "输出要求：必须非空；若执行了工具/生成了文件，向用户汇报结果与下一步。",
                "提醒内容：",
                self.message.strip(),
            ]
        ).strip()
        messages = [{"role": "user", "content": user_text}]
        assistant_text, tool_output = _run_agent_to_text(
            ctx.observer_agent,
            messages,
            checkpoint_ns="reminder",
            thread_id=(self.thread_id or "").strip() or None,
            user_id=(self.user_id or "").strip() or None,
        )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        final_text = "\n".join([summarized, assistant_text]).strip() if summarized else assistant_text
        if self.on_complete is not None:
            try:
                self.on_complete(final_text)
            except Exception:
                pass
        try:
            ctx.memory_manager.record_turn(user_text=user_text, assistant_text=final_text, token_usage=None)
        except Exception:
            pass
        if ctx.notify_user is not None:
            try:
                ctx.notify_user(
                    {
                        "type": "push",
                        "event": "reminder",
                        "assistant": final_text,
                        "user_id": (self.user_id or "").strip() or "default",
                    }
                )
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
