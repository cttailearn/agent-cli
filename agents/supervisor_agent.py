from __future__ import annotations

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware

from .runtime import _format_tools, _init_model


def build_supervisor_agent(
    *,
    model_name: str,
    skill_middleware: SkillMiddleware,
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
):
    model = _init_model(model_name)

    tasks: dict[str, dict[str, object]] = {}

    tools: list[object] = []

    @tool
    def list_tools() -> str:
        """List available tools with short descriptions."""
        return _format_tools(tools)

    @tool
    def start_task(task_id: str, description: str) -> str:
        """Create or reset a task record in memory."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        existing = tasks.get(tid)
        if isinstance(existing, dict) and not existing.get("completed"):
            if not existing.get("description") and (description or "").strip():
                existing["description"] = (description or "").strip()
            return "OK"
        tasks[tid] = {
            "task_id": tid,
            "description": (description or "").strip(),
            "observations": [],
            "executor_outputs": [],
            "judgements": [],
            "completed": False,
        }
        return "OK"

    @tool
    def add_observation(task_id: str, note: str) -> str:
        """Append an observation note to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        obs = t.get("observations")
        if isinstance(obs, list):
            obs.append((note or "").strip())
        return "OK"

    @tool
    def record_executor_output(task_id: str, output: str) -> str:
        """Append executor output to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        outs = t.get("executor_outputs")
        if isinstance(outs, list):
            outs.append((output or "").strip())
        return "OK"

    @tool
    def add_judgement(task_id: str, judgement: str) -> str:
        """Append a judgement item to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        js = t.get("judgements")
        if isinstance(js, list):
            js.append((judgement or "").strip())
        return "OK"

    @tool
    def get_task(task_id: str) -> str:
        """Get the current task record as JSON."""
        tid = (task_id or "").strip()
        if not tid:
            return ""
        t = tasks.get(tid)
        if not t:
            return ""
        return json.dumps(t, ensure_ascii=False, sort_keys=True)

    @tool
    def mark_completed(task_id: str) -> str:
        """Mark a task record as completed."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        t["completed"] = True
        return "OK"

    @tool
    def forget_task(task_id: str) -> str:
        """Forget a task record only if completed."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Not found."
        if not t.get("completed"):
            return "Task not completed."
        tasks.pop(tid, None)
        return "OK"

    tools.extend(
        [
            *skill_middleware.tools,
            list_tools,
            start_task,
            add_observation,
            record_executor_output,
            add_judgement,
            get_task,
            mark_completed,
            forget_task,
        ]
    )

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个监督者（Supervisor）。你不与用户直接对话，你的输出将交由观察者使用。",
            "你的职责是：判断任务是否被正确、完整地执行，并指出未完成项、风险与下一步。",
            "当任务被视为复杂任务时，观察者会把 task_id 和任务描述交给你。",
            "你必须把任务记忆写入工具：start_task/add_observation/record_executor_output/add_judgement。",
            "在任务未完成前，严禁遗忘任务记忆；只有当任务完成并且 mark_completed 被调用后，才允许 forget_task。",
            "严禁虚构已执行的动作：你只能依据执行者提供的可验证结果判断完成情况。",
            f"项目目录：{project_root.as_posix()}",
            f"输出目录：{output_dir.as_posix()}",
            f"工作目录：{work_dir.as_posix()}",
        ]
    )

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[skill_middleware],
    )
    return agent, tools
