from __future__ import annotations

import uuid
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware

from .runtime import _format_tools, _init_model, _run_agent_to_text, _summarize_tool_output_for_terminal


def build_observer_agent(
    *,
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
    model_name: str,
    skill_middleware: SkillMiddleware,
    executor_agent,
    executor_tools: list[object],
    supervisor_agent,
    supervisor_tools: list[object],
):
    model = _init_model(model_name)
    memory: dict[str, str] = {}
    last_supervision_task_id: str = ""

    tools: list[object] = []

    @tool
    def list_tools() -> str:
        """List available tools with short descriptions."""
        return _format_tools(tools)

    @tool
    def list_executor_tools() -> str:
        """List executor tools with short descriptions."""
        return _format_tools(executor_tools)

    @tool
    def list_supervisor_tools() -> str:
        """List supervisor tools with short descriptions."""
        return _format_tools(supervisor_tools)

    @tool
    def remember(key: str, value: str) -> str:
        """Store a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        memory[k] = (value or "").strip()
        return "OK"

    @tool
    def recall(key: str) -> str:
        """Recall a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        return memory.get(k, "")

    @tool
    def forget(key: str) -> str:
        """Delete a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        return "OK" if memory.pop(k, None) is not None else "Not found."

    def _is_complex_task(task_text: str) -> bool:
        t = (task_text or "").strip()
        if len(t) >= 300:
            return True
        if "\n" in t and t.count("\n") >= 3:
            return True
        multi_signal = 0
        for s in ["并且", "同时", "分别", "然后", "最后", "再", "以及", "另外", "多文件", "重构", "测试", "验证", "部署"]:
            if s in t:
                multi_signal += 1
        if multi_signal >= 2:
            return True
        for s in ["1.", "2.", "3.", "- ", "•"]:
            if s in t:
                return True
        return False

    @tool
    def is_complex_task(task: str) -> str:
        """Return 'true' if the task looks complex, else 'false'."""
        return "true" if _is_complex_task(task) else "false"

    @tool
    def delegate_to_executor(task: str) -> str:
        """Delegate a task to the executor agent and return its final answer."""
        task_text = (task or "").strip()
        if not task_text:
            return "Empty task."
        answer, tool_output = _run_agent_to_text(executor_agent, [{"role": "user", "content": task_text}])
        summarized = _summarize_tool_output_for_terminal(tool_output)
        if summarized:
            return f"{summarized}\n{answer}"
        return answer

    @tool
    def delegate_to_supervisor(task: str) -> str:
        """Delegate a task to the supervisor agent and return its final answer."""
        task_text = (task or "").strip()
        if not task_text:
            return "Empty task."
        answer, tool_output = _run_agent_to_text(supervisor_agent, [{"role": "user", "content": task_text}])
        summarized = _summarize_tool_output_for_terminal(tool_output)
        if summarized:
            return f"{summarized}\n{answer}"
        return answer

    @tool
    def start_supervision(task: str) -> str:
        """Start supervision for a complex task and return task_id."""
        nonlocal last_supervision_task_id
        task_text = (task or "").strip()
        if not task_text:
            return ""
        if not _is_complex_task(task_text):
            last_supervision_task_id = ""
            return ""
        task_id = uuid.uuid4().hex[:12]
        last_supervision_task_id = task_id
        _run_agent_to_text(
            supervisor_agent,
            [
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "请启动并记录一个新的任务监督。",
                            f"task_id={task_id}",
                            f"task_description={task_text}",
                            "要求：调用 start_task(task_id, task_description) 写入记忆。",
                        ]
                    ),
                }
            ],
        )
        return task_id

    @tool
    def supervised_check(task_id: str, executor_result: str) -> str:
        """Ask supervisor to judge progress based on executor output."""
        tid = (task_id or "").strip()
        if not tid:
            tid = last_supervision_task_id
        if not tid:
            return ""
        result_text = (executor_result or "").strip()
        answer, tool_output = _run_agent_to_text(
            supervisor_agent,
            [
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "请根据执行者结果判断任务完成情况，并把必要信息写入任务记忆。",
                            f"task_id={tid}",
                            "你必须：",
                            "- 调用 record_executor_output(task_id, output)",
                            "- 产出 judgement（完成/未完成、缺失项、风险、下一步）并调用 add_judgement(task_id, judgement)",
                            "执行者结果如下：",
                            result_text,
                        ]
                    ),
                }
            ],
        )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        if summarized:
            return f"{summarized}\n{answer}"
        return answer

    @tool
    def finish_supervision(task_id: str) -> str:
        """Finalize supervision, mark completed, then forget task memory."""
        nonlocal last_supervision_task_id
        tid = (task_id or "").strip()
        if not tid:
            tid = last_supervision_task_id
        if not tid:
            return ""
        answer, tool_output = _run_agent_to_text(
            supervisor_agent,
            [
                {
                    "role": "user",
                    "content": "\n".join(
                        [
                            "请在确认任务已完成后，标记任务完成并清理任务记忆。",
                            f"task_id={tid}",
                            "要求：先调用 mark_completed(task_id)，再调用 forget_task(task_id)。",
                        ]
                    ),
                }
            ],
        )
        last_supervision_task_id = ""
        summarized = _summarize_tool_output_for_terminal(tool_output)
        if summarized:
            return f"{summarized}\n{answer}"
        return answer

    tools.extend(
        [
            *skill_middleware.tools,
            list_tools,
            list_executor_tools,
            list_supervisor_tools,
            remember,
            recall,
            forget,
            is_complex_task,
            delegate_to_executor,
            delegate_to_supervisor,
            start_supervision,
            supervised_check,
            finish_supervision,
        ]
    )

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个观察者（Observer）。你负责与用户对话、理解意图、规划步骤、维护会话记忆。",
            "所有会产生副作用的执行（写文件、改代码、运行命令、调用外部工具）必须委派给执行者。",
            "你可以使用 delegate_to_executor(task) 让执行者完成具体工作，并基于其返回结果与用户沟通。",
            "复杂任务时必须启动监督者：先调用 start_supervision(task) 获取 task_id，再在每次执行后调用 supervised_check(task_id, executor_result)。",
            "只有当监督者判断任务完成后，才调用 finish_supervision(task_id) 清理监督者的任务记忆。",
            "你可以使用 remember/recall/forget 管理你自己的会话记忆。",
            "你只能使用查看与委派类工具，不得直接调用任何项目读写或命令执行类工具。",
            "严禁虚构已执行的动作：除非执行者确实通过工具产生了可验证的结果，否则不要声称“已创建/已运行/已完成”。",
            f"执行者通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"执行者可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"执行者可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
        ]
    )

    return create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[skill_middleware],
    )
