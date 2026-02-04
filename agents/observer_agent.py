from __future__ import annotations

import os
import uuid
from pathlib import Path

from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain_core.tools import tool

from skills.skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware

from memory import load_core_prompt

from agents.tools import memory_core_read, memory_kg_recall, memory_kg_stats, memory_user_read
from . import console_print

from .runtime import (
    UnifiedAgentState,
    _format_tools,
    _init_model,
    _run_agent_to_text,
    _summarize_tool_output_for_terminal,
    stream_nested_agent_reply,
)


def build_observer_agent(
    *,
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
    model_name: str,
    skill_middleware: SkillMiddleware,
    memory_middleware,
    executor_agent,
    executor_tools: list[object],
    supervisor_agent,
    supervisor_tools: list[object],
    store,
    checkpointer,
):
    model = _init_model(model_name)
    memory: dict[str, str] = {}
    shared_context: dict[str, str] = {}
    last_supervision_task_id: str = ""

    tools: list[object] = []

    def _runtime_thread_id(runtime: ToolRuntime) -> str:
        cfg = getattr(runtime, "config", None) or {}
        if isinstance(cfg, dict):
            configurable = cfg.get("configurable") or {}
            if isinstance(configurable, dict):
                tid = (configurable.get("thread_id") or "").strip()
                if tid:
                    return tid
        return "default"

    def _state_shared_context(runtime: ToolRuntime) -> dict[str, str] | None:
        st = getattr(runtime, "state", None)
        if not isinstance(st, dict):
            return None
        shared = st.get("shared")
        if not isinstance(shared, dict):
            shared = {}
            st["shared"] = shared
        ctx = shared.get("context")
        if isinstance(ctx, dict):
            out: dict[str, str] = {}
            for k, v in ctx.items():
                if isinstance(k, str) and isinstance(v, str):
                    out[k] = v
            shared["context"] = out
            return out
        out = {}
        shared["context"] = out
        return out

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
    def remember(key: str, value: str, runtime: ToolRuntime) -> str:
        """Store a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        v = (value or "").strip()
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            memory[k] = v
            return "OK"
        store_obj.put(("sessions", _runtime_thread_id(runtime)), k, {"value": v})
        return "OK"

    @tool
    def recall(key: str, runtime: ToolRuntime) -> str:
        """Recall a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return memory.get(k, "")
        item = store_obj.get(("sessions", _runtime_thread_id(runtime)), k)
        if item is None:
            return ""
        val = getattr(item, "value", None)
        if isinstance(val, dict):
            v = val.get("value")
            return v if isinstance(v, str) else ""
        return ""

    @tool
    def forget(key: str, runtime: ToolRuntime) -> str:
        """Delete a memory item for the current session."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return "OK" if memory.pop(k, None) is not None else "Not found."
        store_obj.delete(("sessions", _runtime_thread_id(runtime)), k)
        return "OK"

    @tool
    def shared_context_put(key: str, value: str, runtime: ToolRuntime) -> str:
        """Store a key/value into shared context for the current session/thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        v = (value or "").strip()
        ctx = _state_shared_context(runtime)
        if ctx is not None:
            ctx[k] = v
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            shared_context[k] = v
            return "OK"
        store_obj.put(("shared_context", _runtime_thread_id(runtime)), k, {"value": v})
        return "OK"

    @tool
    def shared_context_get(key: str, runtime: ToolRuntime) -> str:
        """Get a value from shared context for the current session/thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        ctx = _state_shared_context(runtime)
        if ctx is not None and k in ctx:
            return ctx.get(k, "")
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return shared_context.get(k, "")
        item = store_obj.get(("shared_context", _runtime_thread_id(runtime)), k)
        if item is None:
            return ""
        val = getattr(item, "value", None)
        if isinstance(val, dict):
            v = val.get("value")
            out = v if isinstance(v, str) else ""
        else:
            out = val if isinstance(val, str) else ""
        if ctx is not None and out:
            ctx[k] = out
        return out

    @tool
    def shared_context_forget(key: str, runtime: ToolRuntime) -> str:
        """Delete a key from shared context for the current session/thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        ctx = _state_shared_context(runtime)
        if ctx is not None:
            ctx.pop(k, None)
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            shared_context.pop(k, None)
            return "OK"
        store_obj.delete(("shared_context", _runtime_thread_id(runtime)), k)
        return "OK"

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
        tid = last_supervision_task_id
        if not tid and _is_complex_task(task_text):
            tid = start_supervision(task_text)
        call_thread_id = uuid.uuid4().hex[:12]
        stream = (os.environ.get("AGENT_STREAM_DELEGATION") or "").strip().lower() in {"1", "true", "yes", "on"}
        if stream:
            console_print(f"\n[observer -> executor]\n{task_text}\n", flush=True)
            answer, tool_output = stream_nested_agent_reply(
                executor_agent, [{"role": "user", "content": task_text}], label="executor", thread_id=call_thread_id
            )
        else:
            answer, tool_output = _run_agent_to_text(
                executor_agent,
                [{"role": "user", "content": task_text}],
                checkpoint_ns="executor",
                thread_id=call_thread_id,
            )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        out = f"{summarized}\n{answer}" if summarized else answer
        if tid:
            judgement = supervised_check(tid, out)
            if judgement:
                out = f"{out}\n\n{judgement}"
        return out

    @tool
    def delegate_to_supervisor(task: str) -> str:
        """Delegate a task to the supervisor agent and return its final answer."""
        task_text = (task or "").strip()
        if not task_text:
            return "Empty task."
        supervisor_thread_id = _supervision_thread_id()
        stream = (os.environ.get("AGENT_STREAM_DELEGATION") or "").strip().lower() in {"1", "true", "yes", "on"}
        if stream:
            console_print(f"\n[observer -> supervisor]\n{task_text}\n", flush=True)
            answer, tool_output = stream_nested_agent_reply(
                supervisor_agent,
                [{"role": "user", "content": task_text}],
                label="supervisor",
                thread_id=supervisor_thread_id,
            )
        else:
            answer, tool_output = _run_agent_to_text(
                supervisor_agent,
                [{"role": "user", "content": task_text}],
                checkpoint_ns="supervisor",
                thread_id=supervisor_thread_id,
            )
        summarized = _summarize_tool_output_for_terminal(tool_output)
        if summarized:
            return f"{summarized}\n{answer}"
        return answer

    @tool
    def append_core_memory(kind: str, content: str) -> str:
        """Delegate to executor to append content into core memory markdown."""
        k = (kind or "").strip()
        text = (content or "").strip()
        if not k:
            return "Missing kind."
        if not text:
            return "Empty content."
        return delegate_to_executor(
            "\n".join(
                [
                    "请将以下内容写入项目 core 记忆 Markdown（追加写入）。",
                    "要求：只调用一次 memory_core_append(kind, content)，并返回调用结果。",
                    f"kind={k}",
                    "content:",
                    text,
                ]
            )
        )

    @tool
    def append_episodic_memory(content: str) -> str:
        """Delegate to executor to append one entry into episodic fragmented memory markdown."""
        text = (content or "").strip()
        if not text:
            return "Empty content."
        return delegate_to_executor(
            "\n".join(
                [
                    "请将以下内容写入项目长期分片记忆（episodic）Markdown（追加一条）。",
                    "要求：只调用一次 memory_episodic_append(content)，并返回调用结果。",
                    "content:",
                    text,
                ]
            )
        )

    @tool
    def write_core_memory(kind: str, content: str) -> str:
        """Delegate to executor to overwrite core memory markdown."""
        k = (kind or "").strip()
        text = (content or "").strip()
        if not k:
            return "Missing kind."
        return delegate_to_executor(
            "\n".join(
                [
                    "请将以下内容写入项目 core 记忆 Markdown（覆盖写入）。",
                    "要求：只调用一次 memory_core_write(kind, content)，并返回调用结果。",
                    f"kind={k}",
                    "content:",
                    text,
                ]
            )
        )

    def _supervision_thread_id() -> str:
        base = (os.environ.get("AGENT_THREAD_ID") or "").strip()
        if not base:
            base = uuid.uuid4().hex[:12]
            os.environ["AGENT_THREAD_ID"] = base
        return f"{base}-supervisor"

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
            checkpoint_ns="supervisor",
            thread_id=_supervision_thread_id(),
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
            checkpoint_ns="supervisor",
            thread_id=_supervision_thread_id(),
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
                            "请在确认任务已完成后，生成任务快照并清理任务记忆。",
                            f"task_id={tid}",
                            "要求：调用 finalize_task(task_id, keep_record=false)。",
                        ]
                    ),
                }
            ],
            checkpoint_ns="supervisor",
            thread_id=_supervision_thread_id(),
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
            shared_context_put,
            shared_context_get,
            shared_context_forget,
            memory_core_read,
            memory_user_read,
            memory_kg_recall,
            memory_kg_stats,
            is_complex_task,
            delegate_to_executor,
            delegate_to_supervisor,
            append_core_memory,
            append_episodic_memory,
            write_core_memory,
            start_supervision,
            supervised_check,
            finish_supervision,
        ]
    )

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个观察者（Observer）。你负责与用户对话、理解意图、规划步骤、维护会话记忆。",
            "所有会产生副作用的执行（写文件、改代码、运行命令、调用外部工具）必须通过内部执行接口完成（例如 delegate_to_executor）。",
            "对用户只输出你自己的自然语言结果：不要提及内部角色、委派、监督、执行链路、工具名或任何实现细节。",
            "严禁出现或近似表达：让我委派给监督者/执行者、我去叫监督者/执行者、我把任务交给监督者/执行者、后台由监督者/执行者处理。",
            "复杂任务时必须在内部启用监督流程：先调用 start_supervision(task) 获取 task_id，并在每次执行后调用 supervised_check(task_id, executor_result)，完成后调用 finish_supervision(task_id)。",
            "你可以使用 remember/recall/forget 管理你自己的会话记忆。",
            "你可以使用 shared_context_put/shared_context_get/shared_context_forget 管理跨回合可复用的共享上下文。",
            "灵魂/特性/身份：始终以 memory_core_read 读取的 core 记忆为准，并在关键决策时对齐。",
            "每次会话开始时，先用 memory_core_read 分别读取 identity 与 traits：若内容仍是默认模板/信息为空/缺少明确的名字或表达风格，则在与你的正常回复里自然插入一次简短引导（2~4 轮问题）。引导结束后把稳定信息写入 core：identity（包含名字、边界，并追加一行 onboarding_status: done）、traits（表达风格/协作方式）、user（用户偏好与目标摘要）。",
            "当用户要求设定/修改你的身体、性格、表达风格、身份边界、原则时，你必须把稳定信息写入 core 记忆：用 append_core_memory(kind, content) 委派执行者追加写入（identity/traits/soul/user）。需要覆盖写入时，使用 write_core_memory(kind, content)。",
            "只有当 append_core_memory 返回 OK 时，你才可以对用户说“已保存/已写入”。",
            "当用户希望沉淀流程、补齐能力、优化技能生态时，在内部使用 delegate_to_supervisor 去完成 create/find/install/enable/disable 等动作；对用户只陈述最终结果与必要的操作说明。",
            "需要回忆长期事实、偏好、人物/项目关系时，先调用 memory_kg_recall(query) 检索知识图谱。",
            "你只能使用查看与委派类工具，不得直接调用任何项目读写或命令执行类工具。",
            "当你与执行者或者监督者交互时，不得告诉用户，必须让用户觉得一直在与你交互",
            "严禁虚构已执行的动作：除非执行者确实通过工具产生了可验证的结果，否则不要声称“已创建/已运行/已完成”。",
            f"执行者通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"执行者可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"执行者可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
            load_core_prompt(project_root),
        ]
    )

    return create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[skill_middleware, memory_middleware],
        state_schema=UnifiedAgentState,
        store=store,
        checkpointer=checkpointer,
    )
