from __future__ import annotations

from pathlib import Path

from langgraph.prebuilt import ToolRuntime
from langchain.agents import create_agent
from langchain_core.tools import tool

from skills.skills_manager import SkillManager
from skills.skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware

from agents.tools import (
    Bash,
    Exec,
    Process,
    Read,
    Write,
    delete_path,
    edit_file,
    glob_paths,
    grep,
    list_dir,
    memory_core_append,
    memory_core_read,
    memory_core_write,
    memory_session_query,
    memory_session_search,
    memory_user_read,
    memory_user_write,
    read_file,
    reminder_cancel,
    reminder_list,
    reminder_schedule_at,
    reminder_schedule_in,
    run_cli,
    system_time,
    write_file,
    write_project_file,
)

from .runtime import UnifiedAgentState, _format_tools, _init_model


def build_single_agent(
    *,
    skills_dirs: list[Path],
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
    model_name: str,
    skill_middleware: SkillMiddleware,
    memory_middleware,
    mcp_tools: list[object],
    store,
    checkpointer,
):
    model = _init_model(model_name)
    memory: dict[str, str] = {}
    shared_context: dict[str, str] = {}
    skill_manager = SkillManager(skills_dirs=skills_dirs, project_root=project_root, work_dir=work_dir)

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

    def _runtime_user_id(runtime: ToolRuntime) -> str:
        cfg = getattr(runtime, "config", None) or {}
        if isinstance(cfg, dict):
            configurable = cfg.get("configurable") or {}
            if isinstance(configurable, dict):
                uid = (configurable.get("user_id") or "").strip()
                if uid:
                    return uid
        return "default"

    @tool
    def list_tools() -> str:
        """List available tools with short descriptions."""
        return _format_tools(tools)

    @tool
    def skills_scan() -> str:
        """Scan available skills (including disabled) with usage stats."""
        return skill_manager.scan_json()

    @tool
    def skills_disable(name: str, reason: str = "") -> str:
        """Disable a skill (it will be hidden from the catalog)."""
        return skill_manager.disable(name, reason=reason)

    @tool
    def skills_enable(name: str) -> str:
        """Enable a previously disabled skill."""
        return skill_manager.enable(name)

    @tool
    def skills_remove(name: str) -> str:
        """Delete a skill directory from project skills dir."""
        return skill_manager.remove_from_project(name)

    @tool
    def skills_create(name: str, description: str, body: str = "") -> str:
        """Create a new local skill under project skills dir."""
        return skill_manager.create(name, description, body=body)

    @tool
    def skills_find(query: str) -> str:
        """Find skills from the skills ecosystem via npx skills."""
        return skill_manager.find_via_npx(query)

    @tool
    def skills_install(package_spec: str) -> str:
        """Install a skill into project skills dir via npx skills add."""
        return skill_manager.install_via_npx(package_spec)

    @tool
    def skills_ensure(name_or_package_spec: str) -> str:
        """Ensure a skill is installed; if missing, install via npx."""
        return skill_manager.ensure_installed(name_or_package_spec)

    @tool
    def skills_prune_disabled() -> str:
        """Delete disabled skills from project skills dir."""
        return skill_manager.prune_disabled_from_project()

    @tool
    def remember(key: str, value: str, runtime: ToolRuntime) -> str:
        """Store a long-term memory item for the current user (shared across threads)."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        v = (value or "").strip()
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            memory[k] = v
            return "OK"
        store_obj.put((_runtime_user_id(runtime), "memories"), k, {"value": v})
        return "OK"

    @tool
    def recall(key: str, runtime: ToolRuntime) -> str:
        """Recall a long-term memory item for the current user (shared across threads)."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return memory.get(k, "")
        item = store_obj.get((_runtime_user_id(runtime), "memories"), k)
        if item is None:
            return ""
        val = getattr(item, "value", None)
        if isinstance(val, dict):
            v = val.get("value")
            return v if isinstance(v, str) else ""
        return ""

    @tool
    def forget(key: str, runtime: ToolRuntime) -> str:
        """Delete a long-term memory item for the current user (shared across threads)."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return "OK" if memory.pop(k, None) is not None else "Not found."
        store_obj.delete((_runtime_user_id(runtime), "memories"), k)
        return "OK"

    @tool
    def shared_context_put(key: str, value: str, runtime: ToolRuntime) -> str:
        """Store a shared context item for the current thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        v = (value or "").strip()
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            shared_context[k] = v
            return "OK"
        store_obj.put(("shared", _runtime_thread_id(runtime)), k, {"value": v})
        return "OK"

    @tool
    def shared_context_get(key: str, runtime: ToolRuntime) -> str:
        """Get a shared context item for the current thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return shared_context.get(k, "")
        item = store_obj.get(("shared", _runtime_thread_id(runtime)), k)
        if item is None:
            return ""
        val = getattr(item, "value", None)
        if isinstance(val, dict):
            v = val.get("value")
            return v if isinstance(v, str) else ""
        return ""

    @tool
    def shared_context_forget(key: str, runtime: ToolRuntime) -> str:
        """Delete a shared context item for the current thread."""
        k = (key or "").strip()
        if not k:
            return "Missing key."
        store_obj = getattr(runtime, "store", None)
        if store_obj is None:
            return "OK" if shared_context.pop(k, None) is not None else "Not found."
        store_obj.delete(("shared", _runtime_thread_id(runtime)), k)
        return "OK"

    tools.extend(
        [
            *skill_middleware.tools,
            list_tools,
            skills_scan,
            skills_disable,
            skills_enable,
            skills_remove,
            skills_create,
            skills_find,
            skills_install,
            skills_ensure,
            skills_prune_disabled,
            remember,
            recall,
            forget,
            shared_context_put,
            shared_context_get,
            shared_context_forget,
            system_time,
            reminder_schedule_at,
            reminder_schedule_in,
            reminder_list,
            reminder_cancel,
            memory_core_read,
            memory_user_read,
            memory_session_query,
            memory_session_search,
            Read,
            Write,
            edit_file,
            glob_paths,
            grep,
            Exec,
            Process,
            Bash,
            write_file,
            read_file,
            list_dir,
            write_project_file,
            delete_path,
            run_cli,
            memory_core_append,
            memory_core_write,
            memory_user_write,
            *mcp_tools,
        ]
    )

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个智能体（Agent）。你负责与用户对话、理解意图、规划步骤，并在需要时直接执行任务（读写文件、运行命令、调用工具）。",
            "当你与用户交互时，需要根据你的核心记忆，表现得像是一个人在和用户对话，而不是一个机器。",
            "关于模型信息：你无法通过推理/能力特征/时间背景来确认底层模型供应商或具体版本。严禁臆测或声称自己是某个特定模型（例如 Claude 3.5 Sonnet、GPT-4 等）。",
            f"当用户询问“你是什么模型/用的什么模型”时：只允许报告当前配置的模型名称：{model_name}；同时说明这只是运行时配置名，并不等于你能确认底层供应商或版本。",
            "严禁虚构已执行的动作：除非工具确实返回了可验证的结果，否则不要声称“已创建/已运行/已完成”。",
            "当工具返回错误（如 exit_code!=0、timeout、permission/auth、network、clone failed 等）时：不要停在错误输出处。你必须先分析错误原因，并在可行时自动重试（最多 2 次）。重试需要调整策略（例如增加 timeout_s、补充 --yes/-y、关闭交互输入、改用更合适的工具/命令）。",
            "若多次重试仍失败：给出明确的失败类型、可能原因、以及用户可直接复制执行的诊断/修复命令。",
            "当命令可能长时间运行或可能卡住时：优先使用 Exec 工具并启用 background/yield_ms，以便随时用 process kill 退出；不要让对话长期阻塞在单次命令上。",
            "当需要长期一致性时：先读取 core 记忆（memory_core_read: identity/traits），再做关键决策。",
            "当用户要求设定/修改你的身份、边界、原则、表达风格时，把稳定信息写入 core 记忆（memory_core_append/memory_core_write）。",
            "你可以使用 remember/recall/forget 管理长期记忆（按 user_id 跨线程共享）；shared_context_put/shared_context_get/shared_context_forget 管理线程内共享上下文。",
            "当用户询问你在过去某天/某段时间做了什么、某项目进展、之前的结论/决定时：优先用 memory_session_query（或 memory_session_search）检索会话记忆，再基于检索结果回答。",
            "当收到以 [REMINDER id=...] 或 [SYSTEM_TASK id=...] 开头的消息时：把它当作系统事件处理，直接执行或输出可执行步骤；不要寒暄；不要使用表情符号；不要编造“仍在倒计时/仍在运行”等状态。",
            f"通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
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
