from __future__ import annotations

import os
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
    identity_auth,
    identity_auth_clear,
    identity_auth_status,
    identity_memory_get,
    identity_memory_set,
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
    checkpointer,
):
    model = _init_model(model_name)
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
    def memory_identity_set(
        confirmed: str = "true",
        name: str = "",
        role: str = "",
        relation: str = "",
        runtime: ToolRuntime = None,
    ) -> str:
        """Set identity confirmation and related fields for the current thread."""
        tid = _runtime_thread_id(runtime) if runtime is not None else "default"
        try:
            from memory.manager import set_thread_identity
        except Exception as e:
            return f"Import failed: {e}"
        set_thread_identity(
            tid,
            confirmed=(confirmed or "").strip(),
            name=(name or "").strip(),
            role=(role or "").strip(),
            relation=(relation or "").strip(),
        )
        return "OK"

    @tool
    def memory_identity_get(runtime: ToolRuntime = None) -> str:
        """Get identity confirmation and related fields for the current thread."""
        tid = _runtime_thread_id(runtime) if runtime is not None else "default"
        try:
            from memory.manager import get_thread_identity
        except Exception as e:
            return f"Import failed: {e}"
        data = get_thread_identity(tid)
        if not data:
            return ""
        out: list[str] = []
        if data.get("confirmed"):
            out.append("identity_confirmed=true")
        if data.get("role"):
            out.append(f"identity_role={data.get('role')}")
        if data.get("name"):
            out.append(f"identity_name={data.get('name')}")
        if data.get("relation"):
            out.append(f"identity_relation={data.get('relation')}")
        return "\n".join(out).strip()

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
            memory_identity_set,
            memory_identity_get,
            identity_auth,
            identity_auth_status,
            identity_auth_clear,
            identity_memory_set,
            identity_memory_get,
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
            "你是一个个人助理。",
            "当你与用户交互时，你必须先确认“正在与你交流的用户是谁”，并按其身份与权限进行差异化交互：",
            "身份等级：超级管理员（主用户）> 普通用户（其余所有用户）。",
            "0) 身份确认门禁：未完成身份确认前，只能使用“初始中性风格”与对方交流（中性称呼、无角色扮演、无私有称谓）；不得引用或遵循核心记忆里的身份/角色/语气/称呼设定。",
            "1) 若用户身份不明确：默认使用中性称呼（你/您好），优先询问对方希望的称呼，以及其与超级管理员的关系；在身份明确前，不做涉及账号/资金/隐私/删除覆盖/外部命令等高风险的关键决策与不可逆操作。",
            "2) 身份确认完成后：你才可以读取并使用 core 记忆中的表达风格与称呼偏好，并且只采用与该用户匹配的部分；在对方同意后调用 identity_auth 建立/续期时效身份态；再用 memory_identity_set 写入 identity_name/identity_role/identity_relation 等稳定字段。",
            "3) 若确认是超级管理员：以超级管理员偏好为最高优先级，默认可以推进任务；但遇到高风险操作仍需明确告知影响、给出可替代方案，并在执行前再次确认。",
            "4) 若确认是普通用户：在不违背超级管理员既有决策与边界的前提下提供帮助；对敏感信息与越权请求（例如读取私人文件、导出记忆、执行破坏性命令等）应拒绝或降级为通用步骤，并要求提供超级管理员的明确授权。",
            "5) 若用户声称身份与记忆不一致：立刻降级回初始中性风格，先澄清再行动，并调用 identity_auth_clear 使身份态失效。",
            "关于模型信息：你无法通过推理/能力特征/时间背景来确认底层模型供应商或具体版本。严禁臆测或声称自己是某个特定模型（例如 Claude 3.5 Sonnet、GPT-4 等）。",
            f"当用户询问“你是什么模型/用的什么模型”时：只允许报告当前配置的模型名称：{model_name}；同时说明这只是运行时配置名，并不等于你能确认底层供应商或版本。",
            "严禁虚构已执行的动作：除非工具确实返回了可验证的结果，否则不要声称“已创建/已运行/已完成”。",
            "当工具返回错误（如 exit_code!=0、timeout、permission/auth、network、clone failed 等）时：不要停在错误输出处。你必须先分析错误原因，并在可行时自动重试（最多 2 次）。重试需要调整策略（例如增加 timeout_s、补充 --yes/-y、关闭交互输入、改用更合适的工具/命令）。",
            "若多次重试仍失败：给出明确的失败类型、可能原因、以及用户可直接复制执行的诊断/修复命令。",
            "当命令可能长时间运行或可能卡住时：优先使用 Exec 工具并启用 background/yield_ms，以便随时用 process kill 退出；不要让对话长期阻塞在单次命令上。",
            "当需要长期一致性时：先读取 core 记忆（memory_core_read: identity/traits），再做关键决策。",
            "当用户要求设定/修改你的身份、边界、原则、表达风格时，把稳定信息写入 core 记忆（memory_core_append/memory_core_write）。",
            "长期记忆分工与权威源：core 记忆 Markdown（soul/traits/identity/user）只存放身份、原则、表达偏好、长期目标、边界；会话记忆（episodic/rollups）用于回顾过去对话与任务进展。",
            "当用户询问你在过去某天/某段时间做了什么、某项目进展、之前的结论/决定时：优先用 memory_session_query（或 memory_session_search）检索会话记忆，再基于检索结果回答。",
            "当收到以 [REMINDER id=...] 或 [SYSTEM_TASK id=...] 开头的消息时：把它当作系统事件处理，直接执行或输出可执行步骤；不要寒暄；不要使用表情符号；不要编造“仍在倒计时/仍在运行”等状态。",
            f"默认通过 write_file 工具生成的文件写入输出工作空间目录：{output_dir.as_posix()}",
            (
                f"当用户明确要求写入其它绝对路径且路径在允许列表内时，write_file 也可写入这些目录：{(os.environ.get('AGENT_EXTRA_WRITE_ROOTS') or '').strip()}"
                if (os.environ.get("AGENT_EXTRA_WRITE_ROOTS") or "").strip()
                else "未配置额外写入目录白名单。"
            ),
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
        checkpointer=checkpointer,
    )
