from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

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
    memory_core_write,
    memory_episodic_append,
    memory_user_write,
    read_file,
    run_cli,
    write_file,
    write_project_file,
)

from memory import load_core_prompt

from .runtime import UnifiedAgentState, _format_tools, _init_model


def build_executor_agent(
    *,
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

    tools: list[object] = []

    @tool
    def list_tools() -> str:
        """List available tools with short descriptions."""
        return _format_tools(tools)

    tools.extend(executor_tools(mcp_tools=mcp_tools, skill_middleware=skill_middleware))
    tools.append(list_tools)

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个执行者（Executor）。你不与用户直接对话，你的输出将交由观察者转述给用户。",
            "你的职责是：使用工具与技能完成任务，并返回可验证的执行结果或明确的失败原因。",
            "当你需要某个技能的详细规则与示例时，调用 load_skill(name) 获取该技能的完整内容，再执行任务。",
            "除非确有必要，否则不要加载技能全文。",
            "严禁虚构已执行的动作：只有在实际调用工具成功后，才可以声称“已创建/已运行/已完成”。",
            "如果工具不可用或执行失败，必须明确说明未完成，并给出可执行的替代方案或下一步。",
            "你的输出可能会被直接展示给用户：不得提及“执行者/观察者/监督者/委派/转述/工具调用/内部流程”等字样，不得暴露多智能体结构。",
            "直接给出结论、关键证据（路径/退出码/摘要）、下一步建议；保持简洁、可验证。",
            "灵魂/特性/身份：始终以项目 memory/ 下的 core 记忆为准；需要更新时可使用 memory_core_append。",
            "任务需要回忆历史事实时，优先让观察者使用 memory_kg_recall 检索知识图谱再决策。",
            f"你通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"你可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"你可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
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


def executor_tools(*, mcp_tools: list[object], skill_middleware: SkillMiddleware) -> list[object]:
    return [
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
        memory_episodic_append,
        memory_user_write,
        *mcp_tools,
        *skill_middleware.tools,
    ]
