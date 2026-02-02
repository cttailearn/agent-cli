from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware
from tools import (
    Bash,
    Read,
    Write,
    delete_path,
    edit_file,
    glob_paths,
    grep,
    list_dir,
    read_file,
    run_cli,
    write_file,
    write_project_file,
)

from .runtime import _format_tools, _init_model


def build_executor_agent(
    *,
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
    model_name: str,
    skill_middleware: SkillMiddleware,
    mcp_tools: list[object],
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
            f"你通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"你可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"你可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
        ]
    )

    return create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[skill_middleware],
    )


def executor_tools(*, mcp_tools: list[object], skill_middleware: SkillMiddleware) -> list[object]:
    return [
        Read,
        Write,
        edit_file,
        glob_paths,
        grep,
        Bash,
        write_file,
        read_file,
        list_dir,
        write_project_file,
        delete_path,
        run_cli,
        *mcp_tools,
        *skill_middleware.tools,
    ]
