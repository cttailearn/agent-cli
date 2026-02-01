from __future__ import annotations

import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_deepseek import ChatDeepSeek

from skills_support import BASE_SYSTEM_PROMPT, create_skill_middleware
from tools import (
    Bash,
    delete_path,
    edit_file,
    glob_paths,
    grep,
    list_dir,
    read_file,
    Read,
    load_mcp_tools_from_config,
    run_cli,
    write_file,
    write_project_file,
    Write,
)


def normalize_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return model
    if ":" in model:
        return model
    if model == "deepseek-reasoner":
        return "deepseek:deepseek-reasoner"
    if model == "deepseek-chat":
        return "deepseek:deepseek-chat"
    return model


class ChatDeepSeekThinkingTools(ChatDeepSeek):
    def _get_request_payload(
        self,
        input_: object,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if isinstance(input_, list):
            for orig, msg in zip(input_, payload.get("messages", [])):
                if not isinstance(orig, BaseMessage) or not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                reasoning_content = orig.additional_kwargs.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    msg["reasoning_content"] = reasoning_content
        return payload


def count_skills_in_catalog_text(skill_catalog_text: str) -> int:
    if not skill_catalog_text:
        return 0
    lines = skill_catalog_text.splitlines()
    if lines and lines[0].strip().lower() == "no skills found.":
        return 0
    return sum(1 for line in lines if line.startswith("- "))


def build_agent(
    *,
    skills_dirs: list[Path],
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
    model_name: str,
) -> tuple[object, str, int]:
    skill_catalog_text, skill_middleware = create_skill_middleware(skills_dirs)
    skill_count = count_skills_in_catalog_text(skill_catalog_text)

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个智能助手，名字叫“小智”。你服务于用户，可以按需加载技能来完成用户的请求。",
            "当你需要某个技能的详细规则与示例时，调用 load_skill(name) 获取该技能的完整内容，再执行任务。",
            "除非确有必要，否则不要加载技能全文。",
            "优化使用技能和工具来完成任务"
            f"你通过 write_file 工具生成的任何文件都必须写入该目录：{output_dir.as_posix()}",
            f"你可以通过 read_file/list_dir/write_project_file/delete_path 工具读取与修改项目目录：{project_root.as_posix()}",
            f"你可以使用 Bash（或 run_cli）工具在工作目录内执行命令行命令：{work_dir.as_posix()}（受超时限制）。",
            "严禁虚构已执行的动作：只有在实际调用工具成功后，才可以声称“已创建/已运行/已完成”。",
            "如果工具不可用或执行失败，必须明确说明未完成，并给出可执行的替代方案或下一步。",
        ]
    )

    normalized_model = normalize_model_name(model_name)
    if normalized_model.startswith("deepseek:"):
        deepseek_model_name = normalized_model.split(":", 1)[1]
        model = ChatDeepSeekThinkingTools(model=deepseek_model_name, streaming=True)
    else:
        model = init_chat_model(model=normalized_model, streaming=True)

    mcp_tools = load_mcp_tools_from_config()
    agent = create_agent(
        model,
        tools=[
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
        ],
        system_prompt=system_prompt,
        middleware=[skill_middleware],
    )
    return agent, skill_catalog_text, skill_count
