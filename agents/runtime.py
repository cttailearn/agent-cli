from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk, ToolMessage, ToolMessageChunk
from langchain_deepseek import ChatDeepSeek

from skills_support import create_skill_middleware
from tools import load_mcp_tools_from_config


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


def _extract_text(msg: BaseMessage | BaseMessageChunk) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _summarize_tool_output_for_terminal(raw: str) -> str:
    lines: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("path: "):
            lines.append(s)
            continue
        if s.startswith("Wrote file: "):
            lines.append(s)
            continue
        if s.startswith("Edited: "):
            lines.append(s)
            continue
        if s.startswith("Deleted: "):
            lines.append(s)
            continue
    return "\n".join(lines)


def _run_agent_to_text(agent, messages: list[dict[str, str]]) -> tuple[str, str]:
    chunks: list[str] = []
    tool_chunks: list[str] = []
    for event in agent.stream({"messages": messages}, stream_mode="messages"):
        if isinstance(event, tuple) and event:
            msg = event[0]
        else:
            msg = event
        if isinstance(msg, AIMessageChunk):
            text = _extract_text(msg)
            if text:
                chunks.append(text)
        elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
            text = _extract_text(msg)
            if text:
                tool_chunks.append(text)
    return "".join(chunks), "".join(tool_chunks)


def _tool_name(t: object) -> str:
    name = getattr(t, "name", None)
    if isinstance(name, str) and name:
        return name
    fallback = getattr(t, "__name__", None)
    if isinstance(fallback, str) and fallback:
        return fallback
    return t.__class__.__name__


def _tool_description(t: object) -> str:
    description = getattr(t, "description", None)
    if isinstance(description, str) and description.strip():
        return description.strip()
    doc = getattr(t, "__doc__", None)
    if isinstance(doc, str) and doc.strip():
        return doc.strip()
    return ""


def _format_tools(tools: Iterable[object]) -> str:
    lines: list[str] = []
    for t in tools:
        name = _tool_name(t)
        desc = _tool_description(t)
        if desc:
            lines.append(f"- {name}: {desc}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines) if lines else "No tools found."


def _init_model(model_name: str):
    normalized_model = normalize_model_name(model_name)
    if normalized_model.startswith("deepseek:"):
        deepseek_model_name = normalized_model.split(":", 1)[1]
        return ChatDeepSeekThinkingTools(model=deepseek_model_name, streaming=True)
    return init_chat_model(model=normalized_model, streaming=True)


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
    mcp_tools = load_mcp_tools_from_config()

    from .executor_agent import build_executor_agent, executor_tools
    from .observer_agent import build_observer_agent
    from .supervisor_agent import build_supervisor_agent

    executor_agent = build_executor_agent(
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
        skill_middleware=skill_middleware,
        mcp_tools=mcp_tools,
    )

    executor_tools_list = executor_tools(mcp_tools=mcp_tools, skill_middleware=skill_middleware)

    supervisor_agent, supervisor_tools = build_supervisor_agent(
        model_name=model_name,
        skill_middleware=skill_middleware,
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
    )

    observer_agent = build_observer_agent(
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
        skill_middleware=skill_middleware,
        executor_agent=executor_agent,
        executor_tools=executor_tools_list,
        supervisor_agent=supervisor_agent,
        supervisor_tools=supervisor_tools,
    )

    return observer_agent, skill_catalog_text, skill_count
