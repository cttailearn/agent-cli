from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk, ToolMessage, ToolMessageChunk
from langchain_deepseek import ChatDeepSeek

from skills.skills_support import create_skill_middleware
from agents.tools import load_mcp_tools_from_config


_TOKEN_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def reset_token_usage() -> None:
    _TOKEN_USAGE["input_tokens"] = 0
    _TOKEN_USAGE["output_tokens"] = 0
    _TOKEN_USAGE["total_tokens"] = 0


def get_token_usage() -> dict[str, int]:
    return dict(_TOKEN_USAGE)


def _normalize_token_usage(data: object) -> dict[str, int] | None:
    if not isinstance(data, dict):
        return None
    prompt = data.get("prompt_tokens")
    completion = data.get("completion_tokens")
    total = data.get("total_tokens")

    if prompt is None:
        prompt = data.get("input_tokens")
    if completion is None:
        completion = data.get("output_tokens")
    if total is None:
        total = data.get("total")

    if not isinstance(prompt, int):
        prompt = 0
    if not isinstance(completion, int):
        completion = 0
    if not isinstance(total, int):
        total = prompt + completion

    if prompt <= 0 and completion <= 0 and total <= 0:
        return None
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


def _extract_token_usage_from_message(msg: object) -> dict[str, int] | None:
    usage = getattr(msg, "usage_metadata", None)
    normalized = _normalize_token_usage(usage)
    if normalized is not None:
        return normalized

    response_metadata = getattr(msg, "response_metadata", None)
    if isinstance(response_metadata, dict):
        normalized = _normalize_token_usage(response_metadata.get("token_usage"))
        if normalized is not None:
            return normalized
        normalized = _normalize_token_usage(response_metadata.get("usage"))
        if normalized is not None:
            return normalized

    additional_kwargs = getattr(msg, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        normalized = _normalize_token_usage(additional_kwargs.get("token_usage"))
        if normalized is not None:
            return normalized
        normalized = _normalize_token_usage(additional_kwargs.get("usage"))
        if normalized is not None:
            return normalized

    return None


def capture_token_usage_from_message(msg: object) -> None:
    usage = _extract_token_usage_from_message(msg)
    if usage is None:
        return
    _TOKEN_USAGE["input_tokens"] += usage["input_tokens"]
    _TOKEN_USAGE["output_tokens"] += usage["output_tokens"]
    _TOKEN_USAGE["total_tokens"] += usage["total_tokens"]


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_count = 0
    cjk_count = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF:
            cjk_count += 1
        elif o < 128:
            ascii_count += 1
    return cjk_count + (ascii_count + 3) // 4


def estimate_token_usage(prompt_text: str, completion_text: str) -> dict[str, int]:
    input_tokens = _estimate_tokens(prompt_text or "")
    output_tokens = _estimate_tokens(completion_text or "")
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}


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


def _recursion_limit() -> int:
    raw = (os.environ.get("AGENT_RECURSION_LIMIT") or "").strip()
    if not raw:
        return 64
    try:
        v = int(raw)
    except ValueError:
        return 64
    return max(10, min(500, v))


def _run_agent_to_text(agent, messages: list[dict[str, str]]) -> tuple[str, str]:
    chunks: list[str] = []
    tool_chunks: list[str] = []
    try:
        from langgraph.errors import GraphRecursionError
    except Exception:
        GraphRecursionError = None

    try:
        for event in agent.stream(
            {"messages": messages},
            stream_mode="messages",
            config={"recursion_limit": _recursion_limit()},
        ):
            if isinstance(event, tuple) and event:
                msg = event[0]
            else:
                msg = event
            if isinstance(msg, AIMessageChunk):
                capture_token_usage_from_message(msg)
                text = _extract_text(msg)
                if text:
                    chunks.append(text)
            elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                text = _extract_text(msg)
                if text:
                    tool_chunks.append(text)
    except Exception as e:
        if GraphRecursionError is not None and isinstance(e, GraphRecursionError):
            chunks.append(
                f"\n\n[错误] 模型工具调用步数达到上限（recursion_limit={_recursion_limit()}），可能陷入循环。"
            )
        else:
            chunks.append(f"\n\n[错误] agent 运行失败：{type(e).__name__}: {e}")
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
        skills_dirs=skills_dirs,
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
