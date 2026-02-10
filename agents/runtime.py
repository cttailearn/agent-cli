from __future__ import annotations

from collections.abc import Iterable
import os
import threading
import time
import uuid
from pathlib import Path

from langchain.agents.middleware.types import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk, ToolMessage, ToolMessageChunk
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict

from . import console_write
from skills.skills_support import create_skill_middleware
from agents.tools import action_log_snapshot, actions_since, action_scope, load_mcp_tools_from_config
from memory import create_memory_middleware


_TOKEN_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
_ESTIMATED_TOKEN_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def reset_token_usage() -> None:
    _TOKEN_USAGE["input_tokens"] = 0
    _TOKEN_USAGE["output_tokens"] = 0
    _TOKEN_USAGE["total_tokens"] = 0


def reset_estimated_token_usage() -> None:
    _ESTIMATED_TOKEN_USAGE["input_tokens"] = 0
    _ESTIMATED_TOKEN_USAGE["output_tokens"] = 0
    _ESTIMATED_TOKEN_USAGE["total_tokens"] = 0


def get_token_usage() -> dict[str, int]:
    return dict(_TOKEN_USAGE)


def get_estimated_token_usage() -> dict[str, int]:
    return dict(_ESTIMATED_TOKEN_USAGE)


def add_estimated_token_usage(usage: dict[str, int]) -> None:
    if not isinstance(usage, dict):
        return
    _ESTIMATED_TOKEN_USAGE["input_tokens"] += int(usage.get("input_tokens") or 0)
    _ESTIMATED_TOKEN_USAGE["output_tokens"] += int(usage.get("output_tokens") or 0)
    _ESTIMATED_TOKEN_USAGE["total_tokens"] += int(usage.get("total_tokens") or 0)


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


class _LockedProxy:
    def __init__(self, obj: object) -> None:
        self._obj = obj
        self._lock = threading.RLock()

    def __getattr__(self, name: str):
        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return _wrapped


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


def _format_actions_for_console(actions: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for a in actions:
        kind = a.get("kind")
        ok = a.get("ok")
        if kind == "write_file":
            lines.append(f"- write_file ok={ok} path={a.get('path')} size={a.get('size')}")
        elif kind == "write_project_file":
            lines.append(f"- write_project_file ok={ok} path={a.get('path')} size={a.get('size')}")
        elif kind == "read_file":
            lines.append(f"- read_file ok={ok} path={a.get('path')} truncated={a.get('truncated')}")
        elif kind == "list_dir":
            lines.append(f"- list_dir ok={ok} path={a.get('path')} recursive={a.get('recursive')} entries={a.get('entries')}")
        elif kind == "delete_path":
            lines.append(f"- delete_path ok={ok} path={a.get('path')} recursive={a.get('recursive')}")
        elif kind == "run_cli":
            lines.append(
                f"- run_cli ok={ok} exit_code={a.get('exit_code')} cwd={a.get('cwd')} cmd={a.get('command')}"
            )
        elif kind == "skills_subprocess":
            lines.append(
                f"- skills_subprocess ok={ok} exit_code={a.get('exit_code')} cwd={a.get('cwd')} cmd={a.get('command')}"
            )
        else:
            lines.append(f"- {a}")
    return "\n".join(lines)


def _summarize_actions_for_console(actions: list[dict[str, object]]) -> str:
    if not actions:
        return ""
    counts: dict[str, int] = {}
    for a in actions:
        kind = a.get("kind")
        if not isinstance(kind, str) or not kind:
            kind = "unknown"
        counts[kind] = counts.get(kind, 0) + 1
    parts = [f"{k} x{v}" for k, v in sorted(counts.items())]
    return "，".join(parts)


def _diff_token_usage(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {
        "input_tokens": max(0, int(after.get("input_tokens") or 0) - int(before.get("input_tokens") or 0)),
        "output_tokens": max(0, int(after.get("output_tokens") or 0) - int(before.get("output_tokens") or 0)),
        "total_tokens": max(0, int(after.get("total_tokens") or 0) - int(before.get("total_tokens") or 0)),
    }


def _is_tool_calls_sequence_error(e: Exception) -> bool:
    msg = str(e or "")
    if not msg:
        return False
    if "tool_calls" not in msg:
        return False
    if "must be followed by tool messages" in msg:
        return True
    if "insufficient tool messages" in msg:
        return True
    if "tool_call_id" in msg and "followed by tool messages" in msg:
        return True
    return False


def _stream_text_delta(new_text: str, *, assembled: str) -> tuple[str, str]:
    t = new_text or ""
    if not t:
        return "", assembled

    base = assembled or ""
    if not base:
        return t, t

    if t.startswith(base):
        delta = t[len(base) :]
        return delta, t

    if len(t) <= 4:
        return t, base + t

    max_k = min(len(base), len(t))
    for k in range(max_k, 0, -1):
        if base.endswith(t[:k]):
            delta = t[k:]
            if not delta:
                return "", base
            return delta, base + delta

    return t, base + t



def stream_nested_agent_reply(agent, messages: list[dict[str, str]], *, label: str, thread_id: str | None = None) -> tuple[str, str]:
    snapshot = action_log_snapshot()
    last_action_index = 0
    usage_before = get_token_usage()

    console_write(f"\n[{label}] 开始\n", flush=True)

    def _write_out(s: str) -> None:
        if not s:
            return
        console_write(s, flush=True)

    def _drain_actions() -> None:
        nonlocal last_action_index
        actions = actions_since(snapshot, scope=label)
        if len(actions) <= last_action_index:
            return
        delta = actions[last_action_index:]
        last_action_index = len(actions)
        if not delta:
            return
        summary = _summarize_actions_for_console(delta)
        if summary:
            _write_out(f"\n[{label}] {summary}\n")
        _write_out(f"{_format_actions_for_console(delta)}\n")

    active_thread_id = thread_id
    reply = ""
    tool_output = ""
    with action_scope(label):
        for attempt in range(2):
            chunks: list[str] = []
            tool_chunks: list[str] = []
            assistant_buf = ""
            last_flush_t = time.monotonic()
            assistant_assembled = ""
            tool_assembled = ""

            def _flush_assistant(force: bool = False) -> None:
                nonlocal assistant_buf, last_flush_t
                if not assistant_buf:
                    return
                now = time.monotonic()
                if force or "\n" in assistant_buf or len(assistant_buf) >= 256 or (now - last_flush_t) >= 0.03:
                    _write_out(assistant_buf)
                    assistant_buf = ""
                    last_flush_t = now

            try:
                for event in agent.stream(
                    {"messages": messages},
                    stream_mode="messages",
                    config=_agent_stream_config(checkpoint_ns=label, thread_id=active_thread_id),
                ):
                    if isinstance(event, tuple) and event:
                        msg = event[0]
                    else:
                        msg = event
                    if isinstance(msg, AIMessageChunk):
                        capture_token_usage_from_message(msg)
                        text = _extract_text(msg)
                        delta, assistant_assembled = _stream_text_delta(text, assembled=assistant_assembled)
                        if delta:
                            assistant_buf += delta
                            _flush_assistant(force=False)
                            chunks.append(delta)
                        _drain_actions()
                    elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                        text = _extract_text(msg)
                        delta, tool_assembled = _stream_text_delta(text, assembled=tool_assembled)
                        if delta:
                            tool_chunks.append(delta)
                            summarized = _summarize_tool_output_for_terminal(delta)
                            if summarized:
                                _write_out(f"\n[{label}] {summarized}\n")
                        _drain_actions()
                _flush_assistant(force=True)
                reply = assistant_assembled
                tool_output = tool_assembled
                break
            except Exception as e:
                if attempt == 0 and _is_tool_calls_sequence_error(e):
                    _flush_assistant(force=True)
                    _write_out(f"\n[{label}] 检测到 tool_calls 断链，重置线程并重试\n")
                    _drain_actions()
                    active_thread_id = uuid.uuid4().hex[:12]
                    continue
                _flush_assistant(force=True)
                _write_out(f"\n[{label}] 发生错误：{type(e).__name__}: {e}\n")
                _drain_actions()
                break

    _drain_actions()

    usage_after = get_token_usage()
    delta = _diff_token_usage(usage_after, usage_before)
    if delta.get("total_tokens", 0) <= 0:
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        est = estimate_token_usage(prompt_text, reply)
        add_estimated_token_usage(est)
        _write_out(
            f"\n[{label}] tokens(估算): input={est.get('input_tokens', 0)} output={est.get('output_tokens', 0)} total={est.get('total_tokens', 0)}\n"
        )
    else:
        _write_out(
            f"\n[{label}] tokens: input={delta.get('input_tokens', 0)} output={delta.get('output_tokens', 0)} total={delta.get('total_tokens', 0)}\n"
        )

    console_write(f"[{label}] 结束\n\n", flush=True)
    return reply, tool_output


def _recursion_limit() -> int:
    raw = (os.environ.get("AGENT_RECURSION_LIMIT") or "").strip()
    if not raw:
        return 64
    try:
        v = int(raw)
    except ValueError:
        return 64
    return max(10, min(500, v))


def _run_agent_to_text(
    agent,
    messages: list[dict[str, str]],
    *,
    checkpoint_ns: str = "observer",
    thread_id: str | None = None,
) -> tuple[str, str]:
    try:
        from langgraph.errors import GraphRecursionError
    except Exception:
        GraphRecursionError = None

    active_thread_id = thread_id
    reply = ""
    tool_output = ""
    with action_scope(checkpoint_ns):
        for attempt in range(2):
            assistant_assembled = ""
            tool_assembled = ""
            try:
                for event in agent.stream(
                    {"messages": messages},
                    stream_mode="messages",
                    config=_agent_stream_config(checkpoint_ns=checkpoint_ns, thread_id=active_thread_id),
                ):
                    if isinstance(event, tuple) and event:
                        msg = event[0]
                    else:
                        msg = event
                    if isinstance(msg, AIMessageChunk):
                        capture_token_usage_from_message(msg)
                        text = _extract_text(msg)
                        delta, assistant_assembled = _stream_text_delta(text, assembled=assistant_assembled)
                    elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                        text = _extract_text(msg)
                        delta, tool_assembled = _stream_text_delta(text, assembled=tool_assembled)
                reply = assistant_assembled
                tool_output = tool_assembled
                break
            except Exception as e:
                if attempt == 0 and _is_tool_calls_sequence_error(e):
                    active_thread_id = uuid.uuid4().hex[:12]
                    continue
                if GraphRecursionError is not None and isinstance(e, GraphRecursionError):
                    reply = f"\n\n[错误] 模型工具调用步数达到上限（recursion_limit={_recursion_limit()}），可能陷入循环。"
                else:
                    reply = f"\n\n[错误] agent 运行失败：{type(e).__name__}: {e}"
                break
    return reply, tool_output


def _ensure_thread_id() -> str:
    tid = (os.environ.get("AGENT_THREAD_ID") or "").strip()
    if tid:
        return tid
    tid = uuid.uuid4().hex[:12]
    os.environ["AGENT_THREAD_ID"] = tid
    return tid


def _agent_stream_config(*, checkpoint_ns: str, thread_id: str | None = None) -> dict[str, object]:
    tid = (thread_id or "").strip() or _ensure_thread_id()
    return {
        "recursion_limit": _recursion_limit(),
        "configurable": {"thread_id": tid, "checkpoint_ns": (checkpoint_ns or "").strip()},
    }


class UnifiedAgentState(AgentState[object], total=False):
    shared: dict[str, object]


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
    try:
        return init_chat_model(model=normalized_model, streaming=True)
    except ValueError as e:
        if "Unable to infer model provider" not in str(e):
            raise
        return init_chat_model(model=normalized_model, model_provider="openai", streaming=True)


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
    memory_middleware = create_memory_middleware(project_root)
    mcp_tools = load_mcp_tools_from_config()
    (project_root / ".agents").mkdir(parents=True, exist_ok=True)

    from langgraph.checkpoint.memory import InMemorySaver

    from memory.paths import langgraph_store_path
    from memory.storage import PersistentInMemoryStore

    langgraph_path = langgraph_store_path(project_root)
    langgraph_path.parent.mkdir(parents=True, exist_ok=True)
    store = PersistentInMemoryStore(path=langgraph_path)
    checkpointer = InMemorySaver()

    store = _LockedProxy(store)

    from .agent import build_single_agent

    agent = build_single_agent(
        skills_dirs=skills_dirs,
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
        skill_middleware=skill_middleware,
        memory_middleware=memory_middleware,
        mcp_tools=mcp_tools,
        store=store,
        checkpointer=checkpointer,
    )

    return agent, skill_catalog_text, skill_count
