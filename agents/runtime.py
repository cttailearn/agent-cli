from __future__ import annotations

from collections.abc import Iterable
import os
import sqlite3
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


def _maybe_collapse_stutter(text: str) -> str:
    s = text or ""
    if len(s) < 12:
        return s

    def _allow_ws_join_for_single_char(ch: str) -> bool:
        if not ch:
            return False
        if "\u4e00" <= ch <= "\u9fff":
            return True
        if ch.isalnum():
            return False
        return True

    repeats = 0
    repeats_ge2 = 0
    repeats_ge3 = 0
    distinct: set[str] = set()
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in s)
    max_seg_len = min(12, max(1, len(s) // 4))
    for seg_len in range(1, max_seg_len + 1):
        i = 0
        while i + 2 * seg_len <= len(s):
            a = s[i : i + seg_len]
            b = s[i + seg_len : i + 2 * seg_len]
            if a and a == b:
                repeats += 1
                if seg_len >= 2:
                    repeats_ge2 += 1
                if seg_len >= 3:
                    repeats_ge3 += 1
                if len(distinct) < 12:
                    distinct.add(a)
                i += seg_len * 2
                continue
            if i + seg_len < len(s) and s[i + seg_len].isspace():
                if seg_len == 1 and not _allow_ws_join_for_single_char(a):
                    i += 1
                    continue
                j = i + seg_len
                while j < len(s) and s[j].isspace():
                    j += 1
                if j + seg_len <= len(s):
                    b2 = s[j : j + seg_len]
                    if a and a == b2:
                        repeats += 1
                        if seg_len >= 2:
                            repeats_ge2 += 1
                        if seg_len >= 3:
                            repeats_ge3 += 1
                        if len(distinct) < 12:
                            distinct.add(a)
                        i = j + seg_len
                        continue
            i += 1

    if repeats < 2:
        return s

    if has_cjk:
        if repeats_ge2 < 1 and repeats < 4:
            return s
    else:
        if repeats_ge3 < 2 and (repeats < 6 or len(distinct) < 3):
            return s
        if repeats_ge3 < 2 and len(distinct) < 2:
            return s

    if len(distinct) < 1:
        return s

    out: list[str] = []
    i = 0
    longest = min(24, max(8, len(s) // 3))
    while i < len(s):
        collapsed = False
        for seg_len in range(longest, 0, -1):
            if i + 2 * seg_len > len(s):
                continue
            a = s[i : i + seg_len]
            b = s[i + seg_len : i + 2 * seg_len]
            if a and a == b:
                out.append(a)
                j = i + seg_len * 2
                while j + seg_len <= len(s) and s[j : j + seg_len] == a:
                    j += seg_len
                i = j
                collapsed = True
                break
            if i + seg_len < len(s) and s[i + seg_len].isspace():
                if seg_len == 1 and not _allow_ws_join_for_single_char(a):
                    continue
                j = i + seg_len
                while j < len(s) and s[j].isspace():
                    j += 1
                if j + seg_len <= len(s) and a and s[j : j + seg_len] == a:
                    out.append(a)
                    j2 = j + seg_len
                    while True:
                        if j2 + seg_len <= len(s) and s[j2 : j2 + seg_len] == a:
                            j2 += seg_len
                            continue
                        k = j2
                        while k < len(s) and s[k].isspace():
                            k += 1
                        if k != j2 and k + seg_len <= len(s) and s[k : k + seg_len] == a:
                            j2 = k + seg_len
                            continue
                        break
                    i = j2
                    collapsed = True
                    break
        if not collapsed:
            out.append(s[i])
            i += 1
    return "".join(out)


def _extract_text(msg: BaseMessage | BaseMessageChunk) -> str:
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return _maybe_collapse_stutter(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if not parts:
            return ""
        if len(parts) == 1:
            return _maybe_collapse_stutter(parts[0])

        uniq: list[str] = []
        for s in parts:
            if s not in uniq:
                uniq.append(s)
        parts = uniq

        parts_by_len = sorted(parts, key=len, reverse=True)
        for candidate in parts_by_len:
            if all((p in candidate) for p in parts):
                return _maybe_collapse_stutter(candidate)

        kept_rev: list[str] = []
        for s in reversed(parts):
            if any(s and s in k for k in kept_rev):
                continue
            kept_rev.append(s)
        kept = list(reversed(kept_rev))

        assembled = ""
        for s in kept:
            if not assembled:
                assembled = s
                continue
            if s.startswith(assembled):
                assembled = s
                continue
            if assembled.endswith(s):
                continue
            max_k = min(len(assembled), len(s))
            k_found = 0
            for k in range(max_k, 0, -1):
                if assembled.endswith(s[:k]):
                    k_found = k
                    break
            if k_found:
                assembled += s[k_found:]
            else:
                assembled += s
        return _maybe_collapse_stutter(assembled)
    return _maybe_collapse_stutter(str(content))


def self_test_extract_text_dedup() -> bool:
    class _Msg:
        def __init__(self, content: object) -> None:
            self.content = content

    cases: list[tuple[object, str]] = [
        (["按照", "我将按照"], "我将按照"),
        (["我将", "按照", "我将按照"], "我将按照"),
        (["Markdown", "Markdown"], "Markdown"),
        (["foo", "bar"], "foobar"),
        ([{"text": "按照"}, {"text": "我将按照"}], "我将按照"),
        (["主人", "主人", "，", "，", "我来", "我来"], "主人，我来"),
        (["主人", "主人，", "主人，我来"], "主人，我来"),
        (["主人主人，，我来我来为您为您搜索前端搜索前端设计相关的设计相关的技能。"], "主人，我来为您搜索前端设计相关的技能。"),
        ("主人主人，，我来我来检查浏览器检查浏览器自动化技能自动化技能的 setup.json setup.json 文件 文件状态。", "主人，我来检查浏览器自动化技能的 setup.json 文件状态。"),
        ("npx npx skills find skills find frontend design", "npx skills find frontend design"),
        ("bookkeeper", "bookkeeper"),
    ]
    for content, expected in cases:
        if _extract_text(_Msg(content)) != expected:
            return False
    return True


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


def _stream_text_delta(new_text: str, *, last_seen: str | None, assembled: str) -> tuple[str, str | None]:
    def _strip_overlap(prev: str, delta: str) -> str:
        if not prev or not delta:
            return delta
        max_k = min(len(prev), len(delta))
        for k in range(max_k, 0, -1):
            if prev.endswith(delta[:k]):
                return delta[k:]
        return delta

    t = new_text or ""
    if not t:
        return "", last_seen
    if last_seen is not None and t == last_seen:
        return "", last_seen
    if assembled and t.startswith(assembled):
        delta = t[len(assembled) :]
        return _strip_overlap(assembled, delta), t
    if last_seen and t.startswith(last_seen):
        delta = t[len(last_seen) :]
        return _strip_overlap(assembled, delta), t
    if last_seen and last_seen.startswith(t):
        return "", last_seen
    if assembled and assembled.endswith(t):
        return "", t
    if assembled:
        max_k = min(len(assembled), len(t))
        for k in range(max_k, 0, -1):
            if assembled.endswith(t[:k]):
                return _strip_overlap(assembled, t[k:]), t
    return _strip_overlap(assembled, t), t



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
            last_assistant_seen: str | None = None
            last_tool_seen: str | None = None
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
                        delta, last_assistant_seen = _stream_text_delta(
                            text, last_seen=last_assistant_seen, assembled=assistant_assembled
                        )
                        if delta:
                            assistant_assembled += delta
                            assistant_buf += delta
                            _flush_assistant(force=False)
                            chunks.append(delta)
                        _drain_actions()
                    elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                        text = _extract_text(msg)
                        delta, last_tool_seen = _stream_text_delta(
                            text, last_seen=last_tool_seen, assembled=tool_assembled
                        )
                        if delta:
                            tool_assembled += delta
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
            last_assistant_seen: str | None = None
            last_tool_seen: str | None = None
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
                        delta, last_assistant_seen = _stream_text_delta(
                            text, last_seen=last_assistant_seen, assembled=assistant_assembled
                        )
                        if delta:
                            assistant_assembled += delta
                    elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                        text = _extract_text(msg)
                        delta, last_tool_seen = _stream_text_delta(
                            text, last_seen=last_tool_seen, assembled=tool_assembled
                        )
                        if delta:
                            tool_assembled += delta
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
    agents_dir = (project_root / ".agents").resolve()
    agents_dir.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.store.sqlite import SqliteStore

        store_conn = sqlite3.connect(
            str((agents_dir / "store.sqlite").resolve()),
            check_same_thread=False,
            timeout=30.0,
        )
        try:
            store_conn.execute("PRAGMA journal_mode=WAL;")
            store_conn.execute("PRAGMA synchronous=NORMAL;")
            store_conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        store = SqliteStore(store_conn)
        store.setup()
    except Exception:
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint_conn = sqlite3.connect(
            str((agents_dir / "checkpoints.sqlite").resolve()),
            check_same_thread=False,
            timeout=30.0,
        )
        try:
            checkpoint_conn.execute("PRAGMA journal_mode=WAL;")
            checkpoint_conn.execute("PRAGMA synchronous=NORMAL;")
            checkpoint_conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        checkpointer = SqliteSaver(checkpoint_conn)
    except Exception:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()

    store = _LockedProxy(store)

    from .executor_agent import build_executor_agent, executor_tools
    from .observer_agent import build_observer_agent
    from .supervisor_agent import build_supervisor_agent

    executor_agent = build_executor_agent(
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

    executor_tools_list = executor_tools(mcp_tools=mcp_tools, skill_middleware=skill_middleware)

    supervisor_agent, supervisor_tools = build_supervisor_agent(
        model_name=model_name,
        skill_middleware=skill_middleware,
        memory_middleware=memory_middleware,
        skills_dirs=skills_dirs,
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        store=store,
        checkpointer=checkpointer,
    )

    observer_agent = build_observer_agent(
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
        skill_middleware=skill_middleware,
        memory_middleware=memory_middleware,
        executor_agent=executor_agent,
        executor_tools=executor_tools_list,
        supervisor_agent=supervisor_agent,
        supervisor_tools=supervisor_tools,
        store=store,
        checkpointer=checkpointer,
    )

    return observer_agent, skill_catalog_text, skill_count
