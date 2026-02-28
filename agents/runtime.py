from __future__ import annotations

from collections.abc import Iterable
import base64
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from langchain.agents.middleware.types import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk, ToolMessage, ToolMessageChunk
from langchain_deepseek import ChatDeepSeek
from typing import Any
from typing_extensions import TypedDict

from . import console_write
from skills.skills_support import create_skill_middleware
from agents.tools import action_log_snapshot, actions_since, action_scope, load_mcp_tools_from_config
from memory import create_memory_middleware


_TOKEN_USAGE_MSG: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
_TOKEN_USAGE_CB: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
_ESTIMATED_TOKEN_USAGE: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def reset_token_usage() -> None:
    _TOKEN_USAGE_MSG["input_tokens"] = 0
    _TOKEN_USAGE_MSG["output_tokens"] = 0
    _TOKEN_USAGE_MSG["total_tokens"] = 0
    _TOKEN_USAGE_CB["input_tokens"] = 0
    _TOKEN_USAGE_CB["output_tokens"] = 0
    _TOKEN_USAGE_CB["total_tokens"] = 0


def reset_estimated_token_usage() -> None:
    _ESTIMATED_TOKEN_USAGE["input_tokens"] = 0
    _ESTIMATED_TOKEN_USAGE["output_tokens"] = 0
    _ESTIMATED_TOKEN_USAGE["total_tokens"] = 0


def get_token_usage() -> dict[str, int]:
    msg = dict(_TOKEN_USAGE_MSG)
    cb = dict(_TOKEN_USAGE_CB)
    msg_total = int(msg.get("total_tokens") or 0)
    cb_total = int(cb.get("total_tokens") or 0)
    return cb if cb_total >= msg_total else msg


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
    _TOKEN_USAGE_MSG["input_tokens"] += usage["input_tokens"]
    _TOKEN_USAGE_MSG["output_tokens"] += usage["output_tokens"]
    _TOKEN_USAGE_MSG["total_tokens"] += usage["total_tokens"]


class _TokenUsageCallback(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            usage = _normalize_token_usage(llm_output.get("token_usage"))
            if usage is None:
                usage = _normalize_token_usage(llm_output.get("usage"))
            if usage is not None:
                _TOKEN_USAGE_CB["input_tokens"] += usage["input_tokens"]
                _TOKEN_USAGE_CB["output_tokens"] += usage["output_tokens"]
                _TOKEN_USAGE_CB["total_tokens"] += usage["total_tokens"]
                return

        generations = getattr(response, "generations", None)
        if not isinstance(generations, list):
            return
        for group in generations:
            if not isinstance(group, list):
                continue
            for gen in group:
                msg = getattr(gen, "message", None)
                usage2 = _extract_token_usage_from_message(msg) if msg is not None else None
                if usage2 is None:
                    continue
                _TOKEN_USAGE_CB["input_tokens"] += usage2["input_tokens"]
                _TOKEN_USAGE_CB["output_tokens"] += usage2["output_tokens"]
                _TOKEN_USAGE_CB["total_tokens"] += usage2["total_tokens"]


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
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                    if isinstance(url, str) and url:
                        if url.startswith("data:"):
                            parts.append("\n[image] (inline)\n")
                        else:
                            parts.append(f"\n[image] {url}\n")
                video_url = item.get("video_url")
                if isinstance(video_url, dict):
                    url = video_url.get("url")
                    if isinstance(url, str) and url:
                        if url.startswith("data:"):
                            parts.append("\n[video] (inline)\n")
                        else:
                            parts.append(f"\n[video] {url}\n")
        return "".join(parts)
    return str(content)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_URL_RE = re.compile(r"(https?://[^\s<>\"]+)")
_WIN_ABS_PATH_RE = re.compile(r"([A-Za-z]:\\[^\\n]+)")


def _maybe_file_url_to_path(url: str) -> str | None:
    u = (url or "").strip()
    if not u.lower().startswith("file:"):
        return None
    parsed = urlparse(u)
    p = unquote((parsed.path or "")).lstrip("/")
    if not p:
        return None
    p = p.replace("/", "\\")
    return p


def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
    except Exception:
        return False
    return (p.scheme or "").lower() in {"http", "https"}


def _normalize_media_ref(raw: str) -> str:
    s = (raw or "").strip().strip("<>").strip().strip("'\"")
    return s


def _path_suffix_lower(s: str) -> str:
    try:
        return Path(s).suffix.lower()
    except Exception:
        return ""


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = (it or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _extract_media_refs_from_text(user_text: str) -> tuple[list[str], list[str]]:
    text = user_text or ""
    images: list[str] = []
    videos: list[str] = []

    for m in _MD_IMAGE_RE.finditer(text):
        ref = _normalize_media_ref(m.group(1))
        if ref:
            images.append(ref)

    for m in _URL_RE.finditer(text):
        ref = _normalize_media_ref(m.group(1))
        suf = _path_suffix_lower(ref)
        if suf in _IMAGE_EXTS:
            images.append(ref)
        elif suf in _VIDEO_EXTS:
            videos.append(ref)

    for m in _WIN_ABS_PATH_RE.finditer(text):
        ref = _normalize_media_ref(m.group(1))
        suf = _path_suffix_lower(ref)
        if suf in _IMAGE_EXTS:
            images.append(ref)
        elif suf in _VIDEO_EXTS:
            videos.append(ref)

    tokens = re.split(r"[\s,;]+", text)
    for tok in tokens:
        ref = _normalize_media_ref(tok)
        if not ref:
            continue
        if _is_http_url(ref) or ref.lower().startswith("data:") or ref.lower().startswith("file:"):
            continue
        suf = _path_suffix_lower(ref)
        if suf in _IMAGE_EXTS:
            images.append(ref)
        elif suf in _VIDEO_EXTS:
            videos.append(ref)

    return _dedupe_preserve_order(images), _dedupe_preserve_order(videos)


def _data_url_for_local_file(path_str: str, *, max_bytes: int) -> str | None:
    try:
        p = Path(path_str).expanduser()
    except Exception:
        return None
    if not p.is_absolute():
        try:
            p = (Path.cwd() / p)
        except Exception:
            return None
    try:
        p = p.resolve()
    except OSError:
        return None
    if not p.exists() or not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if max_bytes > 0 and len(data) > max_bytes:
        return None
    mime, _ = mimetypes.guess_type(p.as_posix())
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_user_message(
    user_text: str,
    *,
    images: list[str] | None = None,
    videos: list[str] | None = None,
    max_inline_bytes: int | None = None,
) -> dict[str, Any]:
    text = (user_text or "").strip()
    imgs = list(images) if isinstance(images, list) else None
    vids = list(videos) if isinstance(videos, list) else None
    if imgs is None or vids is None:
        auto_imgs, auto_vids = _extract_media_refs_from_text(text)
        if imgs is None:
            imgs = auto_imgs
        if vids is None:
            vids = auto_vids

    imgs = _dedupe_preserve_order(imgs or [])
    vids = _dedupe_preserve_order(vids or [])

    if not imgs and not vids:
        return {"role": "user", "content": text}

    if isinstance(max_inline_bytes, int):
        max_bytes = max_inline_bytes
    else:
        raw_limit = (os.environ.get("AGENT_MEDIA_MAX_INLINE_BYTES") or "").strip()
        if raw_limit:
            try:
                max_bytes = int(raw_limit)
            except ValueError:
                max_bytes = 6_000_000
        else:
            max_bytes = 6_000_000
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})

    for ref in imgs:
        ref2 = _normalize_media_ref(ref)
        if not ref2:
            continue
        file_path = _maybe_file_url_to_path(ref2)
        if file_path:
            ref2 = file_path
        if ref2.lower().startswith("data:") or _is_http_url(ref2):
            parts.append({"type": "image_url", "image_url": {"url": ref2}})
            continue
        data_url = _data_url_for_local_file(ref2, max_bytes=max_bytes)
        if data_url:
            parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            parts.append({"type": "text", "text": f"[image] {ref2}"})

    for ref in vids:
        ref2 = _normalize_media_ref(ref)
        if not ref2:
            continue
        parts.append({"type": "text", "text": f"[video] {ref2}"})

    return {"role": "user", "content": parts}


def messages_to_prompt_text_for_estimate(messages: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, str):
            lines.append(content)
            continue
        if isinstance(content, list):
            parts: list[str] = []
            for it in content:
                if isinstance(it, str):
                    parts.append(it)
                    continue
                if not isinstance(it, dict):
                    continue
                t = it.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
                    continue
                image_url = it.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str) and image_url.get("url"):
                    parts.append("[image]")
                    continue
                video_url = it.get("video_url")
                if isinstance(video_url, dict) and isinstance(video_url.get("url"), str) and video_url.get("url"):
                    parts.append("[video]")
                    continue
            lines.append("".join(parts))
            continue
        lines.append(str(content or ""))
    return "\n".join(lines)


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



def stream_nested_agent_reply(agent, messages: list[dict[str, Any]], *, label: str, thread_id: str | None = None) -> tuple[str, str]:
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
        prompt_text = messages_to_prompt_text_for_estimate(messages)
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
    messages: list[dict[str, Any]],
    *,
    checkpoint_ns: str = "observer",
    thread_id: str | None = None,
    user_id: str | None = None,
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
                    config=_agent_stream_config(checkpoint_ns=checkpoint_ns, thread_id=active_thread_id, user_id=user_id),
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


def _ensure_user_id() -> str:
    uid = (os.environ.get("AGENT_USER_ID") or "").strip()
    if uid:
        return uid
    os.environ["AGENT_USER_ID"] = "default"
    return "default"


def _agent_stream_config(*, checkpoint_ns: str, thread_id: str | None = None, user_id: str | None = None) -> dict[str, object]:
    tid = (thread_id or "").strip() or _ensure_thread_id()
    uid = (user_id or "").strip() if isinstance(user_id, str) else ""
    if not uid:
        uid = _ensure_user_id()
    return {
        "recursion_limit": _recursion_limit(),
        "configurable": {"thread_id": tid, "user_id": uid, "checkpoint_ns": (checkpoint_ns or "").strip()},
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
    cb = _TokenUsageCallback()
    if normalized_model.startswith("deepseek:"):
        deepseek_model_name = normalized_model.split(":", 1)[1]
        try:
            return ChatDeepSeekThinkingTools(model=deepseek_model_name, streaming=True, callbacks=[cb])
        except TypeError:
            return ChatDeepSeekThinkingTools(model=deepseek_model_name, streaming=True)
    try:
        if normalized_model.startswith("openai:"):
            try:
                return init_chat_model(
                    model=normalized_model,
                    streaming=True,
                    callbacks=[cb],
                    model_kwargs={"stream_options": {"include_usage": True}},
                )
            except TypeError:
                return init_chat_model(model=normalized_model, streaming=True, callbacks=[cb])
        try:
            return init_chat_model(model=normalized_model, streaming=True, callbacks=[cb])
        except TypeError:
            return init_chat_model(model=normalized_model, streaming=True)
    except ValueError as e:
        if "Unable to infer model provider" not in str(e):
            raise
        try:
            return init_chat_model(
                model=normalized_model,
                model_provider="openai",
                streaming=True,
                callbacks=[cb],
                model_kwargs={"stream_options": {"include_usage": True}},
            )
        except TypeError:
            return init_chat_model(model=normalized_model, model_provider="openai", streaming=True, callbacks=[cb])


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

    checkpointer = InMemorySaver()

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
        checkpointer=checkpointer,
    )

    return agent, skill_catalog_text, skill_count
