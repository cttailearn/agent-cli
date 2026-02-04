from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)

from agents import console_write
from agents.runtime import (
    capture_token_usage_from_message,
    estimate_token_usage,
    get_estimated_token_usage,
    get_token_usage,
    reset_estimated_token_usage,
    reset_token_usage,
    _stream_text_delta,
)
from agents.tools import action_log_snapshot, actions_since, action_scope, delete_path, list_dir, read_file, run_cli


_COMPLETION_CLAIM_RE = re.compile(
    r"("
    r"(已|已经|我已|我们已).{0,30}(创建|生成|写入|安装|运行|执行|完成)"
    r"|创建完成|生成完成|已完成|全部完成|完美"
    r")"
)


@dataclass
class CliState:
    show_tool_output: bool = True
    show_action_summary: bool = True
    skill_catalog_text: str = ""
    skill_count: int = 0
    last_actions: list[dict[str, object]] = field(default_factory=list)
    last_tool_output: str = ""
    last_tool_output_truncated: bool = False
    last_token_usage: dict[str, int] | None = None
    last_token_usage_is_estimate: bool = False


def _format_actions(actions: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for a in actions:
        kind = a.get("kind")
        ok = a.get("ok")
        if kind == "write_file":
            path = a.get("path")
            size = a.get("size")
            lines.append(f"- write_file ok={ok} path={path} size={size}")
        elif kind == "write_project_file":
            path = a.get("path")
            size = a.get("size")
            lines.append(f"- write_project_file ok={ok} path={path} size={size}")
        elif kind == "read_file":
            path = a.get("path")
            truncated = a.get("truncated")
            lines.append(f"- read_file ok={ok} path={path} truncated={truncated}")
        elif kind == "list_dir":
            path = a.get("path")
            recursive = a.get("recursive")
            entries = a.get("entries")
            lines.append(f"- list_dir ok={ok} path={path} recursive={recursive} entries={entries}")
        elif kind == "delete_path":
            path = a.get("path")
            recursive = a.get("recursive")
            lines.append(f"- delete_path ok={ok} path={path} recursive={recursive}")
        elif kind == "run_cli":
            command = a.get("command")
            cwd = a.get("cwd")
            exit_code = a.get("exit_code")
            lines.append(f"- run_cli ok={ok} exit_code={exit_code} cwd={cwd} cmd={command}")
        else:
            lines.append(f"- {a}")
    return "\n".join(lines)


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
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        kept_rev: list[str] = []
        for s in reversed(parts):
            if any(k.startswith(s) for k in kept_rev):
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
        return assembled
    return str(content)


def _summarize_actions(actions: list[dict[str, object]]) -> str:
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


def _print_block(title: str, text: str) -> None:
    if title:
        print(title)
    if text:
        print(text)


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


def _print_last(state: CliState, kind: str) -> None:
    kind = (kind or "").strip().lower()
    if kind in {"", "all"}:
        _print_last(state, "actions")
        _print_last(state, "tools")
        return

    if kind in {"actions", "action"}:
        if not state.last_actions:
            print("暂无最近执行记录。")
            return
        summary = _summarize_actions(state.last_actions)
        if summary:
            print(summary)
        _print_block("", _format_actions(state.last_actions))
        return

    if kind in {"tools", "tool"}:
        if not state.last_tool_output:
            print("暂无最近工具输出。")
            return
        summarized = _summarize_tool_output_for_terminal(state.last_tool_output)
        if not summarized:
            print("（无可展示的摘要）")
            return
        _print_block("", summarized)
        if state.last_tool_output_truncated:
            print("（已截断存储）")
        return

    if kind == "skills":
        if not state.skill_catalog_text:
            print("暂无技能目录。")
            return
        _print_block(f"技能目录（共 {state.skill_count} 个）", state.skill_catalog_text)
        return

    print("用法：/last [actions|tools|skills|all]")


def stream_assistant_reply(agent, messages: list[dict[str, str]], state: CliState) -> str:
    label = "observer"
    stream_delegation = (os.environ.get("AGENT_STREAM_DELEGATION") or "").strip().lower() in {"1", "true", "yes", "on"}
    drain_scope = label if stream_delegation else None
    snapshot = action_log_snapshot()
    reset_token_usage()
    reset_estimated_token_usage()
    chunks: list[str] = []
    tool_chunks: list[str] = []
    has_tool_output = False
    channel: str = "assistant"
    last_output_newline = True
    assistant_buf = ""
    last_flush_t = time.monotonic()
    tool_stream_buf = ""
    last_action_index = 0
    last_assistant_seen: str | None = None
    last_tool_seen: str | None = None
    assistant_assembled = ""
    tool_assembled = ""

    def _write_out(s: str, flush: bool = False) -> None:
        nonlocal last_output_newline
        if not s:
            return
        console_write(s, flush=flush)
        last_output_newline = s.endswith("\n")

    def _flush_assistant(force: bool = False) -> None:
        nonlocal assistant_buf, last_flush_t
        if not assistant_buf:
            return
        now = time.monotonic()
        if force or "\n" in assistant_buf or len(assistant_buf) >= 256 or (now - last_flush_t) >= 0.03:
            _write_out(assistant_buf, flush=True)
            assistant_buf = ""
            last_flush_t = now

    def _switch_to_tools() -> None:
        nonlocal channel
        if channel == "tools":
            return
        _flush_assistant(force=True)
        if not last_output_newline:
            _write_out("\n", flush=True)
        channel = "tools"

    def _switch_to_assistant() -> None:
        nonlocal channel
        if channel == "assistant":
            return
        if not last_output_newline:
            _write_out("\n", flush=True)
        channel = "assistant"

    def _drain_actions() -> None:
        nonlocal last_action_index
        actions = actions_since(snapshot, scope=drain_scope) if drain_scope else actions_since(snapshot)
        if len(actions) <= last_action_index:
            return
        delta = actions[last_action_index:]
        last_action_index = len(actions)
        if not delta:
            return
        _switch_to_tools()
        if state.show_action_summary:
            summary = _summarize_actions(delta)
            if summary:
                _write_out(f"{summary}\n", flush=True)
        _write_out(f"{_format_actions(delta)}\n", flush=True)

    def _recursion_limit() -> int:
        raw = (os.environ.get("AGENT_RECURSION_LIMIT") or "").strip()
        if not raw:
            return 64
        try:
            v = int(raw)
        except ValueError:
            return 64
        return max(10, min(500, v))

    try:
        from langgraph.errors import GraphRecursionError
    except Exception:
        GraphRecursionError = None

    try:
        with action_scope(label):
            for event in agent.stream(
                {"messages": messages},
                stream_mode="messages",
                config={
                    "recursion_limit": _recursion_limit(),
                    "configurable": {
                        "thread_id": (os.environ.get("AGENT_THREAD_ID") or "").strip() or "default",
                        "checkpoint_ns": "observer",
                    },
                },
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
                        _switch_to_assistant()
                        assistant_buf += delta
                        _flush_assistant(force=False)
                        chunks.append(delta)
                    _drain_actions()
                elif isinstance(msg, (ToolMessage, ToolMessageChunk)):
                    text = _extract_text(msg)
                    delta, last_tool_seen = _stream_text_delta(text, last_seen=last_tool_seen, assembled=tool_assembled)
                    if delta:
                        tool_assembled += delta
                        tool_chunks.append(delta)
                        has_tool_output = True
                        if state.show_tool_output:
                            _switch_to_tools()
                            tool_stream_buf += delta
                            while "\n" in tool_stream_buf:
                                line, tool_stream_buf = tool_stream_buf.split("\n", 1)
                                s = line.strip()
                                if (
                                    s.startswith("path: ")
                                    or s.startswith("Wrote file: ")
                                    or s.startswith("Edited: ")
                                    or s.startswith("Deleted: ")
                                ):
                                    _write_out(f"{s}\n", flush=True)
                    _drain_actions()
    except Exception as e:
        _flush_assistant(force=True)
        if not last_output_newline:
            _write_out("\n", flush=True)
        if GraphRecursionError is not None and isinstance(e, GraphRecursionError):
            msg = f"发生错误：模型工具调用步数达到上限（recursion_limit={_recursion_limit()}），可能陷入循环。\n"
        else:
            msg = f"发生错误：{type(e).__name__}: {e}\n"
        _write_out(msg, flush=True)
        chunks.append(msg)
        _drain_actions()
    _flush_assistant(force=True)
    if not last_output_newline:
        _write_out("\n", flush=True)
    reply = "".join(chunks)
    state.last_token_usage = get_token_usage()
    state.last_token_usage_is_estimate = False
    if (state.last_token_usage.get("total_tokens", 0) if state.last_token_usage else 0) <= 0:
        prompt_text = "\n".join(str(m.get("content") or "") for m in messages)
        state.last_token_usage = estimate_token_usage(prompt_text, reply)
        state.last_token_usage_is_estimate = True

    extra = get_estimated_token_usage()
    if extra.get("total_tokens", 0) > 0 and state.last_token_usage is not None:
        state.last_token_usage = {
            "input_tokens": int(state.last_token_usage.get("input_tokens") or 0) + int(extra.get("input_tokens") or 0),
            "output_tokens": int(state.last_token_usage.get("output_tokens") or 0) + int(extra.get("output_tokens") or 0),
            "total_tokens": int(state.last_token_usage.get("total_tokens") or 0) + int(extra.get("total_tokens") or 0),
        }
        state.last_token_usage_is_estimate = True
    actions = actions_since(snapshot, scope=drain_scope) if drain_scope else actions_since(snapshot)

    state.last_actions = actions
    raw_tool_output = "".join(tool_chunks)
    max_chars = 20000
    if len(raw_tool_output) > max_chars:
        state.last_tool_output = raw_tool_output[:max_chars]
        state.last_tool_output_truncated = True
    else:
        state.last_tool_output = raw_tool_output
        state.last_tool_output_truncated = False

    if _COMPLETION_CLAIM_RE.search(reply) and not actions:
        print("本轮未执行任何工具调用，因此未产生可验证的文件/命令执行结果。", flush=True)

    return reply


def handle_local_command(
    user_text: str,
    messages: list[dict[str, str]],
    history: list[str],
    state: CliState,
) -> str | None:
    if not user_text:
        return "continue"

    if user_text.startswith("!"):
        cmd = user_text[1:].strip()
        if not cmd:
            print("用法：!<命令>")
            return "continue"
        print(f"! {cmd}")
        print(run_cli.invoke({"command": cmd, "stream": False}))
        return "continue"

    if not user_text.startswith("/"):
        return None

    parts = user_text[1:].strip().split()
    cmd = (parts[0].lower() if parts else "").strip()
    args = parts[1:]

    if cmd in {"quit", "exit", "q"}:
        return "quit"

    if cmd in {"reset", "r"}:
        messages.clear()
        history.clear()
        os.environ["AGENT_THREAD_ID"] = uuid.uuid4().hex[:12]
        print("已重置对话与历史。")
        return "continue"

    if cmd in {"help", "h", "?"}:
        print(
            "\n".join(
                [
                    "内置命令：",
                    "- /help                       显示帮助",
                    "- /ls [path]                  列出目录",
                    "- /lsr [path]                 递归列出目录",
                    "- /cat <path>                 查看文件内容",
                    "- /rm <path>                  删除文件/空目录",
                    "- /rmr <path>                 递归删除目录",
                    "- /pwd                        显示当前工作目录",
                    "- /cd <path>                  切换工作目录（限制在项目目录内）",
                    "- /history [n]                查看历史输入（默认 20）",
                    "- /tools                      查看可用工具概览",
                    "- /skills                     查看技能目录",
                    "- /last [k]                   查看最近输出（k=actions/tools/skills/all）",
                    "- /verbose <on|off>           开关工具输出显示",
                    "- /reset                      清空对话与历史",
                    "- /quit                       退出",
                    "- !<command>                  直接执行命令（在 work_dir 下）",
                ]
            )
        )
        return "continue"

    if cmd in {"ls", "dir"}:
        path = args[0] if args else "."
        print(f"ls {path}")
        print(list_dir.invoke({"path": path, "recursive": False}))
        return "continue"

    if cmd in {"lsr", "tree"}:
        path = args[0] if args else "."
        print(f"lsr {path}")
        print(list_dir.invoke({"path": path, "recursive": True}))
        return "continue"

    if cmd in {"cat", "type"}:
        if not args:
            print("用法：/cat <path>")
            return "continue"
        print(f"cat {args[0]}")
        return "continue"

    if cmd == "rm":
        if not args:
            print("用法：/rm <path>")
            return "continue"
        print(f"rm {args[0]}")
        print(delete_path.invoke({"path": args[0], "recursive": False}))
        return "continue"

    if cmd == "rmr":
        if not args:
            print("用法：/rmr <path>")
            return "continue"
        print(f"rmr {args[0]}")
        print(delete_path.invoke({"path": args[0], "recursive": True}))
        return "continue"

    if cmd in {"pwd", "cwd"}:
        project_root = Path(os.environ.get("AGENT_PROJECT_DIR", ".")).resolve()
        output_dir = Path(os.environ.get("AGENT_OUTPUT_DIR", ".")).resolve()
        work_dir = Path(os.environ.get("AGENT_WORK_DIR", ".")).resolve()
        print(
            "\n".join(
                [
                    f"cwd: {Path.cwd().resolve().as_posix()}",
                    f"project: {project_root.as_posix()}",
                    f"work: {work_dir.as_posix()}",
                    f"output: {output_dir.as_posix()}",
                ]
            )
        )
        return "continue"

    if cmd == "cd":
        if not args:
            print("用法：/cd <path>")
            return "continue"
        project_root = Path(os.environ.get("AGENT_PROJECT_DIR", ".")).resolve()
        target = Path(args[0])
        if not target.is_absolute():
            target = (Path.cwd() / target)
        try:
            resolved = target.resolve()
        except OSError as e:
            print(f"无效路径：{args[0]}（{e}）")
            return "continue"
        try:
            resolved.relative_to(project_root)
        except ValueError:
            print(f"拒绝切换到项目目录之外：{resolved.as_posix()}")
            return "continue"
        if not resolved.exists() or not resolved.is_dir():
            print(f"不是目录：{resolved.as_posix()}")
            return "continue"
        os.environ["AGENT_WORK_DIR"] = str(resolved)
        os.chdir(resolved)
        print(f"cd {resolved.as_posix()}")
        return "continue"

    if cmd in {"clear", "cls"}:
        os.system("cls" if os.name == "nt" else "clear")
        return "continue"

    if cmd in {"history", "his"}:
        n = 20
        if args:
            try:
                n = max(1, int(args[0]))
            except ValueError:
                print("用法：/history [n]")
                return "continue"
        for i, item in enumerate(history[-n:], start=max(1, len(history) - n + 1)):
            print(f"{i}. {item}")
        return "continue"

    if cmd in {"tools", "t"}:
        project_root = Path(os.environ.get("AGENT_PROJECT_DIR", ".")).resolve()
        output_dir = Path(os.environ.get("AGENT_OUTPUT_DIR", ".")).resolve()
        work_dir = Path(os.environ.get("AGENT_WORK_DIR", ".")).resolve()
        print(
            "\n".join(
                [
                    "可用工具（供智能体调用）：",
                    "- Read / Write / Edit",
                    "- Glob / Grep",
                    "- Bash",
                    "- write_file（限制写入 output_dir）",
                    "- read_file / list_dir / write_project_file / delete_path / run_cli",
                    f"工具输出显示: {'on' if state.show_tool_output else 'off'}",
                    f"project_dir: {project_root.as_posix()}",
                    f"work_dir: {work_dir.as_posix()}",
                    f"output_dir: {output_dir.as_posix()}",
                ]
            )
        )
        return "continue"

    if cmd in {"skills"}:
        if not state.skill_catalog_text:
            print("暂无技能目录。")
            return "continue"
        _print_block(f"技能目录（共 {state.skill_count} 个）", state.skill_catalog_text)
        return "continue"

    if cmd in {"last"}:
        kind = args[0] if args else "all"
        _print_last(state, kind=kind)
        return "continue"

    if cmd in {"verbose", "v"}:
        if not args or args[0].lower() not in {"on", "off"}:
            print("用法：/verbose <on|off>")
            return "continue"
        state.show_tool_output = args[0].lower() == "on"
        print(f"工具输出直显已{'开启' if state.show_tool_output else '关闭'}。")
        return "continue"

    print("未知命令，输入 /help 查看。")
    return "continue"
