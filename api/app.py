from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessageChunk

from agents.runtime import (
    _agent_stream_config,
    _stream_text_delta,
    build_agent,
    capture_token_usage_from_message,
    estimate_token_usage,
    get_estimated_token_usage,
    get_token_usage,
    reset_estimated_token_usage,
    reset_token_usage,
)
from config_manager import apply_config_to_environ, load_config
from memory import MemoryManager, ensure_memory_scaffold
from memory import paths as memory_paths


_DOTENV_KEY_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_text(msg: object) -> str:
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


def _load_dotenv_file(path: Path, *, override: bool) -> None:
    try:
        p = path.expanduser().resolve()
    except Exception:
        p = path
    if not p.exists() or not p.is_file():
        return
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    for raw_line in (text or "").lstrip("\ufeff").splitlines():
        line = (raw_line or "").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = (k or "").strip()
        if not k or not _DOTENV_KEY_RX.fullmatch(k):
            continue
        v = (v or "").strip()
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1]
        if override:
            os.environ[k] = v
        else:
            os.environ.setdefault(k, v)


def _sanitize_workspace_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        return "default"
    if name in {".", "./"}:
        return "default"
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        return "default"
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "_", name).strip("/").strip()
    return cleaned or "default"


def _resolve_dir(base: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return p


def _assert_under_project(project_root: Path, p: Path, label: str) -> Path:
    resolved = p.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise RuntimeError(f"{label} must be under project_dir: {resolved.as_posix()}")
    return resolved


@dataclass(frozen=True, slots=True)
class AgentRuntime:
    agent: object
    project_root: Path
    output_dir: Path
    work_dir: Path
    model_name: str
    memory_manager: MemoryManager
    stream_lock: threading.RLock


def _init_environ(repo_root: Path) -> Path:
    try:
        cwd = Path.cwd().resolve()
    except Exception:
        cwd = Path.cwd()

    _load_dotenv_file(cwd / ".env", override=False)
    if cwd != repo_root:
        _load_dotenv_file(repo_root / ".env", override=False)

    config_path = Path(os.environ.get("AGENT_CONFIG_PATH") or (repo_root / "agent.json"))
    try:
        config_path = config_path.expanduser().resolve()
    except Exception:
        pass
    if config_path.parent != repo_root and config_path.parent != cwd:
        _load_dotenv_file(config_path.parent / ".env", override=False)

    os.environ["AGENT_CONFIG_PATH"] = str(config_path)
    cfg = load_config(config_path) or {}
    apply_config_to_environ(cfg, override=False)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return config_path


def _build_runtime(repo_root: Path) -> AgentRuntime:
    model_name = (os.environ.get("LC_MODEL") or "deepseek:deepseek-chat").strip()
    os.environ["LC_MODEL"] = model_name

    project_dir_raw = (os.environ.get("AGENT_PROJECT_DIR") or "").strip()
    project_root = Path(project_dir_raw).expanduser() if project_dir_raw else (repo_root / "workspace")
    if not project_root.is_absolute():
        project_root = (repo_root / project_root).resolve()
    else:
        project_root = project_root.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / ".agents").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("AGENT_MEMORY_DIR", str((project_root / "memory").resolve()))
    os.environ.setdefault("AGENT_LANGGRAPH_STORE_PATH", str((project_root / "memory" / "langgraph_store.json").resolve()))

    ensure_memory_scaffold(project_root)
    try:
        user_p = memory_paths.user_path(project_root)
        user_text = user_p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        user_text = ""
    bootstrap = (not (user_text or "").strip()) or ("<未设置>" in user_text)
    os.environ["AGENT_USER_BOOTSTRAP"] = "1" if bootstrap else "0"

    output_base_raw = (os.environ.get("AGENT_OUTPUT_DIR") or "out").strip()
    output_base_dir = Path(output_base_raw).expanduser()
    if not output_base_dir.is_absolute():
        output_base_dir = (project_root / output_base_dir).resolve()
    else:
        output_base_dir = output_base_dir.resolve()
    output_base_dir.mkdir(parents=True, exist_ok=True)

    output_workspace = _sanitize_workspace_name(os.environ.get("AGENT_OUTPUT_WORKSPACE") or "default")
    output_dir = (output_base_dir / output_workspace).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir_raw = (os.environ.get("AGENT_WORK_DIR") or "").strip()
    work_dir = Path(work_dir_raw).expanduser() if work_dir_raw else project_root
    if not work_dir.is_absolute():
        work_dir = (project_root / work_dir).resolve()
    else:
        work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    mcp_config_raw = (os.environ.get("AGENT_MCP_CONFIG") or "mcp/config.json").strip()
    mcp_config = Path(mcp_config_raw).expanduser()
    if not mcp_config.is_absolute():
        mcp_config = (project_root / mcp_config).resolve()
    else:
        mcp_config = mcp_config.resolve()
    os.environ["AGENT_MCP_CONFIG"] = str(mcp_config)

    os.environ["AGENT_PROJECT_DIR"] = str(project_root)
    os.environ["AGENT_OUTPUT_BASE_DIR"] = str(output_base_dir)
    os.environ["AGENT_OUTPUT_WORKSPACE"] = output_workspace
    os.environ["AGENT_OUTPUT_DIR"] = str(output_dir)
    os.environ["AGENT_WORK_DIR"] = str(work_dir)
    os.environ["AGENT_MODEL_NAME"] = model_name
    os.environ.setdefault("AGENT_STREAM_DELEGATION", "1")
    os.environ.setdefault("AGENT_STREAM_SUBPROCESS", "1")

    skills_dirs: list[Path] = []

    skills_dirs_raw = (os.environ.get("AGENT_SKILLS_DIRS") or "").strip()
    if skills_dirs_raw:
        parts = [s.strip() for s in re.split(r"[;,]+", skills_dirs_raw) if s.strip()]
        for s in parts:
            skills_dirs.append(_assert_under_project(project_root, _resolve_dir(project_root, s), "skills_dirs"))

    skills_dir_raw = (os.environ.get("AGENT_SKILLS_DIR") or "skills").strip()
    project_skills_dir = _assert_under_project(project_root, _resolve_dir(project_root, skills_dir_raw), "skills_dir")
    if project_skills_dir not in {p.resolve() for p in skills_dirs}:
        skills_dirs.insert(0, project_skills_dir)
    project_skills_dir.mkdir(parents=True, exist_ok=True)

    agent, _, _ = build_agent(
        skills_dirs=skills_dirs,
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
    )

    memory_manager = MemoryManager(project_root=project_root, model_name=model_name)
    memory_manager.start()

    return AgentRuntime(
        agent=agent,
        project_root=project_root,
        output_dir=output_dir,
        work_dir=work_dir,
        model_name=model_name,
        memory_manager=memory_manager,
        stream_lock=threading.RLock(),
    )


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


@asynccontextmanager
async def _lifespan(app: FastAPI):
    repo_root = Path(__file__).resolve().parents[1]
    _init_environ(repo_root)
    runtime = _build_runtime(repo_root)
    app.state.runtime = runtime
    try:
        yield
    finally:
        try:
            runtime.memory_manager.stop(flush=True)
        except Exception:
            pass


app = FastAPI(lifespan=_lifespan)

_cors_raw = (os.environ.get("AGENT_UI_ORIGINS") or "").strip()
_cors_origins = [s.strip() for s in re.split(r"[;,]+", _cors_raw) if s.strip()] if _cors_raw else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    runtime: AgentRuntime = ws.app.state.runtime
    thread_id = uuid.uuid4().hex
    conversation: list[dict[str, str]] = []

    while True:
        raw = await ws.receive_text()
        if not raw:
            continue

        if raw.strip() == "/reset":
            conversation = []
            thread_id = uuid.uuid4().hex
            await ws.send_text(_json_dumps({"type": "reset"}))
            continue

        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"text": raw}

        text = payload.get("text") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            await ws.send_text(_json_dumps({"type": "error", "message": "text 不能为空"}))
            continue
        user_text = text.strip()

        request_messages = [*conversation, {"role": "user", "content": user_text}]

        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        assistant_full: dict[str, Any] = {"text": "", "usage": None}
        loop = asyncio.get_running_loop()

        def _emit(obj: dict[str, Any]) -> None:
            asyncio.run_coroutine_threadsafe(q.put(obj), loop)

        def _finish() -> None:
            asyncio.run_coroutine_threadsafe(q.put(None), loop)

        def _worker() -> None:
            with runtime.stream_lock:
                reset_token_usage()
                reset_estimated_token_usage()
                assistant_assembled = ""
                chunks: list[str] = []
                try:
                    for event in runtime.agent.stream(
                        {"messages": request_messages},
                        stream_mode="messages",
                        config=_agent_stream_config(checkpoint_ns="ws", thread_id=thread_id),
                    ):
                        msg = event[0] if isinstance(event, tuple) and event else event
                        if isinstance(msg, AIMessageChunk):
                            capture_token_usage_from_message(msg)
                            t = _extract_text(msg)
                            delta, assistant_assembled2 = _stream_text_delta(t, assembled=assistant_assembled)
                            assistant_assembled = assistant_assembled2
                            if delta:
                                chunks.append(delta)
                                _emit({"type": "assistant_delta", "delta": delta})
                    assistant_text = assistant_assembled or "".join(chunks)
                    usage = get_token_usage()
                    if not usage.get("total_tokens"):
                        est = get_estimated_token_usage()
                        if est.get("total_tokens"):
                            usage = est
                        else:
                            usage = estimate_token_usage(user_text, assistant_text)
                    assistant_full["text"] = assistant_text
                    assistant_full["usage"] = usage
                    _emit({"type": "done", "assistant": assistant_text, "usage": usage})
                except Exception as e:
                    _emit({"type": "error", "message": str(e) or "运行失败"})
                finally:
                    _finish()

        t = threading.Thread(target=_worker, name=f"ws-agent-{thread_id}", daemon=True)
        t.start()

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                await ws.send_text(_json_dumps(item))
        finally:
            pass

        assistant_text2 = assistant_full.get("text")
        usage2 = assistant_full.get("usage")
        if isinstance(assistant_text2, str) and assistant_text2.strip():
            conversation.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text2}])
            if len(conversation) > 40:
                conversation = conversation[-40:]
            try:
                runtime.memory_manager.record_turn(
                    user_text=user_text,
                    assistant_text=assistant_text2,
                    token_usage=usage2 if isinstance(usage2, dict) else None,
                )
            except Exception:
                pass
