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

from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessageChunk

from agents.runtime import (
    _agent_stream_config,
    _extract_text,
    _stream_text_delta,
    build_agent,
    build_user_message,
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


def _dotenv_parse(path: Path) -> dict[str, str]:
    try:
        p = path.expanduser().resolve()
    except Exception:
        p = path
    if not p.exists() or not p.is_file():
        return {}
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    out: dict[str, str] = {}
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
        out[k] = v
    return out


def _apply_workspace_dotenv(path: Path, *, prev: dict[str, str]) -> dict[str, str]:
    cur = _dotenv_parse(path)
    removed = set(prev.keys()) - set(cur.keys())
    for k in removed:
        try:
            if os.environ.get(k) == prev.get(k):
                os.environ.pop(k, None)
        except Exception:
            pass
    for k, v in cur.items():
        os.environ[k] = v
    return cur


def _file_sig(p: Path) -> tuple[float, int]:
    try:
        st = p.stat()
        return float(getattr(st, "st_mtime", 0.0)), int(getattr(st, "st_size", 0))
    except Exception:
        return 0.0, 0


def _resolve_project_root_guess(repo_root: Path) -> Path:
    raw = (os.environ.get("AGENT_PROJECT_DIR") or "").strip() or "workspace"
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def _resolve_effective_config_path(*, repo_root: Path, project_root_guess: Path, explicit_config: str) -> Path:
    if (explicit_config or "").strip():
        p = Path(explicit_config).expanduser()
        try:
            return p.resolve()
        except Exception:
            return p
    ws_cfg = (project_root_guess / "agent.json").resolve()
    if ws_cfg.exists() and ws_cfg.is_file():
        return ws_cfg
    return (repo_root / "agent.json").resolve()

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

    explicit_config_raw = (os.environ.get("AGENT_CONFIG_PATH") or "").strip()
    _load_dotenv_file(cwd / ".env", override=False)
    if cwd != repo_root:
        _load_dotenv_file(repo_root / ".env", override=False)

    project_root_guess = _resolve_project_root_guess(repo_root)
    _apply_workspace_dotenv((project_root_guess / ".env").resolve(), prev={})

    config_path = _resolve_effective_config_path(
        repo_root=repo_root,
        project_root_guess=project_root_guess,
        explicit_config=explicit_config_raw,
    )
    try:
        config_path = config_path.expanduser().resolve()
    except Exception:
        pass
    if config_path.parent != repo_root and config_path.parent != cwd:
        _load_dotenv_file(config_path.parent / ".env", override=False)

    os.environ["AGENT_CONFIG_PATH"] = str(config_path)
    cfg = load_config(config_path) or {}
    apply_config_to_environ(cfg, override=False)
    project_root_guess2 = _resolve_project_root_guess(repo_root)
    if project_root_guess2.resolve() != project_root_guess.resolve():
        _apply_workspace_dotenv((project_root_guess2 / ".env").resolve(), prev={})
        if not explicit_config_raw:
            ws_cfg2 = (project_root_guess2 / "agent.json").resolve()
            if ws_cfg2.exists() and ws_cfg2.is_file() and ws_cfg2 != config_path:
                config_path = ws_cfg2
                os.environ["AGENT_CONFIG_PATH"] = str(config_path)
                cfg = load_config(config_path) or {}
                apply_config_to_environ(cfg, override=True)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return config_path


def _maybe_reload_runtime(app: FastAPI) -> None:
    runtime: AgentRuntime | None = getattr(app.state, "runtime", None)
    if runtime is None:
        return
    repo_root: Path | None = getattr(app.state, "repo_root", None)
    if repo_root is None:
        return
    explicit_config_raw = getattr(app.state, "explicit_config_raw", "") or ""
    reload_lock: threading.Lock | None = getattr(app.state, "reload_lock", None)
    if reload_lock is None:
        return

    with reload_lock:
        runtime = app.state.runtime
        project_root = runtime.project_root
        env_path = (project_root / ".env").resolve()
        cfg_path = (
            (project_root / "agent.json").resolve()
            if (not explicit_config_raw) and (project_root / "agent.json").exists()
            else Path(os.environ.get("AGENT_CONFIG_PATH") or (repo_root / "agent.json")).resolve()
        )
        env_sig = _file_sig(env_path)
        cfg_sig = _file_sig(cfg_path)
        if env_sig == getattr(app.state, "last_ws_env_sig", (0.0, 0)) and cfg_sig == getattr(app.state, "last_cfg_sig", (0.0, 0)):
            return

        prev_env = getattr(app.state, "workspace_env", {}) or {}
        app.state.workspace_env = _apply_workspace_dotenv(env_path, prev=prev_env)
        app.state.last_ws_env_sig = env_sig

        os.environ["AGENT_CONFIG_PATH"] = str(cfg_path)
        cfg = load_config(cfg_path) or {}
        apply_config_to_environ(cfg, override=True)
        app.state.last_cfg_sig = cfg_sig

        try:
            runtime.memory_manager.stop(flush=False)
        except Exception:
            pass
        app.state.runtime = _build_runtime(repo_root)
        prev_mgr = getattr(app.state, "system_manager", None)
        if prev_mgr is not None:
            try:
                prev_mgr.stop()
            except Exception:
                pass
        next_mgr = None
        try:
            from system.manager import SystemManager

            runtime2: AgentRuntime = app.state.runtime
            notify_user = getattr(app.state, "notify_user", None)
            next_mgr = SystemManager(
                project_root=runtime2.project_root,
                output_dir=runtime2.output_dir,
                work_dir=runtime2.work_dir,
                model_name=runtime2.model_name,
                observer_agent=runtime2.agent,
                memory_manager=runtime2.memory_manager,
                notify_user=notify_user if callable(notify_user) else None,
            )
            next_mgr.start()
        except Exception:
            next_mgr = None
        app.state.system_manager = next_mgr


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

async def _ws_send_text(ws: WebSocket, send_lock: asyncio.Lock, text: str) -> None:
    async with send_lock:
        await ws.send_text(text)


async def _ws_send_json(ws: WebSocket, send_lock: asyncio.Lock, obj: Any) -> None:
    await _ws_send_text(ws, send_lock, _json_dumps(obj))


async def _broadcast(app: FastAPI, obj: dict[str, Any]) -> None:
    clients = getattr(app.state, "ws_clients", None) or {}
    if not isinstance(clients, dict) or not clients:
        return
    items = list(clients.items())
    text = _json_dumps(obj)
    target_user_id = (obj.get("user_id") or "") if isinstance(obj, dict) else ""
    target_user_id = target_user_id.strip() if isinstance(target_user_id, str) else ""
    dead: list[WebSocket] = []
    for ws, meta in items:
        lock = None
        user_id = ""
        if isinstance(meta, asyncio.Lock):
            lock = meta
        elif isinstance(meta, dict):
            lock0 = meta.get("lock")
            if isinstance(lock0, asyncio.Lock):
                lock = lock0
            uid0 = meta.get("user_id")
            if isinstance(uid0, str) and uid0.strip():
                user_id = uid0.strip()
        if lock is None:
            continue
        if target_user_id and user_id != target_user_id:
            continue
        try:
            await _ws_send_text(ws, lock, text)
        except Exception:
            dead.append(ws)
    if dead:
        for ws in dead:
            try:
                clients.pop(ws, None)
            except Exception:
                pass


def _int_env(name: str, default: int, *, min_v: int, max_v: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return max(min_v, min(max_v, int(default)))
    try:
        v = int(raw)
    except ValueError:
        v = int(default)
    return max(min_v, min(max_v, v))


def _safe_suffix(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    if "." not in s:
        return ""
    suf = "." + s.rsplit(".", 1)[-1].lower()
    if re.fullmatch(r"\.[a-z0-9]{1,8}", suf):
        return suf
    return ""


def _asr_client() -> tuple[object | None, str, str | None]:
    model = (os.environ.get("AGENT_ASR_MODEL") or "").strip() or "whisper-1"
    api_key = (os.environ.get("AGENT_ASR_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (os.environ.get("AGENT_ASR_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    if not api_key:
        return None, model, base_url
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception:
        return None, model, base_url
    kwargs: dict[str, object] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    try:
        return OpenAI(**kwargs), model, base_url
    except Exception:
        return None, model, base_url


def _asr_transcribe_openai(*, file_path: Path, model: str, language: str | None, prompt: str | None) -> tuple[str | None, str | None]:
    client, fallback_model, _ = _asr_client()
    if client is None:
        return None, "asr_backend_unavailable"
    use_model = (model or "").strip() or fallback_model
    try:
        with file_path.open("rb") as f:
            kwargs: dict[str, object] = {"model": use_model, "file": f}
            if language and language.strip():
                kwargs["language"] = language.strip()
            if prompt and prompt.strip():
                kwargs["prompt"] = prompt.strip()
            resp = client.audio.transcriptions.create(**kwargs)  # type: ignore[attr-defined]
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip(), None
        return "", None
    except Exception as e:
        return None, f"asr_failed:{type(e).__name__}"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    repo_root = Path(__file__).resolve().parents[1]
    explicit_config_raw = (os.environ.get("AGENT_CONFIG_PATH") or "").strip()
    config_path = _init_environ(repo_root)
    runtime = _build_runtime(repo_root)
    app.state.repo_root = repo_root
    app.state.explicit_config_raw = explicit_config_raw
    app.state.runtime = runtime
    app.state.reload_lock = threading.Lock()
    app.state.ws_clients = {}
    app.state.push_loop = asyncio.get_running_loop()
    env_path = (runtime.project_root / ".env").resolve()
    app.state.workspace_env = _apply_workspace_dotenv(env_path, prev={})
    app.state.last_ws_env_sig = _file_sig(env_path)
    app.state.last_cfg_sig = _file_sig(config_path)
    def _notify_user(payload: dict[str, object]) -> None:
        loop = getattr(app.state, "push_loop", None)
        if loop is None:
            return
        obj: dict[str, Any] = payload if isinstance(payload, dict) else {"type": "push", "text": str(payload)}
        try:
            asyncio.run_coroutine_threadsafe(_broadcast(app, obj), loop)
        except Exception:
            pass
    app.state.notify_user = _notify_user
    system_manager = None
    try:
        from system.manager import SystemManager

        system_manager = SystemManager(
            project_root=runtime.project_root,
            output_dir=runtime.output_dir,
            work_dir=runtime.work_dir,
            model_name=runtime.model_name,
            observer_agent=runtime.agent,
            memory_manager=runtime.memory_manager,
            notify_user=_notify_user,
        )
        system_manager.start()
    except Exception:
        system_manager = None
    app.state.system_manager = system_manager
    try:
        yield
    finally:
        mgr = getattr(app.state, "system_manager", None)
        if mgr is not None:
            try:
                mgr.stop()
            except Exception:
                pass
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


@app.post("/voice/transcribe")
async def voice_transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    model: str | None = Form(None),
    transcribe: bool = Form(True),
) -> JSONResponse:
    _maybe_reload_runtime(app)
    runtime: AgentRuntime = app.state.runtime

    max_bytes = _int_env("AGENT_VOICE_MAX_BYTES", 25 * 1024 * 1024, min_v=64 * 1024, max_v=200 * 1024 * 1024)
    suffix = _safe_suffix(file.filename or "") or ".bin"
    voice_dir = (runtime.output_dir / "voice").resolve()
    voice_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:16]
    saved = (voice_dir / f"{file_id}{suffix}").resolve()
    try:
        saved.relative_to(runtime.output_dir)
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid_save_path"}, status_code=500)

    written = 0
    try:
        with saved.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    return JSONResponse({"ok": False, "error": "file_too_large", "max_bytes": max_bytes}, status_code=413)
                out.write(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass

    rel_path = saved.relative_to(runtime.project_root).as_posix()
    payload: dict[str, object] = {
        "ok": True,
        "file_id": file_id,
        "saved_path": rel_path,
        "bytes": written,
    }

    if not transcribe:
        payload["transcribe"] = {"ok": False, "skipped": True}
        return JSONResponse(payload)

    text, err = await asyncio.to_thread(
        _asr_transcribe_openai, file_path=saved, model=(model or ""), language=language, prompt=prompt
    )
    if err is not None:
        payload["transcribe"] = {"ok": False, "error": err}
        return JSONResponse(payload)
    payload["transcribe"] = {"ok": True, "text": text or ""}
    return JSONResponse(payload)

@app.websocket("/ws")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    send_lock = asyncio.Lock()
    clients = getattr(ws.app.state, "ws_clients", None)
    user_id = "default"
    if isinstance(clients, dict):
        clients[ws] = {"lock": send_lock, "user_id": user_id}
    thread_id = uuid.uuid4().hex

    try:
        while True:
            _maybe_reload_runtime(ws.app)
            runtime: AgentRuntime = ws.app.state.runtime
            raw = await ws.receive_text()
            if not raw:
                continue

            if raw.strip() == "/reset":
                thread_id = uuid.uuid4().hex
                await _ws_send_json(ws, send_lock, {"type": "reset"})
                continue

            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"text": raw}

            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                await _ws_send_json(ws, send_lock, {"type": "error", "message": "text 不能为空"})
                continue
            if isinstance(payload, dict):
                uid = payload.get("user_id")
                if isinstance(uid, str) and uid.strip() and uid.strip() != user_id:
                    user_id = uid.strip()
                    if isinstance(clients, dict):
                        meta = clients.get(ws)
                        if isinstance(meta, dict):
                            meta["user_id"] = user_id
            user_text = text.strip()

            images: list[str] = []
            videos: list[str] = []
            if isinstance(payload, dict):
                img = payload.get("image")
                if isinstance(img, str) and img.strip():
                    images.append(img.strip())
                imgs = payload.get("images")
                if isinstance(imgs, list):
                    for it in imgs:
                        if isinstance(it, str) and it.strip():
                            images.append(it.strip())
                vid = payload.get("video")
                if isinstance(vid, str) and vid.strip():
                    videos.append(vid.strip())
                vids = payload.get("videos")
                if isinstance(vids, list):
                    for it in vids:
                        if isinstance(it, str) and it.strip():
                            videos.append(it.strip())

            user_msg = build_user_message(user_text, images=images or None, videos=videos or None)
            request_messages = [user_msg]

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
                            config=_agent_stream_config(checkpoint_ns="ws", thread_id=thread_id, user_id=user_id),
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

            while True:
                item = await q.get()
                if item is None:
                    break
                await _ws_send_json(ws, send_lock, item)

            assistant_text2 = assistant_full.get("text")
            usage2 = assistant_full.get("usage")
            if isinstance(assistant_text2, str) and assistant_text2.strip():
                try:
                    runtime.memory_manager.record_turn(
                        user_text=user_text,
                        assistant_text=assistant_text2,
                        token_usage=usage2 if isinstance(usage2, dict) else None,
                    )
                except Exception:
                    pass
    finally:
        clients2 = getattr(ws.app.state, "ws_clients", None)
        if isinstance(clients2, dict):
            try:
                clients2.pop(ws, None)
            except Exception:
                pass
