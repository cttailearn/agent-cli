from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shlex
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import BaseModel, Field


ExecHost = Literal["local"]
ExecSecurity = Literal["deny", "allowlist", "full"]
ExecAsk = Literal["off", "on-miss", "always"]


DANGEROUS_ENV_KEYS = {
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "NODE_OPTIONS",
    "NODE_PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "RUBYLIB",
    "PERL5LIB",
    "BASH_ENV",
    "ENV",
    "GCONV_PATH",
    "IFS",
    "SSLKEYLOGFILE",
}
DANGEROUS_ENV_PREFIXES = ("DYLD_", "LD_")


_DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bformat(\.com)?\b",
        r"\bshutdown\b",
        r"\brestart-computer\b",
        r"\bstop-computer\b",
        r"\bremove-item\b",
        r"\brm\b",
        r"\bdel\b",
        r"\berase\b",
        r"\brmdir\b",
        r"\brd\b",
        r"\bcurl\b",
        r"\bwget\b",
        r"\binvoke-webrequest\b",
        r"\binvoke-restmethod\b",
        r"\bssh\b",
        r"\bscp\b",
        r"\bset-executionpolicy\b",
        r"\b-encodedcommand\b",
        r"\b-enc\b",
    ]
)


def sanitize_exec_env(env: Dict[str, str]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for k, v in env.items():
        key = (k or "").strip()
        if not key:
            continue
        upper = key.upper()
        if upper.startswith(DANGEROUS_ENV_PREFIXES):
            continue
        if upper in DANGEROUS_ENV_KEYS:
            continue
        cleaned[key] = v
    return cleaned


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _extract_abs_paths(command: str) -> list[str]:
    c = command or ""
    patterns = [
        r"(?<![\w])([A-Za-z]:\\[^\s'\"`]+)",
        r"(\\\\[^\s'\"`]+)",
        r"(?<![\w])(/[^ \t\r\n'\"`]+)",
    ]
    out: list[str] = []
    for pat in patterns:
        for m in re.finditer(pat, c):
            s = (m.group(1) or "").strip()
            if s:
                out.append(s)
    return out


_SCRIPT_EXTS = {".ps1", ".sh", ".bash", ".bat", ".cmd", ".py", ".js", ".ts"}
_SCRIPT_MAX_BYTES = 200_000


def _safe_read_script_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > _SCRIPT_MAX_BYTES:
        data = data[:_SCRIPT_MAX_BYTES]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        try:
            return data.decode(errors="replace")
        except Exception:
            return None


def _resolve_script_candidate(raw: str, cwd: Path) -> Path | None:
    s = (raw or "").strip().strip("'\"`")
    if not s:
        return None
    try:
        p = Path(s).expanduser()
    except Exception:
        return None
    if not p.is_absolute():
        p = (cwd / p)
    try:
        p = p.resolve()
    except OSError:
        return None
    if p.suffix.lower() not in _SCRIPT_EXTS:
        return None
    if not p.exists() or not p.is_file():
        return None
    return p


def _find_script_path(tokens: list[str], cwd: Path) -> Path | None:
    if not tokens:
        return None
    first = tokens[0]
    direct = _resolve_script_candidate(first, cwd)
    if direct is not None:
        return direct

    exe = Path(first.strip("'\"`")).name.lower()
    file_flag_idx = None
    for i, t in enumerate(tokens):
        tt = (t or "").lower()
        if tt in {"-file", "/file", "-f"}:
            file_flag_idx = i
            break
    if file_flag_idx is not None and file_flag_idx + 1 < len(tokens):
        cand = _resolve_script_candidate(tokens[file_flag_idx + 1], cwd)
        if cand is not None:
            return cand

    if exe in {"python", "python3", "py", "node", "bash", "sh", "pwsh", "powershell"}:
        for t in tokens[1:]:
            if not t or t.startswith("-"):
                continue
            cand = _resolve_script_candidate(t, cwd)
            if cand is not None:
                return cand
    return None


def _validate_script_text(script_text: str, work_root: Path) -> str | None:
    if not script_text:
        return None
    for rx in _DANGEROUS_COMMAND_PATTERNS:
        if rx.search(script_text):
            return f"blocked_pattern_in_script:{rx.pattern}"
    for raw in _extract_abs_paths(script_text):
        try:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                continue
            if not _is_under_root(p, work_root):
                return f"absolute_path_outside_work_root_in_script:{p.resolve().as_posix()}"
        except Exception:
            return f"invalid_path_in_script:{raw}"
    return None


def sandbox_validate_command(*, command: str, work_root: Path, cwd: Path) -> str | None:
    mode = (os.environ.get("AGENT_SANDBOX") or "on").strip().lower()
    if mode in {"0", "off", "false", "no"}:
        return None

    cmd = (command or "").strip()
    if not cmd:
        return "empty_command"

    for rx in _DANGEROUS_COMMAND_PATTERNS:
        if rx.search(cmd):
            return f"blocked_pattern:{rx.pattern}"

    for raw in _extract_abs_paths(cmd):
        try:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                continue
            if not _is_under_root(p, work_root):
                return f"absolute_path_outside_work_root:{p.resolve().as_posix()}"
        except Exception:
            return f"invalid_path:{raw}"

    try:
        tokens = _tokenize_shell_command(cmd)
    except Exception:
        return None
    for t in tokens:
        if not isinstance(t, str):
            continue
        if ".." not in t:
            continue
        if "/" not in t and "\\" not in t:
            continue
        raw_tok = t.strip("'\"`")
        try:
            p = Path(raw_tok)
            if p.is_absolute():
                candidate = p
            else:
                candidate = (cwd / p)
            if not _is_under_root(candidate, work_root):
                return f"path_escape_attempt:{raw_tok}"
        except Exception:
            return f"invalid_path_token:{raw_tok}"

    script_path = _find_script_path(tokens, cwd)
    if script_path is not None:
        if not _is_under_root(script_path, work_root):
            return f"script_outside_work_root:{script_path.as_posix()}"
        script_text = _safe_read_script_text(script_path)
        if script_text is not None:
            deny = _validate_script_text(script_text, work_root)
            if deny:
                return deny
    return None


def _validate_host_env(env: Dict[str, str], forbid_path_override: bool) -> None:
    for raw_key in env.keys():
        key = raw_key.strip()
        if not key:
            continue
        upper = key.upper()
        if upper.startswith(DANGEROUS_ENV_PREFIXES):
            raise ValueError(f"forbidden env var: {raw_key}")
        if upper in DANGEROUS_ENV_KEYS:
            raise ValueError(f"forbidden env var: {raw_key}")
        if forbid_path_override and upper == "PATH":
            raise ValueError("custom PATH is forbidden for host execution")


def _tokenize_shell_command(command: str) -> List[str]:
    return shlex.split(command, posix=(os.name != "nt"))


def _split_into_segments(tokens: List[str]) -> List[List[str]]:
    seps = {"|", "&&", "||", ";"}
    segments: List[List[str]] = []
    current: List[str] = []
    for tok in tokens:
        if tok in seps:
            if current:
                segments.append(current)
            current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)
    return segments


def _resolved_executable_path(exe: str) -> Optional[str]:
    if os.path.isabs(exe) and os.path.exists(exe):
        return exe
    return shutil.which(exe)


def _matches_allowlist(resolved_path: Optional[str], allowlist: List[str]) -> bool:
    if not resolved_path:
        return False
    for pat in allowlist:
        p = pat.strip()
        if not p:
            continue
        if fnmatch.fnmatch(resolved_path, p):
            return True
    return False


def _looks_like_pathish(arg: str) -> bool:
    if "/" in arg or "\\" in arg:
        return True
    return False


def _is_safe_bin_usage(argv: List[str], safe_bins: List[str], cwd: str) -> bool:
    if not argv:
        return False
    exe = argv[0]
    if exe not in safe_bins:
        return False
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        if _looks_like_pathish(arg):
            return False
        candidate = os.path.join(cwd, arg)
        if os.path.exists(candidate):
            return False
    return True


def _evaluate_allowlist(
    command: str,
    cwd: str,
    allowlist: List[str],
    safe_bins: List[str],
) -> Tuple[bool, bool]:
    try:
        tokens = _tokenize_shell_command(command)
    except Exception:
        return False, False

    segments = _split_into_segments(tokens)
    if not segments:
        return False, False

    analysis_ok = True
    satisfied = True
    for seg in segments:
        if not seg:
            satisfied = False
            continue
        exe = seg[0]
        resolved = _resolved_executable_path(exe)
        if _matches_allowlist(resolved, allowlist):
            continue
        if _is_safe_bin_usage(seg, safe_bins, cwd):
            continue
        satisfied = False
    return analysis_ok, satisfied


def _requires_approval(
    ask: ExecAsk,
    security: ExecSecurity,
    analysis_ok: bool,
    allowlist_satisfied: bool,
) -> bool:
    if ask == "always":
        return True
    if ask == "off":
        return False
    if security != "allowlist":
        return False
    if not analysis_ok or not allowlist_satisfied:
        return True
    return False


@dataclass
class ExecSession:
    session_id: str
    command: str
    cwd: str
    started_at: float
    process: asyncio.subprocess.Process
    stdout_chunks: List[str] = field(default_factory=list)
    stderr_chunks: List[str] = field(default_factory=list)
    aggregated: str = ""
    tail: str = ""
    max_output_chars: int = 200_000
    backgrounded: bool = False
    exited: bool = False
    exit_code: Optional[int] = None

    def append_output(self, text: str) -> None:
        if not text:
            return
        next_agg = (self.aggregated + text) if self.aggregated else text
        if len(next_agg) > self.max_output_chars:
            next_agg = next_agg[-self.max_output_chars :]
        self.aggregated = next_agg
        self.tail = next_agg[-400:]


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: Dict[str, ExecSession] = {}

    def add(self, session: ExecSession) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[ExecSession]:
        return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()

    def list(self) -> List[ExecSession]:
        return sorted(self._sessions.values(), key=lambda s: s.started_at, reverse=True)


class ExecInput(BaseModel):
    command: str = Field(..., description="Shell command to execute")
    workdir: Optional[str] = Field(None, description="Working directory")
    env: Optional[Dict[str, str]] = Field(None, description="Extra environment variables")
    yield_ms: Optional[int] = Field(10_000, description="Milliseconds before returning running")
    background: Optional[bool] = Field(False, description="Return immediately with session id")
    timeout_sec: Optional[int] = Field(1800, description="Kill process after timeout seconds")
    stdin_data: Optional[str] = Field(None, description="Initial stdin data to write")
    stdin_eof: Optional[bool] = Field(False, description="Close stdin after writing stdin_data")
    host: Optional[ExecHost] = Field("local", description="Execution host")
    security: Optional[ExecSecurity] = Field("full", description="deny|allowlist|full")
    ask: Optional[ExecAsk] = Field("on-miss", description="off|on-miss|always")


class ProcessInput(BaseModel):
    action: Literal["list", "poll", "log", "write", "kill", "remove", "clear"]
    session_id: Optional[str] = None
    data: Optional[str] = None
    newline: Optional[bool] = None
    offset: Optional[int] = None
    limit: Optional[int] = None
    eof: Optional[bool] = None


class ExecTool(BaseTool):
    name: str = "exec"
    description: str = (
        "Execute shell commands with optional backgrounding. "
        "Use yield_ms/background to return a session id; manage sessions via process tool."
    )
    args_schema: type[BaseModel] = ExecInput

    def __init__(
        self,
        registry: SessionRegistry,
        *,
        allowlist: Optional[List[str]] = None,
        safe_bins: Optional[List[str]] = None,
        forbid_path_override: bool = True,
        approval_callback: Optional[Callable[[str], Literal["allow-once", "deny"]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._registry = registry
        self._allowlist = allowlist or []
        self._safe_bins = safe_bins or []
        self._forbid_path_override = forbid_path_override
        self._approval_callback = approval_callback

    async def _arun(self, command: str, **kwargs: Any) -> Any:
        params = ExecInput(command=command, **kwargs)
        if not params.command.strip():
            raise ValueError("command required")

        work_root_env = (os.environ.get("AGENT_WORK_DIR") or os.environ.get("AGENT_OUTPUT_DIR") or "").strip()
        work_root = Path(work_root_env).expanduser() if work_root_env else Path(os.getcwd())
        try:
            work_root = work_root.resolve()
        except OSError:
            work_root = Path(os.getcwd()).resolve()

        raw_cwd = (params.workdir or os.getcwd()).strip() or os.getcwd()
        try:
            cwd_path = Path(raw_cwd).expanduser()
            if not cwd_path.is_absolute():
                cwd_path = (work_root / cwd_path)
            cwd_path = cwd_path.resolve()
        except OSError:
            cwd_path = work_root
        if not _is_under_root(cwd_path, work_root):
            return {"status": "denied", "reason": "cwd_outside_work_root", "cwd": cwd_path.as_posix()}

        deny_reason = sandbox_validate_command(command=params.command, work_root=work_root, cwd=cwd_path)
        if deny_reason:
            return {"status": "denied", "reason": deny_reason, "cwd": cwd_path.as_posix()}

        cwd = cwd_path.as_posix()
        base_env = sanitize_exec_env(dict(os.environ))
        user_env = params.env or {}
        _validate_host_env(user_env, forbid_path_override=self._forbid_path_override)
        env = {**base_env, **user_env}

        security: ExecSecurity = params.security or "allowlist"
        ask: ExecAsk = params.ask or "on-miss"

        if security == "deny":
            return {"status": "denied", "reason": "security=deny"}

        analysis_ok, allowlist_satisfied = _evaluate_allowlist(
            params.command, cwd, self._allowlist, self._safe_bins
        )
        if security == "allowlist" and analysis_ok and not allowlist_satisfied:
            if not _requires_approval(ask, security, analysis_ok, allowlist_satisfied):
                return {"status": "denied", "reason": "allowlist_miss"}

        if _requires_approval(ask, security, analysis_ok, allowlist_satisfied):
            decision = self._approval_callback(params.command) if self._approval_callback else None
            if decision == "deny" or decision is None:
                approval_id = str(uuid.uuid4())
                return {
                    "status": "approval_required",
                    "approval_id": approval_id,
                    "command": params.command,
                    "cwd": cwd,
                }

        started_at = time.time()
        proc = await asyncio.create_subprocess_shell(
            params.command,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        session_id = uuid.uuid4().hex[:12]
        session = ExecSession(
            session_id=session_id,
            command=params.command,
            cwd=cwd,
            started_at=started_at,
            process=proc,
        )
        self._registry.add(session)

        if proc.stdin is not None:
            if params.stdin_data is not None:
                proc.stdin.write(params.stdin_data.encode())
                await proc.stdin.drain()
            if params.stdin_eof:
                proc.stdin.close()

        async def pump(stream: Optional[asyncio.StreamReader], sink: Callable[[str], None]) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                sink(text)

        stdout_task = asyncio.create_task(pump(proc.stdout, session.append_output))
        stderr_task = asyncio.create_task(pump(proc.stderr, session.append_output))

        async def wait_and_finalize() -> None:
            try:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=float(params.timeout_sec or 0))
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            finally:
                session.exited = True
                session.exit_code = proc.returncode

        finalize_task = asyncio.create_task(wait_and_finalize())

        if params.background:
            session.backgrounded = True
            return {
                "status": "running",
                "session_id": session.session_id,
                "pid": proc.pid,
                "started_at": session.started_at,
                "cwd": session.cwd,
                "tail": session.tail,
            }

        yield_ms = int(params.yield_ms or 10_000)
        done, _ = await asyncio.wait({finalize_task}, timeout=yield_ms / 1000.0)

        if finalize_task in done:
            exit_code = session.exit_code
            ok = exit_code == 0
            return {
                "status": "completed" if ok else "failed",
                "exit_code": exit_code,
                "duration_ms": int((time.time() - started_at) * 1000),
                "aggregated": session.aggregated or "",
                "cwd": session.cwd,
            }

        session.backgrounded = True
        return {
            "status": "running",
            "session_id": session.session_id,
            "pid": proc.pid,
            "started_at": session.started_at,
            "cwd": session.cwd,
            "tail": session.tail,
        }

    def _run(self, command: str, **kwargs: Any) -> Any:
        return asyncio.run(self._arun(command, **kwargs))


class ProcessTool(BaseTool):
    name: str = "process"
    description: str = "Manage running exec sessions: list, poll, log, write, kill, remove, clear."
    args_schema: type[BaseModel] = ProcessInput

    def __init__(self, registry: SessionRegistry, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._registry = registry

    async def _arun(self, action: str, **kwargs: Any) -> Any:
        params = ProcessInput(action=action, **kwargs)

        if params.action == "list":
            items = []
            now = time.time()
            for s in self._registry.list():
                items.append(
                    {
                        "session_id": s.session_id,
                        "status": "running" if not s.exited else ("completed" if s.exit_code == 0 else "failed"),
                        "pid": s.process.pid,
                        "runtime_ms": int((now - s.started_at) * 1000),
                        "cwd": s.cwd,
                        "command": s.command,
                        "tail": s.tail,
                        "backgrounded": s.backgrounded,
                    }
                )
            return {"status": "completed", "sessions": items}

        if params.action == "clear":
            self._registry.clear()
            return {"status": "completed"}

        if not params.session_id:
            return {"status": "failed", "reason": "session_id required"}

        s = self._registry.get(params.session_id)
        if not s:
            return {"status": "failed", "reason": "session not found"}

        if params.action == "poll":
            return {
                "status": "completed" if s.exited else "running",
                "session_id": s.session_id,
                "exit_code": s.exit_code if s.exited else None,
                "tail": s.tail,
            }

        if params.action == "log":
            text = s.aggregated or ""
            offset = int(params.offset or 0)
            limit = int(params.limit or 4000)
            sliced = text[offset : offset + limit]
            return {"status": "completed", "session_id": s.session_id, "text": sliced}

        if params.action == "write":
            if s.process.stdin is None:
                return {"status": "failed", "reason": "stdin unavailable"}
            data = params.data or ""
            if params.newline:
                if not data.endswith("\n"):
                    data += "\n"
            s.process.stdin.write(data.encode())
            await s.process.stdin.drain()
            if params.eof:
                s.process.stdin.close()
            return {"status": "completed", "session_id": s.session_id}

        if params.action == "kill":
            if not s.exited:
                s.process.kill()
                await s.process.wait()
                s.exited = True
                s.exit_code = s.process.returncode
            return {"status": "completed", "session_id": s.session_id, "exit_code": s.exit_code}

        if params.action == "remove":
            self._registry.remove(s.session_id)
            return {"status": "completed", "session_id": s.session_id}

        return {"status": "failed", "reason": "unknown action"}

    def _run(self, action: str, **kwargs: Any) -> Any:
        return asyncio.run(self._arun(action, **kwargs))


def build_langchain_exec_tools(
    *,
    allowlist: Optional[List[str]] = None,
    safe_bins: Optional[List[str]] = None,
    forbid_path_override: bool = True,
) -> List[BaseTool]:
    registry = SessionRegistry()
    exec_tool = ExecTool(
        registry,
        allowlist=allowlist or [],
        safe_bins=safe_bins or [],
        forbid_path_override=forbid_path_override,
    )
    process_tool = ProcessTool(registry)
    return [exec_tool, process_tool]
