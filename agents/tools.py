import asyncio
import contextvars
import inspect
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from fnmatch import fnmatch
from langchain_core.tools import tool
from pathlib import Path

from langgraph.prebuilt import ToolRuntime

from .exec import build_langchain_exec_tools, sanitize_exec_env, sandbox_validate_command
from . import console_print


ACTION_LOG: list[dict[str, object]] = []
_ACTION_SCOPE: contextvars.ContextVar[str] = contextvars.ContextVar("ACTION_SCOPE", default="")


class _ActionScope:
    def __init__(self, scope: str) -> None:
        self._scope = scope
        self._token: contextvars.Token[str] | None = None

    def __enter__(self) -> None:
        self._token = _ACTION_SCOPE.set((self._scope or "").strip())

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None:
            _ACTION_SCOPE.reset(self._token)


def action_scope(scope: str) -> _ActionScope:
    return _ActionScope(scope)

try:
    from langchain_core.tools import BaseTool
    from langchain_core.tools.structured import StructuredTool

    def _tool_arg_names(t: BaseTool) -> list[str]:
        schema = getattr(t, "args_schema", None)
        if schema is not None:
            model_fields = getattr(schema, "model_fields", None)
            if isinstance(model_fields, dict) and model_fields:
                return list(model_fields.keys())
            fields_v1 = getattr(schema, "__fields__", None)
            if isinstance(fields_v1, dict) and fields_v1:
                return list(fields_v1.keys())

        fn = getattr(t, "func", None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
            except (TypeError, ValueError):
                return []
            names: list[str] = []
            for p in sig.parameters.values():
                if p.name == "self":
                    continue
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                    names.append(p.name)
            return names
        return []

    def _base_tool_call(self: BaseTool, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            return self.invoke(args[0])
        if args:
            names = _tool_arg_names(self)
            if not names:
                if len(args) == 1 and not kwargs:
                    return self.invoke({"input": args[0]})
                raise TypeError("Tools must be called with keyword arguments or a single dict argument.")

            remaining = [n for n in names if n not in kwargs]
            if len(args) > len(remaining):
                raise TypeError("Tools must be called with keyword arguments or a single dict argument.")
            payload = dict(kwargs)
            for n, v in zip(remaining, args):
                payload[n] = v
            return self.invoke(payload)

        return self.invoke(kwargs)

    if "__call__" not in BaseTool.__dict__:
        BaseTool.__call__ = _base_tool_call  # type: ignore[assignment]
    if "__call__" not in StructuredTool.__dict__:
        StructuredTool.__call__ = _base_tool_call  # type: ignore[assignment]
except Exception:
    pass


_SENSITIVE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_ed25519",
    ".npmrc",
    ".pypirc",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def action_log_snapshot() -> int:
    return len(ACTION_LOG)


def actions_since(snapshot: int, *, scope: str | None = None) -> list[dict[str, object]]:
    if snapshot <= 0:
        out = ACTION_LOG[:]
    else:
        out = ACTION_LOG[snapshot:]
    s = (scope or "").strip()
    if not s:
        return out
    return [a for a in out if a.get("scope") == s]


def _log_action(entry: dict[str, object]) -> None:
    if "scope" not in entry:
        entry["scope"] = _ACTION_SCOPE.get()
    ACTION_LOG.append(entry)


def log_action(kind: str, ok: bool, **fields: object) -> None:
    k = (kind or "").strip() or "unknown"
    entry: dict[str, object] = {"kind": k, "ok": bool(ok), "ts": time.time()}
    for fk, fv in fields.items():
        if fk in {"kind", "ok", "ts"}:
            continue
        entry[fk] = fv
    _log_action(entry)


def _project_root() -> Path:
    project_dir_env = os.environ.get("AGENT_PROJECT_DIR")
    if project_dir_env:
        return Path(project_dir_env).expanduser()
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / "workspace").resolve()


def _resolve_under_root(root: Path, path: str) -> tuple[Path | None, str | None]:
    try:
        root = root.resolve()
    except OSError as e:
        return None, f"Invalid root directory: {root}. Error: {e}"

    raw_path = Path(path)
    target_path = raw_path if raw_path.is_absolute() else (root / raw_path)
    try:
        resolved_target = target_path.resolve()
    except OSError as e:
        return None, f"Invalid path: {path}. Error: {e}"

    try:
        resolved_target.relative_to(root)
    except ValueError:
        return None, f"Refusing to access outside root: {resolved_target.as_posix()}"

    return resolved_target, None


def _is_sensitive_path(path: Path) -> bool:
    if any(part == ".git" for part in path.parts):
        return True
    name = path.name
    if name in _SENSITIVE_FILENAMES:
        return True
    if name.startswith(".env."):
        return True
    suffix = path.suffix.lower()
    if suffix and suffix in _SENSITIVE_SUFFIXES:
        return True
    return False


def _read_text_file(path: Path, encoding: str) -> tuple[str | None, str | None]:
    try:
        return path.read_text(encoding=encoding, errors="replace"), None
    except OSError as e:
        return None, str(e)


def _path_matches_any_glob(rel_posix: str, globs: list[str]) -> bool:
    for g in globs:
        if fnmatch(rel_posix, g):
            return True
    return False


@tool
def write_file(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
    """Write content to a file under the configured output directory.

    Args:
        path: File path, relative to the output directory or an absolute path within it
        content: Text content to write
        mode: File open mode, either "w" (overwrite) or "a" (append)
        encoding: Text encoding
    """
    if mode not in {"w", "a"}:
        return f"Unsupported mode: {mode}. Use 'w' or 'a'."

    project_root = Path(__file__).resolve().parent
    output_dir_env = os.environ.get("AGENT_OUTPUT_DIR")
    output_root = Path(output_dir_env).expanduser() if output_dir_env else project_root
    try:
        output_root = output_root.resolve()
    except OSError as e:
        return f"Invalid output directory: {output_root}. Error: {e}"

    raw_path = Path(path)
    target_path = raw_path if raw_path.is_absolute() else (output_root / raw_path)

    try:
        resolved_target = target_path.resolve()
    except OSError as e:
        return f"Invalid path: {path}. Error: {e}"

    try:
        resolved_target.relative_to(output_root)
    except ValueError:
        return f"Refusing to write outside output directory: {resolved_target.as_posix()}"

    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "w":
            resolved_target.write_text(content, encoding=encoding)
        else:
            with resolved_target.open("a", encoding=encoding) as f:
                f.write(content)
    except OSError as e:
        _log_action(
            {
                "kind": "write_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "mode": mode,
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to write file: {resolved_target.as_posix()}. Error: {e}"

    try:
        size = resolved_target.stat().st_size
    except OSError:
        size = -1
    _log_action(
        {
            "kind": "write_file",
            "ok": True,
            "path": resolved_target.as_posix(),
            "mode": mode,
            "size": size,
            "ts": time.time(),
        }
    )
    return f"Wrote file: {resolved_target.as_posix()} (size={size})"


@tool
def read_file(path: str, encoding: str = "utf-8", max_chars: int = 20000) -> str:
    """Read a text file under the configured project directory."""
    if not path or not path.strip():
        return "Empty path."
    if max_chars <= 0:
        return "max_chars must be > 0."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "read_file",
                "ok": False,
                "path": path,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to read sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "read_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    try:
        text = resolved_target.read_text(encoding=encoding, errors="replace")
    except OSError as e:
        _log_action(
            {
                "kind": "read_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to read file: {resolved_target.as_posix()}. Error: {e}"

    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    _log_action(
        {
            "kind": "read_file",
            "ok": True,
            "path": resolved_target.as_posix(),
            "truncated": truncated,
            "ts": time.time(),
        }
    )
    if truncated:
        return f"path: {resolved_target.as_posix()}\n...[truncated]\n{text}"
    return f"path: {resolved_target.as_posix()}\n{text}"


@tool
def list_dir(path: str = ".", recursive: bool = False, max_entries: int = 200) -> str:
    """List files/directories under the configured project directory."""
    if max_entries <= 0:
        return "max_entries must be > 0."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "list_dir",
                "ok": False,
                "path": path,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to access sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "list_dir",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    if not resolved_target.exists():
        return f"Path does not exist: {resolved_target.as_posix()}"
    if not resolved_target.is_dir():
        return f"Not a directory: {resolved_target.as_posix()}"

    entries: list[tuple[str, str]] = []
    try:
        if recursive:
            it = resolved_target.rglob("*")
        else:
            it = resolved_target.iterdir()
        for p in it:
            if len(entries) >= max_entries:
                break
            if _is_sensitive_path(p):
                continue
            rel = p.relative_to(root).as_posix()
            kind = "dir" if p.is_dir() else "file"
            entries.append((rel, kind))
    except OSError as e:
        _log_action(
            {
                "kind": "list_dir",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to list directory: {resolved_target.as_posix()}. Error: {e}"

    lines = [f"root: {root.resolve().as_posix()}", f"path: {resolved_target.as_posix()}"]
    for rel, kind in entries:
        lines.append(f"- {kind} {rel}")
    if len(entries) >= max_entries:
        lines.append(f"...(truncated at {max_entries} entries)")

    _log_action(
        {
            "kind": "list_dir",
            "ok": True,
            "path": resolved_target.as_posix(),
            "recursive": recursive,
            "entries": len(entries),
            "ts": time.time(),
        }
    )
    return "\n".join(lines)


@tool
def write_project_file(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str:
    """Write content to a file under the configured project directory."""
    if mode not in {"w", "a"}:
        return f"Unsupported mode: {mode}. Use 'w' or 'a'."
    if not path or not path.strip():
        return "Empty path."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "write_project_file",
                "ok": False,
                "path": path,
                "mode": mode,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to write sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "write_project_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "mode": mode,
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if mode == "w":
            resolved_target.write_text(content, encoding=encoding)
        else:
            with resolved_target.open("a", encoding=encoding) as f:
                f.write(content)
    except OSError as e:
        _log_action(
            {
                "kind": "write_project_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "mode": mode,
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to write file: {resolved_target.as_posix()}. Error: {e}"

    try:
        size = resolved_target.stat().st_size
    except OSError:
        size = -1

    _log_action(
        {
            "kind": "write_project_file",
            "ok": True,
            "path": resolved_target.as_posix(),
            "mode": mode,
            "size": size,
            "ts": time.time(),
        }
    )
    return f"Wrote file: {resolved_target.as_posix()} (size={size})"


@tool
def delete_path(path: str, recursive: bool = False) -> str:
    """Delete a file or directory under the configured project directory."""
    if not path or not path.strip():
        return "Empty path."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "delete_path",
                "ok": False,
                "path": path,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to delete sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "delete_path",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    if not resolved_target.exists():
        return f"Path does not exist: {resolved_target.as_posix()}"

    try:
        if resolved_target.is_dir():
            if not recursive:
                resolved_target.rmdir()
            else:
                for p in sorted(resolved_target.rglob("*"), reverse=True):
                    if p.is_dir():
                        p.rmdir()
                    else:
                        p.unlink()
                resolved_target.rmdir()
        else:
            resolved_target.unlink()
    except OSError as e:
        _log_action(
            {
                "kind": "delete_path",
                "ok": False,
                "path": resolved_target.as_posix(),
                "recursive": recursive,
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to delete: {resolved_target.as_posix()}. Error: {e}"

    _log_action(
        {
            "kind": "delete_path",
            "ok": True,
            "path": resolved_target.as_posix(),
            "recursive": recursive,
            "ts": time.time(),
        }
    )
    return f"Deleted: {resolved_target.as_posix()}"


@tool("Edit")
def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    encoding: str = "utf-8",
) -> str:
    """Edit a project file by replacing an exact string (optionally replace all)."""
    if not path or not path.strip():
        return "Empty path."
    if old_string is None or old_string == "":
        return "old_string must be a non-empty string."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": path,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to edit sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    original_text, read_err = _read_text_file(resolved_target, encoding=encoding)
    if read_err is not None or original_text is None:
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": read_err or "read_failed",
                "ts": time.time(),
            }
        )
        return f"Failed to read file: {resolved_target.as_posix()}. Error: {read_err}"

    count = original_text.count(old_string)
    if count == 0:
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": "old_string_not_found",
                "ts": time.time(),
            }
        )
        return "old_string not found in file."

    if not replace_all and count != 1:
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": f"old_string_not_unique(count={count})",
                "ts": time.time(),
            }
        )
        return f"old_string appears {count} times; set replace_all=true to replace all."

    if replace_all:
        updated_text = original_text.replace(old_string, new_string)
        replaced = count
    else:
        updated_text = original_text.replace(old_string, new_string, 1)
        replaced = 1

    try:
        resolved_target.write_text(updated_text, encoding=encoding)
    except OSError as e:
        _log_action(
            {
                "kind": "edit_file",
                "ok": False,
                "path": resolved_target.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to write file: {resolved_target.as_posix()}. Error: {e}"

    _log_action(
        {
            "kind": "edit_file",
            "ok": True,
            "path": resolved_target.as_posix(),
            "replaced": replaced,
            "ts": time.time(),
        }
    )
    return f"Edited: {resolved_target.as_posix()} (replaced={replaced})"


@tool("Glob")
def glob_paths(path: str = ".", pattern: str = "**/*", max_entries: int = 200) -> str:
    """Find files/dirs under project root by glob pattern."""
    if max_entries <= 0:
        return "max_entries must be > 0."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "glob",
                "ok": False,
                "path": path,
                "pattern": pattern,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to access sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "glob",
                "ok": False,
                "path": resolved_target.as_posix(),
                "pattern": pattern,
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    if not resolved_target.exists():
        return f"Path does not exist: {resolved_target.as_posix()}"
    if not resolved_target.is_dir():
        return f"Not a directory: {resolved_target.as_posix()}"

    entries: list[tuple[str, str]] = []
    try:
        for p in resolved_target.glob(pattern):
            if len(entries) >= max_entries:
                break
            if _is_sensitive_path(p):
                continue
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                continue
            kind = "dir" if p.is_dir() else "file"
            entries.append((rel, kind))
    except OSError as e:
        _log_action(
            {
                "kind": "glob",
                "ok": False,
                "path": resolved_target.as_posix(),
                "pattern": pattern,
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to glob: {resolved_target.as_posix()}. Error: {e}"

    lines = [
        f"root: {root.resolve().as_posix()}",
        f"base: {resolved_target.as_posix()}",
        f"pattern: {pattern}",
    ]
    for rel, kind in entries:
        lines.append(f"- {kind} {rel}")
    if len(entries) >= max_entries:
        lines.append(f"...(truncated at {max_entries} entries)")

    _log_action(
        {
            "kind": "glob",
            "ok": True,
            "path": resolved_target.as_posix(),
            "pattern": pattern,
            "entries": len(entries),
            "ts": time.time(),
        }
    )
    return "\n".join(lines)


@tool("Grep")
def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    output_mode: str = "files_with_matches",
    ignore_case: bool = False,
    max_files: int = 200,
    max_matches: int = 200,
    head_limit: int = 100,
    encoding: str = "utf-8",
    max_file_size_bytes: int = 1_000_000,
    ignore_globs: list[str] | None = None,
) -> str:
    """Search text files under project root with a regex (lightweight ripgrep-like)."""
    if not pattern:
        return "Empty pattern."
    if max_files <= 0 or max_matches <= 0 or head_limit <= 0:
        return "max_files/max_matches/head_limit must be > 0."
    if output_mode not in {"content", "files_with_matches", "count"}:
        return "output_mode must be one of: content, files_with_matches, count."

    root = _project_root()
    resolved_target, err = _resolve_under_root(root, path)
    if err:
        _log_action(
            {
                "kind": "grep",
                "ok": False,
                "path": path,
                "pattern": pattern,
                "error": err,
                "ts": time.time(),
            }
        )
        return err
    assert resolved_target is not None

    if _is_sensitive_path(resolved_target):
        msg = f"Refusing to access sensitive path: {resolved_target.as_posix()}"
        _log_action(
            {
                "kind": "grep",
                "ok": False,
                "path": resolved_target.as_posix(),
                "pattern": pattern,
                "error": "sensitive_path",
                "ts": time.time(),
            }
        )
        return msg

    flags = re.MULTILINE
    if ignore_case:
        flags |= re.IGNORECASE
    try:
        rx = re.compile(pattern, flags=flags)
    except re.error as e:
        return f"Invalid regex pattern. Error: {e}"

    ignore_globs = ignore_globs or []
    files_scanned = 0
    files_with_matches: list[str] = []
    match_count_by_file: dict[str, int] = {}
    content_lines: list[str] = []
    total_matches = 0

    def _iter_files() -> list[Path]:
        if resolved_target.is_file():
            return [resolved_target]
        if resolved_target.is_dir():
            base_glob = glob or "**/*"
            return [p for p in resolved_target.glob(base_glob) if p.is_file()]
        return []

    for p in _iter_files():
        if files_scanned >= max_files or total_matches >= max_matches:
            break
        if _is_sensitive_path(p):
            continue
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if ignore_globs and _path_matches_any_glob(rel, ignore_globs):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if max_file_size_bytes > 0 and size > max_file_size_bytes:
            continue

        text, read_err = _read_text_file(p, encoding=encoding)
        if read_err is not None or text is None:
            continue

        files_scanned += 1
        matches_in_file = 0
        for line_no, line in enumerate(text.splitlines(), start=1):
            if total_matches >= max_matches:
                break
            if rx.search(line):
                matches_in_file += 1
                total_matches += 1
                if output_mode == "content":
                    content_lines.append(f"{rel}:{line_no}:{line}")
                    if len(content_lines) >= head_limit:
                        break

        if matches_in_file > 0:
            files_with_matches.append(rel)
            match_count_by_file[rel] = matches_in_file
            if output_mode == "files_with_matches" and len(files_with_matches) >= head_limit:
                break
            if output_mode == "count" and len(match_count_by_file) >= head_limit:
                break

    _log_action(
        {
            "kind": "grep",
            "ok": True,
            "path": resolved_target.as_posix(),
            "pattern": pattern,
            "output_mode": output_mode,
            "files_scanned": files_scanned,
            "files_with_matches": len(files_with_matches),
            "matches": total_matches,
            "ts": time.time(),
        }
    )

    header = [
        f"root: {root.resolve().as_posix()}",
        f"path: {resolved_target.as_posix()}",
        f"pattern: {pattern}",
        f"output_mode: {output_mode}",
    ]
    if output_mode == "content":
        body = content_lines[:head_limit]
        if len(content_lines) >= head_limit:
            body.append(f"...(truncated at {head_limit} lines)")
        return "\n".join(header + body)
    if output_mode == "count":
        lines: list[str] = []
        for rel, c in list(match_count_by_file.items())[:head_limit]:
            lines.append(f"{rel}:{c}")
        if len(match_count_by_file) >= head_limit:
            lines.append(f"...(truncated at {head_limit} files)")
        lines.append(f"total_matches: {total_matches}")
        return "\n".join(header + lines)
    files = files_with_matches[:head_limit]
    if len(files_with_matches) >= head_limit:
        files.append(f"...(truncated at {head_limit} files)")
    return "\n".join(header + files)


@tool
def run_cli(
    command: str,
    timeout_s: int = 120,
    cwd: str | None = None,
    encoding: str = "utf-8",
    stream: bool = False,
    stdin_data: str | None = None,
    stdin_eof: bool = True,
) -> str:
    """Run a CLI command from within the agent work directory."""
    if not command or not command.strip():
        return "Empty command."
    if timeout_s <= 0:
        return "timeout_s must be > 0."

    project_root = Path(__file__).resolve().parent
    work_dir_env = os.environ.get("AGENT_WORK_DIR") or os.environ.get("AGENT_OUTPUT_DIR")
    work_root = Path(work_dir_env).expanduser() if work_dir_env else project_root
    try:
        work_root = work_root.resolve()
    except OSError as e:
        return f"Invalid work directory: {work_root}. Error: {e}"

    raw_cwd = Path(cwd) if cwd else Path(".")
    target_cwd = raw_cwd if raw_cwd.is_absolute() else (work_root / raw_cwd)
    try:
        resolved_cwd = target_cwd.resolve()
    except OSError as e:
        return f"Invalid cwd: {cwd}. Error: {e}"
    try:
        resolved_cwd.relative_to(work_root)
    except ValueError:
        return f"Refusing to run outside work directory: {resolved_cwd.as_posix()}"

    deny_reason = sandbox_validate_command(command=command, work_root=work_root, cwd=resolved_cwd)
    if deny_reason:
        _log_action(
            {
                "kind": "run_cli",
                "ok": False,
                "command": command,
                "cwd": resolved_cwd.as_posix(),
                "error": f"sandbox_denied:{deny_reason}",
                "ts": time.time(),
            }
        )
        return f"Sandbox denied: {deny_reason}"

    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    else:
        cmd = ["bash", "-lc", command]

    env = sanitize_exec_env(dict(os.environ))
    stdin_spec = subprocess.DEVNULL if stdin_data is None else subprocess.PIPE

    def _kill_process_tree(pid: int) -> None:
        if pid <= 0:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
            except Exception:
                pass
            return
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass

    try:
        if not stream:
            process = subprocess.Popen(
                cmd,
                cwd=str(resolved_cwd),
                stdin=stdin_spec,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                encoding=encoding,
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
            try:
                stdout, stderr = process.communicate(input=stdin_data, timeout=timeout_s)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                pid = int(process.pid or 0)
                _kill_process_tree(pid)
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                try:
                    stdout, stderr = process.communicate(timeout=2)
                except Exception:
                    stdout = ""
                    stderr = ""
                exit_code = 124
                _log_action(
                    {
                        "kind": "run_cli",
                        "ok": False,
                        "command": command,
                        "cwd": resolved_cwd.as_posix(),
                        "exit_code": exit_code,
                        "timeout_s": timeout_s,
                        "ts": time.time(),
                    }
                )
                tail = "\n".join((stdout or "").splitlines()[-40:])
                return "\n".join([f"Command timed out after {timeout_s}s.", "stdout_tail:", tail]).rstrip()
        else:
            console_print(f"$ {command}", flush=True, ensure_newline=True)
            process = subprocess.Popen(
                cmd,
                cwd=str(resolved_cwd),
                stdin=stdin_spec,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                encoding=encoding,
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
            if stdin_data is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin_data)
                    process.stdin.flush()
                except OSError:
                    pass
                if stdin_eof:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass

            output_lines: list[str] = []
            output_lock = threading.Lock()

            def _reader() -> None:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\n")
                    console_print(line, flush=True, ensure_newline=True)
                    with output_lock:
                        output_lines.append(line)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                pid = int(process.pid or 0)
                _kill_process_tree(pid)
                try:
                    exit_code = process.wait(timeout=5)
                except Exception:
                    exit_code = 124

            t.join(timeout=2)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass

            with output_lock:
                stdout = "\n".join(output_lines)
            stderr = ""
            if timed_out:
                _log_action(
                    {
                        "kind": "run_cli",
                        "ok": False,
                        "command": command,
                        "cwd": resolved_cwd.as_posix(),
                        "exit_code": exit_code,
                        "timeout_s": timeout_s,
                        "ts": time.time(),
                    }
                )
                tail = "\n".join((stdout or "").splitlines()[-40:])
                return "\n".join([f"Command timed out after {timeout_s}s.", "stdout_tail:", tail]).rstrip()
    except FileNotFoundError as e:
        _log_action(
            {
                "kind": "run_cli",
                "ok": False,
                "command": command,
                "cwd": resolved_cwd.as_posix(),
                "error": f"Shell not found: {e}",
                "ts": time.time(),
            }
        )
        return f"Shell not found: {e}"
    except OSError as e:
        _log_action(
            {
                "kind": "run_cli",
                "ok": False,
                "command": command,
                "cwd": resolved_cwd.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return f"Failed to run command. Error: {e}"

    max_chars = 12000
    if len(stdout) > max_chars:
        stdout = stdout[:max_chars] + "\n...[truncated]"
    if len(stderr) > max_chars:
        stderr = stderr[:max_chars] + "\n...[truncated]"

    _log_action(
        {
            "kind": "run_cli",
            "ok": True,
            "command": command,
            "cwd": resolved_cwd.as_posix(),
            "exit_code": exit_code,
            "ts": time.time(),
        }
    )
    return "\n".join(
        [
            f"cwd: {resolved_cwd.as_posix()}",
            f"ok: {str(exit_code == 0).lower()}",
            f"exit_code: {exit_code}",
            "stdout:",
            stdout.rstrip(),
            "stderr:",
            stderr.rstrip(),
        ]
    ).rstrip()


Read = tool("Read")(read_file.func)
Write = tool("Write")(write_project_file.func)


def _bash(command: str, timeout_s: int = 120, cwd: str | None = None, encoding: str = "utf-8") -> str:
    """Run a shell command within the configured work directory."""
    return run_cli.func(command=command, timeout_s=timeout_s, cwd=cwd, encoding=encoding, stream=False)


Bash = tool("Bash")(_bash)

Exec, Process = build_langchain_exec_tools()


def _run_coroutine(coro):
    def _run_in_new_loop():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            return result
        finally:
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass
            loop.close()

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return _run_in_new_loop()

    if running_loop.is_running():
        result_box: dict[str, object] = {}
        ctx = contextvars.copy_context()

        def _thread_runner():
            try:
                result_box["value"] = ctx.run(_run_in_new_loop)
            except BaseException as e:
                result_box["error"] = e

        t = threading.Thread(target=_thread_runner, daemon=True)
        t.start()
        t.join()
        err = result_box.get("error")
        if isinstance(err, BaseException):
            raise err
        return result_box.get("value")

    return _run_in_new_loop()


def _ensure_sync_invoke_tools(tools: list[object]) -> list[object]:
    if not tools:
        return []
    try:
        from langchain_core.tools.structured import StructuredTool
    except Exception:
        return list(tools)

    wrapped: list[object] = []
    for t in tools:
        if isinstance(t, StructuredTool):
            func = getattr(t, "func", None)
            coroutine = getattr(t, "coroutine", None)
            if func is None and coroutine is not None:

                def _sync_func(*, __coroutine=coroutine, **kwargs):
                    return _run_coroutine(__coroutine(**kwargs))

                wrapped.append(
                    StructuredTool.from_function(
                        func=_sync_func,
                        coroutine=coroutine,
                        name=t.name,
                        description=t.description,
                        return_direct=getattr(t, "return_direct", False),
                        args_schema=getattr(t, "args_schema", None),
                        infer_schema=False,
                        response_format=getattr(t, "response_format", "content"),
                        tags=getattr(t, "tags", None),
                        metadata=getattr(t, "metadata", None),
                    )
                )
                continue
        wrapped.append(t)
    return wrapped


def load_mcp_tools_from_config(config_path: str | Path | None = None) -> list[object]:
    config_path_str = str(config_path) if config_path is not None else os.environ.get("AGENT_MCP_CONFIG", "")
    if not config_path_str:
        return []

    p = Path(config_path_str).expanduser()
    if not p.is_absolute():
        p = (_project_root() / p).resolve()
    else:
        p = p.resolve()

    if not p.exists() or not p.is_file():
        return []

    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace") or "{}")
    except Exception as e:
        _log_action(
            {
                "kind": "mcp_load_config",
                "ok": False,
                "path": p.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return []

    if isinstance(raw, dict) and isinstance(raw.get("mcpServers"), dict):
        servers = raw.get("mcpServers") or {}
        tool_name_prefix = bool(raw.get("tool_name_prefix", True))
    elif isinstance(raw, dict):
        servers = raw
        tool_name_prefix = True
    else:
        return []

    connections: dict[str, dict[str, object]] = {}
    for name, cfg in servers.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(cfg, dict):
            continue
        if cfg.get("enabled") is False:
            continue
        cfg = dict(cfg)
        cfg.pop("enabled", None)
        transport = cfg.get("transport")
        if not isinstance(transport, str) or not transport.strip():
            cfg.pop("transport", None)
            transport_from_type = cfg.get("type")
            if isinstance(transport_from_type, str) and transport_from_type.strip():
                cfg.pop("type", None)
                cfg["transport"] = transport_from_type.strip()
            elif "command" in cfg:
                cfg["transport"] = "stdio"
            elif "url" in cfg:
                cfg["transport"] = "streamable_http"
        connections[name.strip()] = cfg

    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as e:
        _log_action(
            {
                "kind": "mcp_import",
                "ok": False,
                "path": p.as_posix(),
                "error": str(e),
                "ts": time.time(),
            }
        )
        return []

    loaded_tools: list[object] = []
    failed: dict[str, str] = {}

    for name, cfg in connections.items():
        try:
            client = MultiServerMCPClient({name: cfg}, tool_name_prefix=tool_name_prefix)
            server_tools = _run_coroutine(client.get_tools())
            if isinstance(server_tools, list):
                loaded_tools.extend(server_tools)
        except Exception as e:
            failed[name] = str(e)
            _log_action(
                {
                    "kind": "mcp_get_tools_server",
                    "ok": False,
                    "path": p.as_posix(),
                    "server": name,
                    "error": str(e),
                    "ts": time.time(),
                }
            )

    _log_action(
        {
            "kind": "mcp_get_tools",
            "ok": True,
            "path": p.as_posix(),
            "servers": list(connections.keys()),
            "failed_servers": list(failed.keys()),
            "tool_count": len(loaded_tools),
            "ts": time.time(),
        }
    )
    return _ensure_sync_invoke_tools(list(loaded_tools))


def _memory_project_root() -> Path:
    try:
        return _project_root().expanduser().resolve()
    except OSError:
        return Path(__file__).resolve().parent


@tool
def memory_core_read(kind: str) -> str:
    """Read long-term core memory markdown: soul / traits / identity / user."""
    try:
        from memory.paths import core_file_by_kind
    except Exception as e:
        return f"Import failed: {e}"
    p = core_file_by_kind(_memory_project_root(), kind)
    if p is None:
        return "Unknown kind. Use soul|traits|identity|user."
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Read failed: {e}"


@tool
def memory_core_append(kind: str, content: str) -> str:
    """Append content to core memory markdown: soul / traits / identity / user."""
    try:
        from memory.paths import core_file_by_kind
    except Exception as e:
        return f"Import failed: {e}"
    p = core_file_by_kind(_memory_project_root(), kind)
    if p is None:
        return "Unknown kind. Use soul|traits|identity|user."
    p.parent.mkdir(parents=True, exist_ok=True)
    text = (content or "").strip()
    if not text:
        return "Empty content."
    try:
        existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        sep = "\n\n" if existing and not existing.endswith("\n") else "\n"
        p.write_text(f"{existing}{sep}{text}\n", encoding="utf-8", errors="replace")
        _log_action({"kind": "memory_core_append", "ok": True, "path": p.as_posix(), "size": len(text), "ts": time.time()})
        return "OK"
    except OSError as e:
        _log_action({"kind": "memory_core_append", "ok": False, "path": p.as_posix(), "error": str(e), "ts": time.time()})
        return f"Write failed: {e}"

@tool
def memory_core_write(kind: str, content: str) -> str:
    """Overwrite core memory markdown: soul / traits / identity / user."""
    try:
        from memory.paths import core_file_by_kind
    except Exception as e:
        return f"Import failed: {e}"
    p = core_file_by_kind(_memory_project_root(), kind)
    if p is None:
        return "Unknown kind. Use soul|traits|identity|user."
    p.parent.mkdir(parents=True, exist_ok=True)
    text = (content or "").strip()
    try:
        p.write_text(f"{text}\n" if text else "", encoding="utf-8", errors="replace")
        _log_action({"kind": "memory_core_write", "ok": True, "path": p.as_posix(), "size": len(text), "ts": time.time()})
        return "OK"
    except OSError as e:
        _log_action({"kind": "memory_core_write", "ok": False, "path": p.as_posix(), "error": str(e), "ts": time.time()})
        return f"Write failed: {e}"


@tool
def memory_user_read() -> str:
    """Read user memory markdown."""
    try:
        from memory.paths import user_path
    except Exception as e:
        return f"Import failed: {e}"
    p = user_path(_memory_project_root())
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Read failed: {e}"


@tool
def memory_user_write(content: str) -> str:
    """Overwrite user memory markdown."""
    try:
        from memory.paths import user_path
    except Exception as e:
        return f"Import failed: {e}"
    p = user_path(_memory_project_root())
    p.parent.mkdir(parents=True, exist_ok=True)
    text = (content or "").strip()
    try:
        p.write_text(f"{text}\n" if text else "", encoding="utf-8", errors="replace")
        _log_action({"kind": "memory_user_write", "ok": True, "path": p.as_posix(), "size": len(text), "ts": time.time()})
        return "OK"
    except OSError as e:
        _log_action({"kind": "memory_user_write", "ok": False, "path": p.as_posix(), "error": str(e), "ts": time.time()})
        return f"Write failed: {e}"


def _session_memory_md_path(project_root: Path, date_str: str) -> tuple[Path | None, str | None]:
    raw = (date_str or "").strip()
    if not raw or raw.lower() in {"today", "now"}:
        d = date.today().isoformat()
    else:
        d = raw
    try:
        date.fromisoformat(d)
    except Exception:
        return None, "Invalid date. Use YYYY-MM-DD."
    try:
        from memory.paths import episodic_dir
    except Exception as e:
        return None, f"Import failed: {e}"
    root = episodic_dir(project_root)
    p = (root / f"{d}.md").resolve()
    return p, None


def _parse_session_md(text: str) -> dict[str, str]:
    lines = (text or "").splitlines()
    out: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        if line.startswith("## "):
            current = line[3:].strip()
            if current:
                out.setdefault(current, [])
            else:
                current = None
            continue
        if current is None:
            continue
        out[current].append(line)
    return {k: "\n".join(v).strip() for k, v in out.items()}

def _session_memory_md_paths(project_root: Path, selector: str) -> tuple[list[tuple[str, Path]], str | None]:
    raw = (selector or "").strip()
    if not raw or raw.lower() in {"today", "now"}:
        d = date.today().isoformat()
        p, err = _session_memory_md_path(project_root, d)
        if err:
            return [], err
        assert p is not None
        return [(d, p)], None

    try:
        from memory.paths import episodic_dir
    except Exception as e:
        return [], f"Import failed: {e}"

    sessions_dir = episodic_dir(project_root)
    if raw.lower() in {"all", "*"}:
        if not sessions_dir.exists() or not sessions_dir.is_dir():
            return [], None
        out: list[tuple[str, Path]] = []
        for p in sorted(sessions_dir.glob("*.md")):
            name = p.stem
            try:
                date.fromisoformat(name)
            except Exception:
                continue
            out.append((name, p.resolve()))
        out.sort(key=lambda x: x[0], reverse=True)
        return out, None

    if ".." in raw:
        parts = [s.strip() for s in raw.split("..", 1)]
        if len(parts) == 2:
            a, b = parts
            try:
                da = date.fromisoformat(a)
                db = date.fromisoformat(b)
            except Exception:
                return [], "Invalid date range. Use YYYY-MM-DD..YYYY-MM-DD."
            if da > db:
                da, db = db, da
            out: list[tuple[str, Path]] = []
            cur = da
            while cur <= db:
                ds = cur.isoformat()
                out.append((ds, (sessions_dir / f"{ds}.md").resolve()))
                cur = cur.fromordinal(cur.toordinal() + 1)
            return out, None

    if len(raw) == 7 and raw[4] == "-":
        yyyy_mm = raw
        if not sessions_dir.exists() or not sessions_dir.is_dir():
            return [], None
        out: list[tuple[str, Path]] = []
        for p in sorted(sessions_dir.glob(f"{yyyy_mm}-*.md")):
            name = p.stem
            try:
                date.fromisoformat(name)
            except Exception:
                continue
            out.append((name, p.resolve()))
        out.sort(key=lambda x: x[0], reverse=True)
        return out, None

    p, err = _session_memory_md_path(project_root, raw)
    if err:
        return [], err
    assert p is not None
    return [(raw, p)], None


def _infer_date_selector_from_question(q: str) -> str:
    s = (q or "").strip()
    if not s:
        return "today"
    s2 = s.replace(" ", "")
    today = date.today()
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", s):
        m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", s)
        if m:
            return m.group(1)
    if any(x in s2 for x in ["今天", "今日", "现在", "刚才"]):
        return today.isoformat()
    if "昨天" in s2:
        return (today - timedelta(days=1)).isoformat()
    if "前天" in s2:
        return (today - timedelta(days=2)).isoformat()
    if any(x in s2 for x in ["本周", "这周", "这一周"]):
        start = today - timedelta(days=today.weekday())
        return f"{start.isoformat()}..{today.isoformat()}"
    if any(x in s2 for x in ["上周", "上一周"]):
        end = today - timedelta(days=today.weekday() + 1)
        start = end - timedelta(days=6)
        return f"{start.isoformat()}..{end.isoformat()}"
    if any(x in s2 for x in ["本月", "这个月"]):
        return today.strftime("%Y-%m")
    if "上个月" in s2:
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        return last_prev_month.strftime("%Y-%m")
    m = re.search(r"(最近|过去)(\d{1,3})天", s2)
    if m:
        n = int(m.group(2))
        n = max(1, min(365, n))
        start = today - timedelta(days=n - 1)
        return f"{start.isoformat()}..{today.isoformat()}"
    if any(x in s2 for x in ["之前", "上次", "以前", "还记得", "曾经"]):
        return "all"
    return today.isoformat()


def _infer_keyword_from_question(q: str) -> str:
    s = (q or "").strip()
    if not s:
        return ""
    s2 = s
    for w in ["今天", "昨日", "昨天", "前天", "本周", "这周", "上周", "本月", "这个月", "上个月", "最近", "过去"]:
        s2 = s2.replace(w, " ")
    s2 = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", s2)
    s2 = re.sub(r"(最近|过去)\d{1,3}天", " ", s2)
    if re.search(r"(做了什么|干了什么|做了些(什么|啥)|发生了什么|有什么进展|进展怎么样)", s2.replace(" ", "")):
        s2 = re.sub(r"(你|我|我们)?(做了什么|干了什么|做了些(什么|啥)|发生了什么|有什么进展|进展怎么样)\??", " ", s2)
    s2 = re.sub(r"[？?。！!，,：:；;（）()【】\[\]<>《》“”\"'`]+", " ", s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    if not s2:
        return ""
    for w in ["关于", "相关", "一下", "一些", "哪些", "什么", "怎么", "怎么样", "情况", "事情", "内容", "结果", "结论"]:
        s2 = s2.replace(w, " ")
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2[:80]

def _rollup_md_path(project_root: Path, *, layer: str, key: str) -> tuple[Path | None, str | None]:
    lay = (layer or "").strip().lower()
    k = (key or "").strip()
    try:
        from memory.paths import rollup_dir
    except Exception as e:
        return None, f"Import failed: {e}"

    if lay in {"day", "daily"}:
        try:
            date.fromisoformat(k)
        except Exception:
            return None, "Invalid daily rollup key. Use YYYY-MM-DD."
        return (rollup_dir(project_root, "daily") / f"{k}.md").resolve(), None
    if lay in {"week", "weekly"}:
        if not re.match(r"^\d{4}-W\d{2}$", k):
            return None, "Invalid weekly rollup key. Use YYYY-Www (e.g., 2026-W06)."
        return (rollup_dir(project_root, "weekly") / f"{k}.md").resolve(), None
    if lay in {"month", "monthly"}:
        if not re.match(r"^\d{4}-\d{2}$", k):
            return None, "Invalid monthly rollup key. Use YYYY-MM."
        return (rollup_dir(project_root, "monthly") / f"{k}.md").resolve(), None
    if lay in {"year", "yearly"}:
        if not re.match(r"^\d{4}$", k):
            return None, "Invalid yearly rollup key. Use YYYY."
        return (rollup_dir(project_root, "yearly") / f"{k}.md").resolve(), None
    return None, "Invalid rollup layer."

def _infer_rollup_target_from_question(q: str, date_selector: str) -> tuple[str, str] | None:
    s = (q or "").strip()
    ds = (date_selector or "").strip()
    s2 = s.replace(" ", "")
    today = date.today()

    if any(x in s2 for x in ["今年", "本年", "这一年"]):
        return "yearly", f"{today.year:04d}"
    if any(x in s2 for x in ["去年", "上一年"]):
        return "yearly", f"{today.year - 1:04d}"

    if any(x in s2 for x in ["本月", "这个月"]):
        return "monthly", today.strftime("%Y-%m")
    if "上个月" in s2:
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        return "monthly", last_prev_month.strftime("%Y-%m")

    if any(x in s2 for x in ["今天", "今日"]):
        return "daily", today.isoformat()
    if "昨天" in s2:
        return "daily", (today - timedelta(days=1)).isoformat()
    if "前天" in s2:
        return "daily", (today - timedelta(days=2)).isoformat()

    if any(x in s2 for x in ["本周", "这周", "这一周"]):
        iso_year, iso_week, _ = today.isocalendar()
        return "weekly", f"{int(iso_year):04d}-W{int(iso_week):02d}"
    if any(x in s2 for x in ["上周", "上一周"]):
        ref = today - timedelta(days=today.weekday() + 1)
        iso_year, iso_week, _ = ref.isocalendar()
        return "weekly", f"{int(iso_year):04d}-W{int(iso_week):02d}"

    if re.match(r"^\d{4}-\d{2}$", ds):
        return "monthly", ds
    if re.match(r"^\d{4}-\d{2}-\d{2}$", ds):
        return "daily", ds
    if ".." in ds:
        parts = [x.strip() for x in ds.split("..", 1)]
        if len(parts) == 2:
            a, b = parts
            try:
                da = date.fromisoformat(a)
                db = date.fromisoformat(b)
            except Exception:
                return None
            if da > db:
                da, db = db, da
            if (db.toordinal() - da.toordinal()) <= 6:
                if da.weekday() == 0 and db.weekday() == 6:
                    iso_year, iso_week, _ = da.isocalendar()
                    return "weekly", f"{int(iso_year):04d}-W{int(iso_week):02d}"
    return None

def _filter_text(text: str, tokens: list[str], *, max_lines: int = 220) -> str:
    if not tokens:
        return (text or "").strip()
    out: list[str] = []
    for line in (text or "").splitlines():
        low = line.lower()
        if line.startswith("#") or line.startswith("## "):
            if out and out[-1] != "":
                out.append("")
            out.append(line)
            continue
        if not line.strip():
            continue
        if any(t in low for t in tokens):
            out.append(line)
        if len(out) >= max_lines:
            break
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out).strip()


@tool
def memory_session_query(question: str, date: str = "") -> str:
    """Query session memories by natural language question (auto date + scene/keyword)."""
    q = (question or "").strip()
    if not q:
        return ""
    ds = (date or "").strip() or _infer_date_selector_from_question(q)
    kw = _infer_keyword_from_question(q)
    project_root = _memory_project_root()
    rollup = _infer_rollup_target_from_question(q, ds)
    if rollup is not None:
        layer, key = rollup
        p, err = _rollup_md_path(project_root, layer=layer, key=key)
        if err:
            return err
        assert p is not None
        if p.exists() and p.is_file():
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Read failed: {e}"
            if not kw:
                return text.strip()
            tokens = [t.lower() for t in re.split(r"\s+", kw) if t.strip()]
            filtered = _filter_text(text, tokens)
            if filtered:
                return filtered
    if not kw:
        if ds.lower() in {"all", "*"}:
            today = datetime.now().date()
            start = today - timedelta(days=6)
            ds = f"{start.isoformat()}..{today.isoformat()}"
        paths, err = _session_memory_md_paths(project_root, ds)
        if err:
            return err
        out_lines: list[str] = []
        for d, p in paths:
            if not p.exists() or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            items = _parse_session_md(text)
            if not items:
                continue
            if len(paths) != 1:
                if out_lines:
                    out_lines.append("")
                out_lines.append(f"# {d}")
                out_lines.append("")
            for k in sorted(items.keys()):
                body = (items.get(k) or "").strip()
                out_lines.append(f"## {k}\n{body}".strip() if body else f"## {k}")
                out_lines.append("")
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()
        return "\n".join(out_lines).strip()
    return memory_session_search(ds, keyword=kw)


@tool
def memory_session_search(date: str, keyword: str = "") -> str:
    """Search session memories by keyword in the given date selector.

    date selector:
    - YYYY-MM-DD (single day)
    - today|now (today)
    - all|* (all days)
    - YYYY-MM (month)
    - YYYY-MM-DD..YYYY-MM-DD (inclusive range)
    """
    project_root = _memory_project_root()
    paths, err = _session_memory_md_paths(project_root, date)
    if err:
        return err
    q = (keyword or "").strip()

    def _score_section(*, section_key: str, section_body: str, tokens: list[str]) -> tuple[int, int, int]:
        k = (section_key or "").lower()
        b = (section_body or "").lower()
        exact = 0
        key_hits = 0
        body_hits = 0
        if tokens and len(tokens) == 1 and tokens[0] and tokens[0] == k.strip():
            exact = 1
        for t in tokens:
            if not t:
                continue
            if t in k:
                key_hits += 1
            if t in b:
                body_hits += 1
        return exact, key_hits, body_hits

    if not q:
        if len(paths) == 1:
            ds, p = paths[0]
            if not p.exists() or not p.is_file():
                return ""
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                return f"Read failed: {e}"
            items = _parse_session_md(text)
            keys = sorted(items.keys())
            _log_action(
                {
                    "kind": "memory_session_search",
                    "ok": True,
                    "path": p.as_posix(),
                    "date": ds,
                    "keyword": "",
                    "hits": len(keys),
                    "ts": time.time(),
                }
            )
            return "\n".join(keys)
        _log_action(
            {
                "kind": "memory_session_search",
                "ok": True,
                "path": "",
                "date": date,
                "keyword": "",
                "hits": 0,
                "ts": time.time(),
            }
        )
        return ""

    tokens = [t.lower() for t in re.split(r"\s+", q) if t.strip()]
    rollup = _infer_rollup_target_from_question("", date)
    if rollup is not None:
        layer, key = rollup
        p2, err2 = _rollup_md_path(project_root, layer=layer, key=key)
        if err2:
            return err2
        assert p2 is not None
        if p2.exists() and p2.is_file():
            try:
                text2 = p2.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text2 = ""
            if text2:
                filtered2 = _filter_text(text2, tokens)
                if filtered2:
                    _log_action(
                        {
                            "kind": "memory_session_search",
                            "ok": True,
                            "path": p2.as_posix(),
                            "date": date,
                            "keyword": q,
                            "hits": -1,
                            "ts": time.time(),
                        }
                    )
                    return filtered2
    max_hits = 30
    scored: list[tuple[int, str, str, str]] = []
    for ds, p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        items = _parse_session_md(text)
        for k, v in items.items():
            exact, key_hits, body_hits = _score_section(section_key=k, section_body=v, tokens=tokens)
            if exact == 0 and key_hits == 0 and body_hits == 0:
                continue
            score = 0
            score += 1000 if exact else 0
            score += key_hits * 100
            score += body_hits * 40
            scored.append((score, ds, k, v))

    if not scored:
        _log_action(
            {
                "kind": "memory_session_search",
                "ok": True,
                "path": "",
                "date": date,
                "keyword": q,
                "hits": 0,
                "ts": time.time(),
            }
        )
        return ""

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    scored = scored[:max_hits]

    def _filter_body(body: str, tokens: list[str]) -> str:
        lines = (body or "").splitlines()
        kept: list[str] = []
        for line in lines:
            ll = line.lower()
            if any(t in ll for t in tokens if t):
                kept.append(line)
        out = "\n".join([ln for ln in kept if ln.strip()]).strip()
        return out if out else (body or "").strip()

    out_lines: list[str] = []
    last_date: str | None = None
    for _, ds, k, v in scored:
        if len(paths) != 1 and ds != last_date:
            if out_lines:
                out_lines.append("")
            out_lines.append(f"# {ds}")
            out_lines.append("")
            last_date = ds
        body = _filter_body(v, tokens)
        out_lines.append(f"## {k}\n{body}".strip())
        out_lines.append("")

    while out_lines and not out_lines[-1].strip():
        out_lines.pop()
    _log_action(
        {
            "kind": "memory_session_search",
            "ok": True,
            "path": "",
            "date": date,
            "keyword": q,
            "hits": len(scored),
            "ts": time.time(),
        }
    )
    return "\n".join(out_lines).strip()


def _runtime_thread_id(runtime: ToolRuntime) -> str:
    cfg = getattr(runtime, "config", None) or {}
    if isinstance(cfg, dict):
        configurable = cfg.get("configurable") or {}
        if isinstance(configurable, dict):
            tid = (configurable.get("thread_id") or "").strip()
            if tid:
                return tid
    return (os.environ.get("AGENT_THREAD_ID") or "").strip() or "default"


def _parse_namespace(namespace: str, *, default: tuple[str, ...] = ("default",)) -> tuple[str, ...]:
    raw = (namespace or "").strip()
    if not raw:
        return default
    parts = [p.strip() for p in raw.replace("\\", "/").split("/") if p.strip()]
    return tuple(parts) if parts else default


def _parse_run_ts(run_at: str) -> float | None:
    s = (run_at or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    s = s.replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    try:
        return dt.timestamp()
    except Exception:
        return None


@tool
def reminder_schedule_at(message: str, run_at: str, reminder_id: str = "") -> str:
    """Schedule a one-shot reminder at a specific time (ISO string or unix seconds)."""
    from system.manager import get_global_system_manager

    mgr = get_global_system_manager()
    if mgr is None:
        return "SystemManager is not available."
    ts = _parse_run_ts(run_at)
    if ts is None:
        return "Invalid run_at. Use ISO datetime (e.g. 2026-02-10T18:00:00) or unix seconds."
    try:
        rid = mgr.reminder_create_at(run_ts=ts, message=message, reminder_id=reminder_id)
    except Exception as e:
        return f"Failed: {type(e).__name__}: {e}"
    return json.dumps({"ok": True, "id": rid, "run_ts": ts}, ensure_ascii=False, sort_keys=True)


@tool
def reminder_schedule_in(message: str, delay_s: int = 60, reminder_id: str = "") -> str:
    """Schedule a one-shot reminder after delay seconds."""
    from system.manager import get_global_system_manager

    mgr = get_global_system_manager()
    if mgr is None:
        return "SystemManager is not available."
    try:
        rid = mgr.reminder_create_in(delay_s=float(delay_s), message=message, reminder_id=reminder_id)
    except Exception as e:
        return f"Failed: {type(e).__name__}: {e}"
    return json.dumps({"ok": True, "id": rid, "delay_s": int(delay_s)}, ensure_ascii=False, sort_keys=True)


@tool
def reminder_cancel(reminder_id: str) -> str:
    """Cancel a scheduled reminder by id."""
    from system.manager import get_global_system_manager

    mgr = get_global_system_manager()
    if mgr is None:
        return "SystemManager is not available."
    ok = bool(mgr.reminder_cancel(reminder_id))
    return json.dumps({"ok": ok, "id": (reminder_id or "").strip()}, ensure_ascii=False, sort_keys=True)


@tool
def reminder_list(status: str = "") -> str:
    """List reminders recorded in the system."""
    from system.manager import get_global_system_manager

    mgr = get_global_system_manager()
    if mgr is None:
        return "SystemManager is not available."
    items = mgr.reminder_list()
    st = (status or "").strip().lower()
    if st:
        items = [it for it in items if str(it.get("status") or "").lower() == st]
    return json.dumps(items, ensure_ascii=False, sort_keys=True)


@tool
def system_time() -> str:
    """Get current system time with local timezone and UTC."""
    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(tz=timezone.utc)
    offset = now_local.utcoffset()
    offset_s = int(offset.total_seconds()) if offset is not None else 0
    data = {
        "unix_ts": time.time(),
        "local": {
            "iso": now_local.replace(microsecond=0).isoformat(),
            "tzname": str(now_local.tzname() or ""),
            "utc_offset_seconds": offset_s,
        },
        "utc": {"iso": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")},
    }
    try:
        from lunar_python import Solar

        solar = Solar.fromYmdHms(
            now_local.year,
            now_local.month,
            now_local.day,
            now_local.hour,
            now_local.minute,
            now_local.second,
        )
        lunar = solar.getLunar()
        data["lunar"] = {
            "full": lunar.toFullString(),
            "date": lunar.toString(),
            "solar_full": solar.toFullString(),
        }
    except Exception as e:
        data["lunar"] = {
            "available": False,
            "error": f"{type(e).__name__}: {e}",
            "install": "pip install lunar_python",
        }
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
