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
def memory_kg_recall(query: str, limit: int = 12) -> str:
    """Search the pageindex built from chat logs."""
    try:
        from memory.paths import pageindex_chats_dir
        from memory.query import search_pageindex
        from memory.storage import PageIndexStore
    except Exception as e:
        return f"Import failed: {e}"
    q = (query or "").strip()
    if not q:
        return ""
    if limit <= 0:
        limit = 12
    limit = max(1, min(50, int(limit)))
    store = PageIndexStore(root_dir=pageindex_chats_dir(_memory_project_root()))
    return search_pageindex(store, q, limit=limit)


@tool
def memory_kg_stats() -> str:
    """Show pageindex size info."""
    try:
        from memory.paths import pageindex_chats_dir
        from memory.query import pageindex_stats
        from memory.storage import PageIndexStore
    except Exception as e:
        return f"Import failed: {e}"
    store = PageIndexStore(root_dir=pageindex_chats_dir(_memory_project_root()))
    return pageindex_stats(store)


@tool
def memory_pageindex_ingest(path: str, namespace: str = "default", key: str = "") -> str:
    """Ingest a local PDF/Markdown file into PageIndex JSON store."""
    try:
        from memory.paths import pageindex_docs_dir
        from memory.storage import PageIndexStore
        from memory.pageindex.page_index_md import md_to_tree
    except Exception as e:
        return f"Import failed: {e}"
    p = Path((path or "").strip()).resolve()
    if not p.exists() or not p.is_file():
        return "File not found."
    today = time.strftime("%Y-%m-%d", time.localtime())
    uploads_raw = (os.environ.get("AGENT_UPLOADS_DIR") or "").strip() or "memory/uploads"
    uploads_base = Path(uploads_raw).expanduser()
    if not uploads_base.is_absolute():
        uploads_base = (_memory_project_root() / uploads_base).resolve()
    else:
        uploads_base = uploads_base.resolve()
    uploads_root = (uploads_base / today).resolve()
    uploads_root.mkdir(parents=True, exist_ok=True)
    dst = (uploads_root / p.name).resolve()
    try:
        if dst != p:
            dst.write_bytes(p.read_bytes())
    except Exception:
        dst = p

    ns = _parse_namespace(namespace, default=("default", today))
    k = (key or "").strip() or dst.stem
    store = PageIndexStore(root_dir=pageindex_docs_dir(_memory_project_root()))
    suffix = dst.suffix.lower()
    now = time.time()

    defer = (os.environ.get("AGENT_PAGEINDEX_DAILY_ENABLE") or "").strip().lower() in {"1", "true", "yes", "on"}
    realtime = (os.environ.get("AGENT_PAGEINDEX_REALTIME") or "").strip().lower() in {"1", "true", "yes", "on"}
    if defer and not realtime:
        return f"Saved: {dst.as_posix()}"

    if suffix == ".pdf":
        try:
            from memory.pageindex.page_index import page_index
        except Exception as e:
            return f"Import failed: {e}"
        try:
            doc = page_index(dst.as_posix())
        except Exception as e:
            return f"Ingest failed: {e}"
        if not isinstance(doc, dict):
            doc = {"doc_name": dst.stem, "structure": []}
    elif suffix in {".md", ".markdown"}:
        try:
            try:
                doc = asyncio.run(
                    md_to_tree(
                        md_path=dst.as_posix(),
                        if_thinning=False,
                        if_add_node_summary="yes",
                        if_add_node_text="yes",
                        model=None,
                    )
                )
            except Exception:
                doc = asyncio.run(
                    md_to_tree(
                        md_path=dst.as_posix(),
                        if_thinning=False,
                        if_add_node_summary="no",
                        if_add_node_text="yes",
                        model=None,
                    )
                )
        except Exception as e:
            return f"Ingest failed: {e}"
        if not isinstance(doc, dict):
            doc = {"doc_name": dst.stem, "structure": []}
    else:
        return "Unsupported file type. Use .pdf or .md."
    meta = doc.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)))
    meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    meta["source_path"] = dst.as_posix()
    meta["source_type"] = "pdf" if suffix == ".pdf" else "markdown"
    doc["meta"] = meta
    store.put(ns, k, doc)
    return "OK"


@tool
def memory_pageindex_search(query: str, namespace_prefix: str = "", limit: int = 12) -> str:
    """Search PageIndex docs store (PDF/Markdown ingested documents)."""
    try:
        from memory.paths import pageindex_docs_dir
        from memory.query import search_pageindex
        from memory.storage import PageIndexStore
    except Exception as e:
        return f"Import failed: {e}"
    q = (query or "").strip()
    if not q:
        return ""
    lim = max(1, min(50, int(limit or 12)))
    pre = _parse_namespace(namespace_prefix, default=())
    store = PageIndexStore(root_dir=pageindex_docs_dir(_memory_project_root()))
    return search_pageindex(store, q, limit=lim, namespace_prefix=pre if pre else None)


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


_LTM_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_LTM_STOPWORDS = {
    "昨天",
    "今天",
    "前天",
    "回忆",
    "回想",
    "记得",
    "内容",
    "事情",
    "一下",
    "帮我",
    "请",
    "需要",
    "想",
    "问题",
    "怎么",
    "如何",
    "什么",
    "以及",
    "然后",
    "进行",
    "先",
    "再",
    "最后",
    "定位",
    "功能",
    "实现",
    "优化",
    "长期",
    "记忆",
    "智能体",
}


def _utc_iso(ts: float | None = None) -> str:
    t = time.gmtime(ts if ts is not None else time.time())
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", t)


def _local_date_str(ts: float | None = None) -> str:
    t = time.localtime(ts if ts is not None else time.time())
    return time.strftime("%Y-%m-%d", t)


def _extract_keywords(text: str, *, limit: int = 12) -> list[str]:
    raw = (text or "").strip().lower()
    if not raw:
        return []
    toks: list[str] = []
    for m in _LTM_TOKEN_RE.finditer(raw):
        tok = (m.group(0) or "").strip()
        if not tok:
            continue
        if tok in _LTM_STOPWORDS:
            continue
        if tok.isascii():
            if len(tok) < 3:
                continue
        else:
            if len(tok) < 2:
                continue
        toks.append(tok)
        if len(toks) >= max(1, min(50, int(limit or 12))):
            break
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _merge_ltm_meta(
    existing: dict[str, object] | None,
    *,
    now_ts: float,
    thread_id: str,
    keywords: list[str],
) -> dict[str, object]:
    meta = existing if isinstance(existing, dict) else {}
    created_at = meta.get("created_at")
    created_ts = meta.get("created_ts")
    created_date = meta.get("created_date")
    if not isinstance(created_at, str) or not created_at:
        meta["created_at"] = _utc_iso(now_ts)
    if not isinstance(created_ts, (int, float)):
        meta["created_ts"] = float(now_ts)
    if not isinstance(created_date, str) or not created_date:
        meta["created_date"] = _local_date_str(now_ts)
    meta["updated_at"] = _utc_iso(now_ts)
    meta["updated_ts"] = float(now_ts)
    meta["updated_date"] = _local_date_str(now_ts)
    if thread_id:
        meta["thread_id"] = thread_id

    merged_keywords: list[str] = []
    existing_kw = meta.get("keywords")
    if isinstance(existing_kw, list):
        for k in existing_kw:
            if isinstance(k, str) and k.strip():
                merged_keywords.append(k.strip().lower())
    for k in keywords:
        if isinstance(k, str) and k.strip():
            merged_keywords.append(k.strip().lower())
    seen: set[str] = set()
    out_kw: list[str] = []
    for k in merged_keywords:
        if k in seen:
            continue
        seen.add(k)
        out_kw.append(k)
        if len(out_kw) >= 50:
            break
    if out_kw:
        meta["keywords"] = out_kw
    return meta


def _parse_date_str(s: str) -> date | None:
    raw = (s or "").strip()
    if not raw:
        return None
    raw = raw.replace("/", "-").replace(".", "-")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日?", raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日?(?!\d)", raw)
    if m:
        y = date.today().year
        try:
            return date(int(y), int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    return None


def _infer_date_window(query: str, filt: dict[str, object] | None) -> tuple[date | None, date | None]:
    f = filt if isinstance(filt, dict) else {}
    if isinstance(f.get("date"), str):
        d = _parse_date_str(f.get("date") or "")
        if d:
            return d, d
    if isinstance(f.get("start_date"), str) or isinstance(f.get("end_date"), str):
        sd = _parse_date_str(f.get("start_date") or "")
        ed = _parse_date_str(f.get("end_date") or "")
        return sd, ed
    days_ago = f.get("days_ago")
    if isinstance(days_ago, (int, float)):
        d = date.today() - timedelta(days=int(days_ago))
        return d, d

    q = (query or "").strip()
    if not q:
        return None, None
    if "昨天" in q:
        d = date.today() - timedelta(days=1)
        return d, d
    if "前天" in q:
        d = date.today() - timedelta(days=2)
        return d, d
    if "今天" in q:
        d = date.today()
        return d, d
    d = _parse_date_str(q)
    if d:
        return d, d
    if "上周" in q or "最近7天" in q or "近7天" in q:
        end = date.today()
        start = end - timedelta(days=7)
        return start, end
    return None, None


def _keywords_from_filter_or_query(query: str, filt: dict[str, object] | None) -> list[str]:
    f = filt if isinstance(filt, dict) else {}
    kws: list[str] = []
    fk = f.get("keywords") or f.get("keyword") or f.get("tags") or f.get("tag")
    if isinstance(fk, str) and fk.strip():
        kws.extend([x.strip().lower() for x in re.split(r"[,\s]+", fk.strip()) if x.strip()])
    elif isinstance(fk, list):
        for x in fk:
            if isinstance(x, str) and x.strip():
                kws.append(x.strip().lower())
    if not kws:
        kws = _extract_keywords(query or "", limit=6)
    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        kk = (k or "").strip().lower()
        if not kk or kk in _LTM_STOPWORDS:
            continue
        if kk in seen:
            continue
        seen.add(kk)
        out.append(kk)
    return out[:12]


def _safe_str_excerpt(s: str, *, limit: int = 220) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    n = max(20, min(2000, int(limit or 220)))
    if len(t) <= n:
        return t
    return t[:n].rstrip() + "…"


def _ltm_snapshot_items(store: object, ns: tuple[str, ...]) -> list[dict[str, object]]:
    snap_fn = getattr(store, "snapshot_items", None)
    if callable(snap_fn):
        try:
            items = snap_fn(ns)
            if isinstance(items, list):
                return [x for x in items if isinstance(x, dict)]
        except Exception:
            pass
    return []


def _ltm_item_date(item: dict[str, object]) -> date | None:
    val = item.get("value")
    if isinstance(val, dict):
        meta = val.get("meta")
        if not isinstance(meta, dict):
            meta = val.get("_meta") if isinstance(val.get("_meta"), dict) else {}
        cd = meta.get("created_date")
        if isinstance(cd, str):
            d = _parse_date_str(cd)
            if d:
                return d
    ca = item.get("created_at")
    if isinstance(ca, str) and ca:
        try:
            return date.fromisoformat(ca[:10])
        except ValueError:
            return None
    return None


def _ltm_item_keywords(item: dict[str, object]) -> list[str]:
    val = item.get("value")
    if not isinstance(val, dict):
        return []
    meta = val.get("meta")
    if not isinstance(meta, dict):
        meta = val.get("_meta") if isinstance(val.get("_meta"), dict) else {}
    kws: list[str] = []
    if isinstance(meta, dict):
        mk = meta.get("keywords")
        if isinstance(mk, list):
            for x in mk:
                if isinstance(x, str) and x.strip():
                    kws.append(x.strip().lower())
    return kws


def _ltm_item_text(item: dict[str, object]) -> str:
    val = item.get("value")
    if not isinstance(val, dict):
        return ""
    t = val.get("text")
    return t if isinstance(t, str) else ""


def _ltm_match_keywords(*, keywords: list[str], item_keywords: list[str], item_text: str) -> tuple[bool, int]:
    if not keywords:
        return True, 0
    hits = 0
    text = (item_text or "").lower()
    kw_set = set(item_keywords or [])
    for k in keywords:
        if not k:
            continue
        ok = False
        if k in kw_set:
            ok = True
            hits += 2
        if not ok and k in text:
            ok = True
            hits += 1
        if not ok:
            return False, hits
    return True, hits


@tool
def memory_ltm_put(namespace: str, key: str, value: dict[str, object], runtime: ToolRuntime) -> str:
    """Store a long-term memory item as a PageIndex JSON document under (namespace, key)."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
        from memory.pageindex.page_index_md import (
            build_tree_from_nodes,
            extract_node_text_content,
            extract_nodes_from_markdown,
            generate_summaries_for_structure_md,
        )
        from memory.pageindex.utils import ConfigLoader, format_structure, write_node_id
    except Exception as e:
        return f"Import failed: {e}"

    ns = _parse_namespace(namespace, default=("mem",))
    k = (key or "").strip()
    if not k:
        return "Missing key."
    v = value if isinstance(value, dict) else {}

    raw_text = v.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        try:
            raw_text = json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            raw_text = str(v)
    md = f"# {k}\n\n{raw_text.strip()}\n"

    node_list, markdown_lines = extract_nodes_from_markdown(md)
    nodes_with_content = extract_node_text_content(node_list, markdown_lines)
    tree_structure = build_tree_from_nodes(nodes_with_content)

    opt = ConfigLoader().load()
    if str(getattr(opt, "if_add_node_id", "yes")) == "yes":
        write_node_id(tree_structure)

    tree_structure = format_structure(
        tree_structure, order=["title", "node_id", "summary", "prefix_summary", "text", "line_num", "nodes"]
    )
    if str(getattr(opt, "if_add_node_summary", "yes")) == "yes":
        summary_token_threshold = 200
        try:
            tree_structure = asyncio.run(
                generate_summaries_for_structure_md(
                    tree_structure,
                    summary_token_threshold=summary_token_threshold,
                    model=str(getattr(opt, "model", "")),
                )
            )
        except Exception:
            pass

    now_ts = time.time()
    keywords: list[str] = []
    user_keywords = v.get("keywords")
    if isinstance(user_keywords, list):
        for kw in user_keywords:
            if isinstance(kw, str) and kw.strip():
                keywords.append(kw.strip().lower())
    keywords.extend(_extract_keywords(raw_text))

    thread_id = _runtime_thread_id(runtime)
    meta_in = v.get("meta")
    if not isinstance(meta_in, dict):
        meta_in = v.get("_meta") if isinstance(v.get("_meta"), dict) else {}
    meta = _merge_ltm_meta(meta_in, now_ts=now_ts, thread_id=thread_id, keywords=keywords)
    meta["namespace"] = "/".join(ns)
    meta["key"] = k

    doc: dict[str, object] = {"doc_name": k, "structure": tree_structure, "meta": meta}
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    store.put(ns, k, doc)
    return "OK"


@tool
def memory_ltm_get(namespace: str, key: str, runtime: ToolRuntime) -> str:
    """Get a long-term memory PageIndex JSON document by (namespace, key)."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
    except Exception as e:
        return f"Import failed: {e}"
    ns = _parse_namespace(namespace, default=("mem",))
    k = (key or "").strip()
    if not k:
        return ""
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    val = store.get(ns, k)
    return json.dumps(val, ensure_ascii=False, sort_keys=True) if isinstance(val, dict) else ""


@tool
def memory_ltm_delete(namespace: str, key: str, runtime: ToolRuntime) -> str:
    """Delete a long-term memory item by (namespace, key)."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
    except Exception:
        return "OK"
    ns = _parse_namespace(namespace, default=("mem",))
    k = (key or "").strip()
    if not k:
        return "Missing key."
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    store.delete(ns, k)
    return "OK"


@tool
def memory_ltm_search(
    namespace: str,
    runtime: ToolRuntime,
    query: str = "",
    filter_json: str = "",
    limit: int = 12,
) -> str:
    """Search long-term memory PageIndex docs within a namespace."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
        from memory.query import _tokens as _pi_tokens
    except Exception as e:
        return f"Import failed: {e}"
    ns = _parse_namespace(namespace, default=("mem",))
    q = (query or "").strip() or None
    filt: dict[str, object] | None = None
    raw_filter = (filter_json or "").strip()
    if raw_filter:
        try:
            parsed = json.loads(raw_filter)
            if isinstance(parsed, dict):
                filt = parsed
        except Exception:
            filt = None
    n = max(1, min(50, int(limit or 12)))
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    start_d, end_d = _infer_date_window(q or "", filt)
    keywords = _keywords_from_filter_or_query(q or "", filt)
    toks = _pi_tokens(q or "") if q else []

    def walk_nodes(nodes: object) -> list[dict[str, object]]:
        out_nodes: list[dict[str, object]] = []
        if isinstance(nodes, dict):
            out_nodes.append(nodes)
            ch = nodes.get("nodes")
            if isinstance(ch, list):
                for c in ch:
                    out_nodes.extend(walk_nodes(c))
        elif isinstance(nodes, list):
            for n0 in nodes:
                out_nodes.extend(walk_nodes(n0))
        return out_nodes

    candidates: list[tuple[float, str, dict[str, object]]] = []
    for key0, _ in store.iter_docs(ns):
        doc = store.get(ns, key0)
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta")
        meta_dict = meta if isinstance(meta, dict) else {}
        d = meta_dict.get("created_date")
        if isinstance(d, str) and d:
            try:
                d_int = int(d.replace("-", ""))
            except Exception:
                d_int = None
        else:
            d_int = None
        if start_d is not None or end_d is not None:
            if d_int is None:
                continue
            if start_d is not None and d_int < start_d:
                continue
            if end_d is not None and d_int > end_d:
                continue
        item_kws = meta_dict.get("keywords") if isinstance(meta_dict.get("keywords"), list) else []
        item_kws_l = [x for x in item_kws if isinstance(x, str)]
        item_text = json.dumps(meta_dict, ensure_ascii=False, sort_keys=True)
        ok, hit_score = _ltm_match_keywords(keywords=keywords, item_keywords=item_kws_l, item_text=item_text)
        if not ok and keywords:
            continue
        score = float(hit_score)
        if toks:
            structure = doc.get("structure")
            for node in walk_nodes(structure):
                text_parts: list[str] = []
                for kk in ("title", "summary", "text"):
                    vv = node.get(kk)
                    if isinstance(vv, str) and vv:
                        text_parts.append(vv)
                blob = " ".join(text_parts).lower()
                for t in toks:
                    if t and t in blob:
                        score += 1.0
        updated_ts = meta_dict.get("updated_ts")
        if isinstance(updated_ts, (int, float)):
            score += float(updated_ts) / 1_000_000_000.0
        candidates.append((score, key0, doc))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, object]] = []
    for score, key0, doc in candidates[:n]:
        out.append({"namespace": list(ns), "key": key0, "score": score, "value": doc})
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


@tool
def memory_ltm_search_index(
    namespace: str,
    runtime: ToolRuntime,
    query: str = "",
    filter_json: str = "",
    limit: int = 12,
) -> str:
    """Search long-term memory by time and metadata first; returns metadata without node content."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
    except Exception as e:
        return f"Import failed: {e}"
    ns = _parse_namespace(namespace, default=("mem",))
    q = (query or "").strip()
    filt: dict[str, object] | None = None
    raw_filter = (filter_json or "").strip()
    if raw_filter:
        try:
            parsed = json.loads(raw_filter)
            if isinstance(parsed, dict):
                filt = parsed
        except Exception:
            filt = None
    n = max(1, min(50, int(limit or 12)))
    start_d, end_d = _infer_date_window(q, filt)
    keywords = _keywords_from_filter_or_query(q, filt)
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    candidates: list[tuple[float, str, dict[str, object]]] = []
    for key0, _ in store.iter_docs(ns):
        doc = store.get(ns, key0)
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta")
        meta_dict = meta if isinstance(meta, dict) else {}
        d = meta_dict.get("created_date")
        if isinstance(d, str) and d:
            try:
                d_int = int(d.replace("-", ""))
            except Exception:
                d_int = None
        else:
            d_int = None
        if start_d is not None or end_d is not None:
            if d_int is None:
                continue
            if start_d is not None and d_int < start_d:
                continue
            if end_d is not None and d_int > end_d:
                continue
        item_kws = meta_dict.get("keywords") if isinstance(meta_dict.get("keywords"), list) else []
        item_kws_l = [x for x in item_kws if isinstance(x, str)]
        item_text = json.dumps(meta_dict, ensure_ascii=False, sort_keys=True)
        ok, hit_score = _ltm_match_keywords(keywords=keywords, item_keywords=item_kws_l, item_text=item_text)
        if not ok and keywords:
            continue
        updated_ts = meta_dict.get("updated_ts")
        score = float(hit_score)
        if isinstance(updated_ts, (int, float)):
            score += float(updated_ts) / 1_000_000_000.0
        out_meta = dict(meta_dict)
        out_meta.pop("structure", None)
        candidates.append((score, key0, out_meta))
    candidates.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, object]] = []
    for score, key0, meta_dict in candidates[:n]:
        out.append({"namespace": list(ns), "key": key0, "score": score, "meta": meta_dict})
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


@tool
def memory_ltm_list_namespaces(runtime: ToolRuntime, prefix: str = "") -> str:
    """List available namespaces in the long-term memory store."""
    try:
        from memory.paths import pageindex_ltm_dir
        from memory.storage import PageIndexStore
    except Exception:
        return "[]"
    store = PageIndexStore(root_dir=pageindex_ltm_dir(_memory_project_root()))
    p = _parse_namespace(prefix, default=())
    namespaces = store.list_namespaces(prefix=p if p else None, limit=200)
    out: list[list[str]] = []
    if isinstance(namespaces, list):
        for ns in namespaces:
            if isinstance(ns, tuple):
                out.append(list(ns))
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


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
    return json.dumps(data, ensure_ascii=False, sort_keys=True)
