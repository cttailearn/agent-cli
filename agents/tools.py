import asyncio
import json
import os
import re
import subprocess
import threading
import time
from fnmatch import fnmatch
from langchain.tools import tool
from pathlib import Path


ACTION_LOG: list[dict[str, object]] = []


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


def actions_since(snapshot: int) -> list[dict[str, object]]:
    if snapshot <= 0:
        return ACTION_LOG[:]
    return ACTION_LOG[snapshot:]


def _log_action(entry: dict[str, object]) -> None:
    ACTION_LOG.append(entry)


def _project_root() -> Path:
    project_dir_env = os.environ.get("AGENT_PROJECT_DIR")
    if project_dir_env:
        return Path(project_dir_env).expanduser()
    return Path(__file__).resolve().parent


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

    try:
        if not stream:
            completed = subprocess.run(
                cmd,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                encoding=encoding,
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = completed.returncode
        else:
            print(f"$ {command}", flush=True)
            process = subprocess.Popen(
                cmd,
                cwd=str(resolved_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding=encoding,
                errors="replace",
            )

            output_lines: list[str] = []
            output_lock = threading.Lock()

            def _reader() -> None:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\n")
                    print(line, flush=True)
                    with output_lock:
                        output_lines.append(line)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()

            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    exit_code = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    exit_code = process.wait(timeout=5)

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
                return f"Command timed out after {timeout_s}s."
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
    except subprocess.TimeoutExpired:
        _log_action(
            {
                "kind": "run_cli",
                "ok": False,
                "command": command,
                "cwd": resolved_cwd.as_posix(),
                "error": f"Command timed out after {timeout_s}s.",
                "timeout_s": timeout_s,
                "ts": time.time(),
            }
        )
        return f"Command timed out after {timeout_s}s."
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

        def _thread_runner():
            try:
                result_box["value"] = _run_in_new_loop()
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
def memory_kg_recall(query: str, limit: int = 12) -> str:
    """Search the knowledge graph built from chat logs."""
    try:
        from memory.paths import graph_path
        from memory.query import search_graph
        from memory.storage import KnowledgeGraphStore
    except Exception as e:
        return f"Import failed: {e}"
    q = (query or "").strip()
    if not q:
        return ""
    if limit <= 0:
        limit = 12
    limit = max(1, min(50, int(limit)))
    store = KnowledgeGraphStore(graph_path=graph_path(_memory_project_root()))
    return search_graph(store, q, limit=limit)


@tool
def memory_kg_stats() -> str:
    """Show knowledge graph size info."""
    try:
        from memory.paths import graph_path
        from memory.query import graph_stats
        from memory.storage import KnowledgeGraphStore
    except Exception as e:
        return f"Import failed: {e}"
    store = KnowledgeGraphStore(graph_path=graph_path(_memory_project_root()))
    return graph_stats(store)


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
