from __future__ import annotations

import os
from pathlib import Path


def _resolve_path(project_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def project_root_from_env(default: Path) -> Path:
    raw = (os.environ.get("AGENT_PROJECT_DIR") or "").strip()
    if not raw:
        return default.resolve()
    try:
        return Path(raw).expanduser().resolve()
    except OSError:
        return default.resolve()


def memory_root(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_DIR") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (project_root / "memory").resolve()


def core_dir(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_CORE_DIR") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return memory_root(project_root)


def soul_path(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_SOUL_PATH") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (core_dir(project_root) / "soul.md").resolve()


def traits_path(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_TRAITS_PATH") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (core_dir(project_root) / "traits.md").resolve()


def identity_path(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_IDENTITY_PATH") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (core_dir(project_root) / "identity.md").resolve()


def user_path(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_MEMORY_USER_PATH") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (core_dir(project_root) / "user.md").resolve()


def core_file_by_kind(project_root: Path, kind: str) -> Path | None:
    k = (kind or "").strip().lower()
    if k in {"soul", "灵魂"}:
        return soul_path(project_root)
    if k in {"traits", "trait", "特性"}:
        return traits_path(project_root)
    if k in {"identity", "身份"}:
        return identity_path(project_root)
    if k in {"user", "用户"}:
        return user_path(project_root)
    return None


def langgraph_store_path(project_root: Path) -> Path:
    raw = (os.environ.get("AGENT_LANGGRAPH_STORE_PATH") or "").strip()
    if raw:
        return _resolve_path(project_root, raw)
    return (memory_root(project_root) / "langgraph_store.json").resolve()

