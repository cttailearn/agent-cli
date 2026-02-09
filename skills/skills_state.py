from __future__ import annotations

import json
import os
import time
from pathlib import Path


CORE_SKILLS: frozenset[str] = frozenset({"find-skills", "skill-creator"})


def _project_root() -> Path:
    root = os.environ.get("AGENT_PROJECT_DIR")
    if root:
        return Path(root).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def _state_path() -> Path:
    return (_project_root() / ".agents" / "skills_state.json").resolve()


def state_path() -> Path:
    return _state_path()


def load_state() -> dict[str, object]:
    p = _state_path()
    if not p.exists():
        return {"disabled": {}, "usage": {}, "installed": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace") or "{}")
    except Exception:
        return {"disabled": {}, "usage": {}, "installed": {}}
    if not isinstance(raw, dict):
        return {"disabled": {}, "usage": {}, "installed": {}}
    raw.setdefault("disabled", {})
    raw.setdefault("usage", {})
    raw.setdefault("installed", {})
    if not isinstance(raw.get("disabled"), dict):
        raw["disabled"] = {}
    if not isinstance(raw.get("usage"), dict):
        raw["usage"] = {}
    if not isinstance(raw.get("installed"), dict):
        raw["installed"] = {}
    return raw


def save_state(state: dict[str, object]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def is_disabled(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if n in CORE_SKILLS:
        return False
    state = load_state()
    disabled = state.get("disabled")
    return isinstance(disabled, dict) and n in disabled


def disable_skill(name: str, reason: str = "") -> None:
    n = (name or "").strip()
    if not n:
        return
    if n in CORE_SKILLS:
        return
    state = load_state()
    disabled = state.get("disabled")
    if not isinstance(disabled, dict):
        disabled = {}
        state["disabled"] = disabled
    disabled[n] = {"reason": (reason or "").strip(), "ts": time.time()}
    save_state(state)


def enable_skill(name: str) -> None:
    n = (name or "").strip()
    if not n:
        return
    state = load_state()
    disabled = state.get("disabled")
    if isinstance(disabled, dict) and n in disabled:
        disabled.pop(n, None)
        save_state(state)


def record_skill_loaded(name: str) -> None:
    n = (name or "").strip()
    if not n:
        return
    state = load_state()
    usage = state.get("usage")
    if not isinstance(usage, dict):
        usage = {}
        state["usage"] = usage
    entry = usage.get(n)
    if not isinstance(entry, dict):
        entry = {}
        usage[n] = entry
    load_count = entry.get("load_count")
    entry["load_count"] = int(load_count) + 1 if isinstance(load_count, int) else 1
    entry["last_loaded_ts"] = time.time()
    save_state(state)


def record_installed(name: str, source: str) -> None:
    n = (name or "").strip()
    if not n:
        return
    state = load_state()
    installed = state.get("installed")
    if not isinstance(installed, dict):
        installed = {}
        state["installed"] = installed
    installed[n] = {"source": (source or "").strip(), "ts": time.time()}
    save_state(state)
