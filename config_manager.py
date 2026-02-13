from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV_REF_RX = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_refs(value: str) -> str:
    s = value or ""

    def _repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV_REF_RX.sub(_repl, s)


def _as_str_dict(v: object) -> dict[str, str]:
    if not isinstance(v, dict):
        return {}
    out: dict[str, str] = {}
    for k, val in v.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if val is None:
            continue
        if not isinstance(val, str):
            val = str(val)
        out[k.strip()] = _expand_env_refs(val)
    return out


def load_config(path: Path) -> dict[str, Any] | None:
    try:
        p = path.expanduser().resolve()
    except Exception:
        p = path
    if not p.exists() or not p.is_file():
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    raw = (raw or "").lstrip("\ufeff").strip()
    if not raw:
        return None
    if not raw.lstrip().startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _active_model_key(cfg: dict[str, Any]) -> str:
    key = cfg.get("active_model")
    if isinstance(key, str) and key.strip():
        return key.strip()
    key2 = cfg.get("model")
    if isinstance(key2, str) and key2.strip():
        return key2.strip()
    return ""


def _active_workspace(cfg: dict[str, Any]) -> str:
    out = cfg.get("output")
    if isinstance(out, dict):
        ws = out.get("workspace")
        if isinstance(ws, str) and ws.strip():
            return ws.strip()
    ws2 = cfg.get("active_workspace")
    if isinstance(ws2, str) and ws2.strip():
        return ws2.strip()
    return ""


def _get_models(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = cfg.get("models")
    if not isinstance(models, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in models.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, dict):
            continue
        out[k.strip()] = v
    return out


def apply_config_to_environ(cfg: dict[str, Any], *, override: bool) -> dict[str, str]:
    applied: dict[str, str] = {}

    env = _as_str_dict(cfg.get("env"))
    for k, v in env.items():
        if override:
            os.environ[k] = v
        else:
            os.environ.setdefault(k, v)
        applied[k] = v

    permissions = cfg.get("permissions")
    if isinstance(permissions, dict):
        write_roots = permissions.get("write_extra_roots")
        if isinstance(write_roots, list) and write_roots:
            joined = ";".join([str(x) for x in write_roots if str(x).strip()])
            if joined:
                if override:
                    os.environ["AGENT_EXTRA_WRITE_ROOTS"] = joined
                else:
                    os.environ.setdefault("AGENT_EXTRA_WRITE_ROOTS", joined)
                applied["AGENT_EXTRA_WRITE_ROOTS"] = joined

        cwd_roots = permissions.get("cli_extra_roots")
        if isinstance(cwd_roots, list) and cwd_roots:
            joined = ";".join([str(x) for x in cwd_roots if str(x).strip()])
            if joined:
                if override:
                    os.environ["AGENT_EXTRA_CWD_ROOTS"] = joined
                else:
                    os.environ.setdefault("AGENT_EXTRA_CWD_ROOTS", joined)
                applied["AGENT_EXTRA_CWD_ROOTS"] = joined

        sandbox = permissions.get("sandbox")
        if isinstance(sandbox, str) and sandbox.strip():
            if override:
                os.environ["AGENT_SANDBOX"] = sandbox.strip()
            else:
                os.environ.setdefault("AGENT_SANDBOX", sandbox.strip())
            applied["AGENT_SANDBOX"] = sandbox.strip()

    ws = _active_workspace(cfg)
    if ws:
        if override:
            os.environ["AGENT_OUTPUT_WORKSPACE"] = ws
        else:
            os.environ.setdefault("AGENT_OUTPUT_WORKSPACE", ws)
        applied["AGENT_OUTPUT_WORKSPACE"] = ws

    models = _get_models(cfg)
    active_key = _active_model_key(cfg)
    if active_key and active_key in models:
        spec = models.get(active_key) or {}
        model_name = spec.get("model")
        if isinstance(model_name, str) and model_name.strip():
            if override:
                os.environ["LC_MODEL"] = model_name.strip()
            else:
                os.environ.setdefault("LC_MODEL", model_name.strip())
            applied["LC_MODEL"] = model_name.strip()
        spec_env = _as_str_dict(spec.get("env"))
        for k, v in spec_env.items():
            if override:
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)
            applied[k] = v

    return applied


def list_models(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in sorted(_get_models(cfg).items(), key=lambda x: x[0].lower()):
        model_name = v.get("model")
        if isinstance(model_name, str) and model_name.strip():
            out.append((k, model_name.strip()))
        else:
            out.append((k, ""))
    return out


def update_config_file(path: Path, *, mutator) -> dict[str, Any]:
    p = path.expanduser().resolve()
    cfg = load_config(p) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg2 = mutator(cfg) or cfg
    if not isinstance(cfg2, dict):
        cfg2 = cfg
    text = json.dumps(cfg2, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return cfg2


def set_active_model(path: Path, key: str) -> dict[str, Any]:
    k = (key or "").strip()
    if not k:
        return load_config(path) or {}

    def _m(cfg: dict[str, Any]) -> dict[str, Any]:
        models = _get_models(cfg)
        if k not in models:
            return cfg
        cfg["active_model"] = k
        cfg.pop("model", None)
        return cfg

    return update_config_file(path, mutator=_m)


def set_active_workspace(path: Path, workspace: str) -> dict[str, Any]:
    ws = (workspace or "").strip()
    if not ws:
        return load_config(path) or {}

    def _m(cfg: dict[str, Any]) -> dict[str, Any]:
        out = cfg.get("output")
        if not isinstance(out, dict):
            out = {}
            cfg["output"] = out
        out["workspace"] = ws
        cfg.pop("active_workspace", None)
        return cfg

    return update_config_file(path, mutator=_m)
