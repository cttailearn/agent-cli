from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import skills_state
from .skills_support import discover_skills_from_dirs


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    skill_dir: Path
    disabled: bool
    load_count: int
    last_loaded_ts: float | None


def _shell_command(command: str) -> list[str]:
    if os.name == "nt":
        return [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    return ["bash", "-lc", command]


def _safe_skill_name(name: str) -> str:
    n = (name or "").strip()
    n = re.sub(r"[^\w.-]+", "-", n).strip("-")
    return n[:80]


def _extract_source_ref(text: str) -> tuple[str | None, str | None]:
    url: str | None = None
    hint: str | None = None
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("◇") or line.startswith("│"):
            m = re.search(r"\bSource:\s*(https?://\S+)(?:\s+@([A-Za-z0-9_.-]+))?", line)
            if m:
                url = (m.group(1) or "").strip() or None
                hint = (m.group(2) or "").strip() or None
                if url:
                    return url, hint
    m = re.search(r"\bSource:\s*(https?://\S+)(?:\s+@([A-Za-z0-9_.-]+))?", text or "")
    if not m:
        return None, None
    url = (m.group(1) or "").strip() or None
    hint = (m.group(2) or "").strip() or None
    return url, hint


def _extract_source_url(text: str) -> str | None:
    url, _ = _extract_source_ref(text)
    return url


def _classify_install_failure(text: str) -> tuple[str, bool]:
    s = (text or "").lower()
    if "invalid agents:" in s:
        return "invalid_agents", False
    if "repository" in s and "does not exist" in s:
        return "repo_not_found", False
    if "failed to clone" in s and ("not found" in s or "does not exist" in s):
        return "repo_not_found", False
    if "clone timed out after" in s or ("failed to clone repository" in s and "timed out" in s):
        return "clone_timeout", True
    if "failed to clone repository" in s:
        return "clone_failed", True
    if "authentication" in s or "private repos" in s or "permission denied" in s:
        return "auth", False
    if "enotfound" in s or "econnreset" in s or "etimedout" in s or "network" in s:
        return "network", True
    if "timeout" in s:
        return "timeout", True
    return "unknown", False


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


def _console_print(*args: object, sep: str = " ", end: str = "\n", flush: bool = False, ensure_newline: bool = False) -> None:
    try:
        from agents import console_print as _cp

        _cp(*args, sep=sep, end=end, flush=flush, ensure_newline=ensure_newline)
    except Exception:
        text = sep.join("" if a is None else str(a) for a in args)
        if end:
            text += end
        print(text, end="", flush=flush)


def _log_action(kind: str, **kwargs: object) -> None:
    try:
        from agents.tools import log_action as _la

        _la(kind, **kwargs)
    except Exception:
        return



class SkillManager:
    def __init__(self, *, skills_dirs: list[Path], project_root: Path, work_dir: Path) -> None:
        self.project_root = project_root.resolve()
        self.work_dir = work_dir.resolve()

        unique_dirs: list[Path] = []
        seen: set[str] = set()
        for p in skills_dirs:
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            key = resolved.as_posix().lower()
            if key in seen:
                continue
            seen.add(key)
            unique_dirs.append(resolved)

        if not unique_dirs:
            unique_dirs.append((self.project_root / ".skills").resolve())

        self.skills_dirs = unique_dirs
        self.project_skills_dir = self.skills_dirs[0]
        self.project_skills_dir.mkdir(parents=True, exist_ok=True)

    def _npx_project_skills_dir(self) -> Path:
        return (self.project_root / ".agents" / "skills").resolve()

    def _sync_npx_project_skills_to_project_skills_dir(self) -> list[str]:
        src_root = self._npx_project_skills_dir()
        if not src_root.exists():
            return []
        self.project_skills_dir.mkdir(parents=True, exist_ok=True)
        synced: list[str] = []
        for src in sorted(src_root.iterdir(), key=lambda p: p.name.lower()):
            if not src.is_dir():
                continue
            if not (src / "SKILL.md").exists():
                continue
            dst = (self.project_skills_dir / src.name).resolve()
            try:
                dst.relative_to(self.project_skills_dir)
            except ValueError:
                continue
            if dst.exists():
                continue
            shutil.copytree(src, dst, dirs_exist_ok=False)
            synced.append(dst.as_posix())
        return synced

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        base = (self.project_root / ".agents").resolve()
        cache_dir = (base / "npm-cache").resolve()
        prefix_dir = (base / "npm-prefix").resolve()
        tmp_dir = (base / "tmp").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        prefix_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        cache = str(cache_dir)
        prefix = str(prefix_dir)
        tmp = str(tmp_dir)
        env["npm_config_cache"] = cache
        env["NPM_CONFIG_CACHE"] = cache
        env["npm_config_prefix"] = prefix
        env["NPM_CONFIG_PREFIX"] = prefix
        env["TMPDIR"] = tmp
        env["TEMP"] = tmp
        env["TMP"] = tmp
        env["npm_config_update_notifier"] = "false"
        env["npm_config_fund"] = "false"
        env["npm_config_audit"] = "false"
        env["npm_config_yes"] = "true"
        env["NPM_CONFIG_YES"] = "true"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GCM_INTERACTIVE"] = "Never"
        return env

    def _usage_snapshot(self) -> dict[str, dict[str, object]]:
        state = skills_state.load_state()
        usage = state.get("usage")
        if isinstance(usage, dict):
            out: dict[str, dict[str, object]] = {}
            for k, v in usage.items():
                if isinstance(k, str) and isinstance(v, dict):
                    out[k] = v
            return out
        return {}

    def scan(self) -> list[SkillEntry]:
        usage = self._usage_snapshot()
        manifests = discover_skills_from_dirs(self.skills_dirs)
        entries: list[SkillEntry] = []
        for m in manifests:
            u = usage.get(m.name, {})
            load_count = u.get("load_count")
            last_loaded_ts = u.get("last_loaded_ts")
            entries.append(
                SkillEntry(
                    name=m.name,
                    description=m.description,
                    skill_dir=m.skill_md_path.parent.resolve(),
                    disabled=skills_state.is_disabled(m.name),
                    load_count=int(load_count) if isinstance(load_count, int) else 0,
                    last_loaded_ts=float(last_loaded_ts) if isinstance(last_loaded_ts, (int, float)) else None,
                )
            )
        return entries

    def scan_json(self) -> str:
        rows: list[dict[str, object]] = []
        for e in self.scan():
            rows.append(
                {
                    "name": e.name,
                    "description": e.description,
                    "path": e.skill_dir.as_posix(),
                    "disabled": e.disabled,
                    "load_count": e.load_count,
                    "last_loaded_ts": e.last_loaded_ts,
                }
            )
        return json.dumps({"skills": rows, "count": len(rows)}, ensure_ascii=False, sort_keys=True)

    def disable(self, name: str, reason: str = "") -> str:
        n = (name or "").strip()
        if n in skills_state.CORE_SKILLS:
            return "Refusing to disable built-in skill."
        skills_state.disable_skill(name, reason=reason)
        return "OK"

    def enable(self, name: str) -> str:
        skills_state.enable_skill(name)
        return "OK"

    def remove_from_project(self, name: str) -> str:
        n = (name or "").strip()
        if not n:
            return "Missing name."
        target_dir: Path | None = None
        for m in discover_skills_from_dirs([self.project_skills_dir]):
            if m.name == n:
                target_dir = m.skill_md_path.parent.resolve()
                break
        if target_dir is None:
            return "Skill not found in project skills dir."
        try:
            target_dir.relative_to(self.project_skills_dir)
        except ValueError:
            return "Refusing to delete outside project skills dir."
        shutil.rmtree(target_dir, ignore_errors=False)
        return f"Deleted: {target_dir.as_posix()}"

    def create(self, name: str, description: str, body: str = "") -> str:
        n = _safe_skill_name(name)
        if not n:
            return "Invalid name."
        skill_dir = (self.project_skills_dir / n).resolve()
        try:
            skill_dir.relative_to(self.project_skills_dir)
        except ValueError:
            return "Refusing to create outside project skills dir."
        if skill_dir.exists():
            return "Skill dir already exists."
        skill_dir.mkdir(parents=True, exist_ok=False)
        skill_md = skill_dir / "SKILL.md"
        front = "\n".join(
            [
                "---",
                f'name: "{n}"',
                f'description: "{(description or "").strip()}"',
                "---",
                "",
            ]
        )
        content = front + (body.strip() + "\n" if (body or "").strip() else f"# {n}\n")
        skill_md.write_text(content, encoding="utf-8")
        skills_state.record_installed(n, source="local:create")
        return f"Created: {skill_md.as_posix()}"

    def _resolve_installed_from_spec(self, spec: str) -> tuple[str, str] | None:
        s = (spec or "").strip()
        if not s:
            return None
        safe = _safe_skill_name(s)
        candidates = {s, safe}
        manifests = discover_skills_from_dirs(self.skills_dirs)
        for m in manifests:
            if m.name in candidates or m.skill_md_path.parent.name in candidates:
                return m.name, m.skill_md_path.parent.resolve().as_posix()
        state = skills_state.load_state()
        installed = state.get("installed")
        if isinstance(installed, dict):
            key = f"npx:{s}"
            for name, meta in installed.items():
                if not isinstance(name, str) or not isinstance(meta, dict):
                    continue
                source = meta.get("source")
                if isinstance(source, str) and source.strip() == key:
                    for m in manifests:
                        if m.name == name:
                            return m.name, m.skill_md_path.parent.resolve().as_posix()
        for d in (self.project_skills_dir / s, self.project_skills_dir / safe):
            try:
                r = d.resolve()
            except Exception:
                continue
            if (r / "SKILL.md").exists():
                return r.name, r.as_posix()
        return None

    def _run(
        self,
        command: str,
        timeout_s: int,
        *,
        cwd: Path | None = None,
        stream: bool | None = None,
        early_abort_patterns: list[str] | None = None,
    ) -> tuple[int, str, str]:
        run_cwd = (cwd or self.work_dir).resolve()
        try:
            run_cwd.relative_to(self.project_root)
        except ValueError:
            return 2, "", "Refusing to run outside project root."
        cmd = _shell_command(command)
        env = self._subprocess_env()
        try:
            resolved_stream = (
                stream
                if stream is not None
                else (os.environ.get("AGENT_STREAM_SUBPROCESS") or "").strip().lower() in {"1", "true", "yes", "on"}
            )
            if resolved_stream:
                _console_print(f"$ {command}", flush=True, ensure_newline=True)
                process = subprocess.Popen(
                    cmd,
                    cwd=run_cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=(os.name != "nt"),
                )
                output_lines: list[str] = []
                output_lock = threading.Lock()
                abort_event = threading.Event()
                abort_reason: list[str] = [""]
                compiled_abort: list[re.Pattern[str]] = []
                for p in early_abort_patterns or []:
                    try:
                        compiled_abort.append(re.compile(p, re.IGNORECASE))
                    except re.error:
                        continue

                def _reader() -> None:
                    assert process.stdout is not None
                    for raw_line in process.stdout:
                        line = raw_line.rstrip("\n")
                        _console_print(line, flush=True, ensure_newline=True)
                        with output_lock:
                            output_lines.append(line)
                        if compiled_abort and not abort_event.is_set():
                            for r in compiled_abort:
                                if r.search(line):
                                    abort_reason[0] = line
                                    abort_event.set()
                                    break

                t = threading.Thread(target=_reader, daemon=True)
                t.start()
                timed_out = False
                early_aborted = False
                try:
                    if compiled_abort:
                        end_at = time.monotonic() + max(1, int(timeout_s))
                        while True:
                            remaining = end_at - time.monotonic()
                            if remaining <= 0:
                                raise subprocess.TimeoutExpired(cmd, timeout_s)
                            try:
                                exit_code = process.wait(timeout=min(0.25, remaining))
                                break
                            except subprocess.TimeoutExpired:
                                if abort_event.is_set():
                                    early_aborted = True
                                    _kill_process_tree(int(process.pid or 0))
                                    try:
                                        exit_code = process.wait(timeout=5)
                                    except Exception:
                                        exit_code = 125
                                    break
                    else:
                        exit_code = process.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_process_tree(int(process.pid or 0))
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
                    out = "\n".join(output_lines)
                err = ""
                _log_action(
                    "skills_subprocess",
                    ok=(exit_code == 0 and not timed_out and not early_aborted),
                    command=command,
                    cwd=run_cwd.as_posix(),
                    exit_code=exit_code,
                    timeout_s=timeout_s,
                    stream=True,
                )
                if timed_out:
                    return 124, out, "timeout"
                if early_aborted:
                    return 125, out, (abort_reason[0] or "early_abort")
                return exit_code, out, err

            process = subprocess.Popen(
                cmd,
                cwd=run_cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
            try:
                out, err = process.communicate(timeout=timeout_s)
                exit_code = int(process.returncode or 0)
            except subprocess.TimeoutExpired:
                _kill_process_tree(int(process.pid or 0))
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                try:
                    out, err = process.communicate(timeout=2)
                except Exception:
                    out, err = "", ""
                _log_action(
                    "skills_subprocess",
                    ok=False,
                    command=command,
                    cwd=run_cwd.as_posix(),
                    exit_code=124,
                    timeout_s=timeout_s,
                    stream=False,
                    error="timeout",
                )
                return 124, out or "", "timeout"
        except FileNotFoundError as e:
            _log_action("skills_subprocess", ok=False, command=command, cwd=run_cwd.as_posix(), error=str(e))
            return 127, "", str(e)
        _log_action(
            "skills_subprocess",
            ok=(exit_code == 0),
            command=command,
            cwd=run_cwd.as_posix(),
            exit_code=exit_code,
            stream=False if stream is None else bool(stream),
        )
        return exit_code, out or "", err or ""

    def _install_from_git_source(self, source_url: str, hint: str | None, *, timeout_s: int) -> dict[str, object]:
        src = (source_url or "").strip()
        if not src:
            return {"ok": False, "error_type": "missing_source_url"}
        if not (shutil.which("git") or "").strip():
            return {"ok": False, "error_type": "missing_git", "message": "git not found in PATH."}
        base = (self.project_root / ".agents" / "tmp").resolve()
        base.mkdir(parents=True, exist_ok=True)
        tmp_name = f"skills-src-{int(time.time())}-{os.getpid()}"
        clone_dir = (base / tmp_name).resolve()
        try:
            clone_dir.relative_to(base)
        except ValueError:
            return {"ok": False, "error_type": "tmp_escape"}
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)

        clone_cmd = f'git clone --depth 1 "{src}" "{clone_dir.as_posix()}"'
        c_code, c_out, c_err = self._run(
            clone_cmd,
            timeout_s=max(60, int(timeout_s)),
            cwd=self.project_root,
            stream=True,
            early_abort_patterns=[
                r"authentication",
                r"permission denied",
                r"could not read username",
                r"fatal:.*not found",
            ],
        )
        if c_code != 0:
            combined = "\n".join([c_out or "", c_err or ""]).strip()
            f_type, retryable = _classify_install_failure(combined)
            return {
                "ok": False,
                "error_type": "git_clone_failed",
                "retryable": retryable,
                "exit_code": c_code,
                "stdout": (c_out or "").strip(),
                "stderr": (c_err or "").strip(),
                "failure_type": f_type,
            }

        skill_dirs: list[Path] = []
        for p in clone_dir.rglob("SKILL.md"):
            try:
                p.relative_to(clone_dir)
            except ValueError:
                continue
            if p.is_file():
                skill_dirs.append(p.parent.resolve())
        picked: list[Path] = []
        hint_norm = (hint or "").strip()
        if hint_norm:
            for d in skill_dirs:
                if d.name == hint_norm:
                    picked.append(d)
        if not picked:
            picked = skill_dirs

        before = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        copied: list[str] = []
        for d in sorted(picked, key=lambda x: x.name.lower()):
            dst = (self.project_skills_dir / d.name).resolve()
            try:
                dst.relative_to(self.project_skills_dir)
            except ValueError:
                continue
            if dst.exists():
                continue
            if not (d / "SKILL.md").exists():
                continue
            shutil.copytree(d, dst, dirs_exist_ok=False)
            copied.append(dst.as_posix())

        after = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        added = sorted(after - before)
        for n in added:
            skills_state.record_installed(n, source=f"git:{src}{'@' + hint_norm if hint_norm else ''}")

        try:
            shutil.rmtree(clone_dir, ignore_errors=False)
        except Exception:
            pass

        return {"ok": True, "added": added, "copied": copied, "source_url": src, "hint": hint_norm}

    def install_via_npx(self, package_spec: str, timeout_s: int = 600, retries: int = 2) -> str:
        spec = (package_spec or "").strip()
        if not spec:
            return "Empty package_spec."
        resolved = self._resolve_installed_from_spec(spec)
        if resolved is not None:
            name, path = resolved
            return json.dumps(
                {"ok": True, "already_installed": True, "spec": spec, "name": name, "path": path, "exit_code": 0, "added": []},
                ensure_ascii=False,
                sort_keys=True,
            )
        if not (shutil.which("npx") or "").strip():
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "missing_npx",
                    "message": "npx not found. Install Node.js (includes npm/npx) or provide an offline skill directory.",
                    "env": {
                        "node": bool((shutil.which("node") or "").strip()),
                        "npm": bool((shutil.which("npm") or "").strip()),
                        "npx": False,
                        "git": bool((shutil.which("git") or "").strip()),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        before = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        agent_candidates = ["trae", "cursor", "cline", "claude-code", "continue"]
        last_code = 1
        last_out = ""
        last_err = ""
        last_combined = ""
        last_failure_type = "unknown"
        last_retryable = False
        last_source_url: str | None = None
        last_source_hint: str | None = None
        attempt_count = 0
        attempts_total = 0
        attempt_log: list[dict[str, object]] = []
        agent_used = ""
        stop_all = False
        for agent in agent_candidates:
            cmd = f'npx --yes skills add "{spec}" -y --copy --agent {agent}'
            last_code = 1
            last_out = ""
            last_err = ""
            attempt_count = 0
            for i in range(max(1, int(retries) + 1)):
                attempt_count = i + 1
                attempts_total += 1
                code, out, err = self._run(
                    cmd,
                    timeout_s=min(int(timeout_s), 240) if i == 0 else int(timeout_s),
                    cwd=self.project_root,
                    stream=True,
                    early_abort_patterns=[
                        r"\bfailed to clone repository\b",
                        r"\bclone timed out\b",
                        r"\binstallation failed\b",
                        r"\binvalid agents:\b",
                        r"\bauthentication\b",
                        r"\bpermission denied\b",
                    ],
                )
                last_code, last_out, last_err = code, out, err
                if code == 0:
                    agent_used = agent
                    break
                combined = "\n".join([out or "", err or ""]).strip()
                last_combined = combined
                if "Invalid agents:" in combined:
                    break
                failure_type, retryable = _classify_install_failure(combined)
                last_failure_type, last_retryable = failure_type, retryable
                source_url, source_hint = _extract_source_ref(combined)
                last_source_url, last_source_hint = source_url, source_hint
                attempt_log.append(
                    {
                        "agent": agent,
                        "attempt": attempt_count,
                        "exit_code": code,
                        "failure_type": failure_type,
                        "retryable": retryable,
                        "source_url": source_url or "",
                        "source_hint": source_hint or "",
                    }
                )
                if failure_type in {"repo_not_found", "auth"}:
                    stop_all = True
                    break
                if failure_type in {"clone_timeout", "clone_failed", "network", "timeout"} and source_url:
                    direct = self._install_from_git_source(source_url, source_hint, timeout_s=max(180, int(timeout_s)))
                    if bool(direct.get("ok")):
                        return json.dumps(
                            {
                                "ok": True,
                                "exit_code": 0,
                                "added": direct.get("added", []),
                                "copied": direct.get("copied", []),
                                "method": "git_clone_fallback",
                                "source_url": direct.get("source_url", source_url),
                                "hint": direct.get("hint", source_hint or ""),
                                "attempts_total": attempts_total,
                                "agent_tried": agent_candidates,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                if not retryable or i >= int(retries):
                    break
                time.sleep(1.0 if i == 0 else 2.0)
            if agent_used:
                break
            if stop_all:
                break

        code, out, err = last_code, last_out, last_err
        if code != 0:
            combined = last_combined or "\n".join([out or "", err or ""]).strip()
            failure_type, retryable = _classify_install_failure(combined)
            source_url, source_hint = _extract_source_ref(combined)
            diagnostics: dict[str, object] = {}
            if source_url:
                diag_cmd = f'git ls-remote "{source_url}"'
                d_code, d_out, d_err = self._run(diag_cmd, timeout_s=30, cwd=self.project_root)
                diagnostics = {
                    "source_url": source_url,
                    "git_ls_remote": {"exit_code": d_code, "stdout": (d_out or "").strip(), "stderr": (d_err or "").strip()},
                }
            suggestions: list[str] = []
            if failure_type == "repo_not_found":
                suggestions.append("package_spec 必须是 owner/repo 或可 clone 的 URL（例如 https://github.com/vercel-labs/agent-skills）")
                suggestions.append('示例：skills_install("vercel-labs/agent-skills")')
                suggestions.append(f'若你只有关键词，可先用命令检索：npx skills find "{spec}"')
                suggestions.append(f'或命令：npx skills add "{spec}" --agent trae -y --copy')
            if failure_type in {"clone_timeout", "timeout"}:
                suggestions.append("网络或代理导致 clone 超时：检查代理/公司网络/防火墙，或切换网络后重试")
                if source_url:
                    suggestions.append(f'可手动验证：git ls-remote "{source_url}"')
            if failure_type in {"network"}:
                suggestions.append("网络不稳定：重试，或配置 npm/git 的代理后重试")
            if failure_type in {"auth"}:
                suggestions.append("私有仓库需要鉴权：确保已配置 SSH key 或 GitHub CLI 登录（gh auth status）")
                suggestions.append("若使用 HTTPS：检查凭据管理器/Token 权限；若使用 SSH：检查 ssh-add -l")
            if failure_type in {"clone_failed"}:
                suggestions.append("clone 失败但非鉴权/不存在：可能是网络抖动或 git 配置问题，建议手动 git clone 验证")
                if source_url and source_hint:
                    suggestions.append(f"如果仓库较大，可只安装该技能目录：{source_hint}")

            msg = ""
            if failure_type == "clone_timeout":
                msg = "克隆仓库超时，已自动终止并重试；仍失败。"
            elif failure_type == "auth":
                msg = "仓库需要鉴权或无权限访问，已停止重试。"
            elif failure_type == "repo_not_found":
                msg = "仓库不存在或不可访问，已停止重试。"
            elif failure_type in {"network", "timeout"}:
                msg = "网络超时/异常，已自动重试；仍失败。"
            else:
                msg = "技能安装失败。"
            return json.dumps(
                {
                    "exit_code": code,
                    "ok": False,
                    "error_type": failure_type,
                    "retryable": retryable,
                    "attempts": attempt_count,
                    "attempts_total": attempts_total,
                    "agent_tried": agent_candidates,
                    "source_url": source_url or "",
                    "source_hint": source_hint or "",
                    "message": msg,
                    "attempt_log": attempt_log[-10:],
                    "stdout": (out or "").strip(),
                    "stderr": (err or "").strip(),
                    "diagnostics": diagnostics,
                    "suggestions": suggestions,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        installed_agent_dir = (self.project_root / f".{agent_used}").resolve()
        agent_skills_root = (installed_agent_dir / "skills").resolve()
        copied: list[str] = []
        if agent_skills_root.exists() and agent_skills_root.is_dir():
            for manifest in discover_skills_from_dirs([agent_skills_root]):
                src = manifest.skill_md_path.parent.resolve()
                dst = (self.project_skills_dir / src.name).resolve()
                try:
                    dst.relative_to(self.project_skills_dir)
                except ValueError:
                    continue
                if dst.exists():
                    continue
                shutil.copytree(src, dst, dirs_exist_ok=False)
                copied.append(dst.as_posix())

        try:
            if installed_agent_dir.exists():
                shutil.rmtree(installed_agent_dir, ignore_errors=False)
        except Exception:
            pass

        after = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        added = sorted(after - before)
        for n in added:
            skills_state.record_installed(n, source=f"npx:{spec}")
        return json.dumps(
            {
                "exit_code": code,
                "added": added,
                "agent_used": agent_used,
                "copied": copied,
                "stdout": out,
                "stderr": err,
            },
            ensure_ascii=False,
        )

    def find_via_npx(self, query: str, timeout_s: int = 120) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty query."
        if not (shutil.which("npx") or "").strip():
            return json.dumps(
                {
                    "ok": False,
                    "error_type": "missing_npx",
                    "message": "npx not found. Install Node.js (includes npm/npx).",
                    "env": {
                        "node": bool((shutil.which("node") or "").strip()),
                        "npm": bool((shutil.which("npm") or "").strip()),
                        "npx": False,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        cmd = f'npx --yes skills find "{q}"'
        code, out, err = self._run(cmd, timeout_s=timeout_s, cwd=self.project_root)
        if code != 0:
            msg = "\n".join([f"exit_code={code}", out.strip(), err.strip()]).strip()
            return msg or f"Failed with exit_code={code}"
        return (out or "").strip()

    def ensure_installed(self, name_or_package_spec: str) -> str:
        x = (name_or_package_spec or "").strip()
        if not x:
            return "Empty name_or_package_spec."
        resolved = self._resolve_installed_from_spec(x)
        if resolved is not None:
            name, path = resolved
            return json.dumps({"already_installed": True, "name": name, "path": path}, ensure_ascii=False, sort_keys=True)
        return self.install_via_npx(x)

    def prune_disabled_from_project(self) -> str:
        state = skills_state.load_state()
        disabled = state.get("disabled")
        if not isinstance(disabled, dict) or not disabled:
            return json.dumps({"deleted": []}, ensure_ascii=False)
        deleted: list[str] = []
        for name in list(disabled.keys()):
            if name in skills_state.CORE_SKILLS:
                continue
            target_dir: Path | None = None
            for m in discover_skills_from_dirs([self.project_skills_dir]):
                if m.name == name:
                    target_dir = m.skill_md_path.parent.resolve()
                    break
            if target_dir is None:
                continue
            try:
                target_dir.relative_to(self.project_skills_dir)
            except ValueError:
                continue
            shutil.rmtree(target_dir, ignore_errors=False)
            deleted.append(target_dir.as_posix())
        return json.dumps({"deleted": deleted}, ensure_ascii=False, sort_keys=True)
