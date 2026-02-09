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

from agents import console_print
from agents.tools import log_action

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


def _extract_source_url(text: str) -> str | None:
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("◇") or line.startswith("│"):
            m = re.search(r"\bSource:\s*(https?://\S+)", line)
            if m:
                return m.group(1).strip()
    m = re.search(r"\bSource:\s*(https?://\S+)", text or "")
    return m.group(1).strip() if m else None


def _classify_install_failure(text: str) -> tuple[str, bool]:
    s = (text or "").lower()
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



class SkillManager:
    def __init__(self, *, skills_dirs: list[Path], project_root: Path, work_dir: Path) -> None:
        self.skills_dirs = [p.resolve() for p in skills_dirs]
        self.project_root = project_root.resolve()
        self.work_dir = work_dir.resolve()
        self.project_skills_dir = self.skills_dirs[0] if self.skills_dirs else (self.project_root / "skills").resolve()

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

    def _run(self, command: str, timeout_s: int, *, cwd: Path | None = None) -> tuple[int, str, str]:
        run_cwd = (cwd or self.work_dir).resolve()
        try:
            run_cwd.relative_to(self.project_root)
        except ValueError:
            return 2, "", "Refusing to run outside project root."
        cmd = _shell_command(command)
        env = self._subprocess_env()
        try:
            stream = (os.environ.get("AGENT_STREAM_SUBPROCESS") or "").strip().lower() in {"1", "true", "yes", "on"}
            if stream:
                console_print(f"$ {command}", flush=True, ensure_newline=True)
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
                log_action(
                    "skills_subprocess",
                    ok=(exit_code == 0 and not timed_out),
                    command=command,
                    cwd=run_cwd.as_posix(),
                    exit_code=exit_code,
                    timeout_s=timeout_s,
                    stream=True,
                )
                if timed_out:
                    return 124, out, "timeout"
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
                log_action(
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
            log_action("skills_subprocess", ok=False, command=command, cwd=run_cwd.as_posix(), error=str(e))
            return 127, "", str(e)
        log_action(
            "skills_subprocess",
            ok=(exit_code == 0),
            command=command,
            cwd=run_cwd.as_posix(),
            exit_code=exit_code,
            stream=False,
        )
        return exit_code, out or "", err or ""

    def install_via_npx(self, package_spec: str, timeout_s: int = 600, retries: int = 2) -> str:
        spec = (package_spec or "").strip()
        if not spec:
            return "Empty package_spec."
        before = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        cmd = f'npx --yes skills add "{spec}" -y'
        last_code = 1
        last_out = ""
        last_err = ""
        attempt_count = 0
        for i in range(max(1, int(retries) + 1)):
            attempt_count = i + 1
            code, out, err = self._run(cmd, timeout_s=timeout_s, cwd=self.project_root)
            last_code, last_out, last_err = code, out, err
            if code == 0:
                break
            combined = "\n".join([out or "", err or ""]).strip()
            failure_type, retryable = _classify_install_failure(combined)
            if not retryable or i >= int(retries):
                break
            time.sleep(1.0 if i == 0 else 2.0)

        code, out, err = last_code, last_out, last_err
        after = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        added = sorted(after - before)
        if code != 0:
            combined = "\n".join([out or "", err or ""]).strip()
            failure_type, retryable = _classify_install_failure(combined)
            source_url = _extract_source_url(combined)
            diagnostics: dict[str, object] = {}
            if source_url:
                diag_cmd = f'git ls-remote "{source_url}"'
                d_code, d_out, d_err = self._run(diag_cmd, timeout_s=30, cwd=self.project_root)
                diagnostics = {
                    "source_url": source_url,
                    "git_ls_remote": {"exit_code": d_code, "stdout": (d_out or "").strip(), "stderr": (d_err or "").strip()},
                }
            return json.dumps(
                {
                    "exit_code": code,
                    "ok": False,
                    "error_type": failure_type,
                    "retryable": retryable,
                    "attempts": attempt_count,
                    "stdout": (out or "").strip(),
                    "stderr": (err or "").strip(),
                    "diagnostics": diagnostics,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        synced = self._sync_npx_project_skills_to_project_skills_dir()
        after = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        added = sorted(after - before)
        for n in added:
            skills_state.record_installed(n, source=f"npx:{spec}")
        return json.dumps({"exit_code": code, "added": added, "synced": synced, "stdout": out, "stderr": err}, ensure_ascii=False)

    def find_via_npx(self, query: str, timeout_s: int = 120) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty query."
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
        installed = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        if x in installed:
            return json.dumps({"already_installed": True, "name": x}, ensure_ascii=False, sort_keys=True)
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
