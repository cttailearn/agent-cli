from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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


class SkillManager:
    def __init__(self, *, skills_dirs: list[Path], project_root: Path, work_dir: Path) -> None:
        self.skills_dirs = [p.resolve() for p in skills_dirs]
        self.project_root = project_root.resolve()
        self.work_dir = work_dir.resolve()
        self.project_skills_dir = self.skills_dirs[0] if self.skills_dirs else (self.project_root / "skills").resolve()

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

    def _run(self, command: str, timeout_s: int) -> tuple[int, str, str]:
        try:
            self.work_dir.relative_to(self.project_root)
        except ValueError:
            return 2, "", "Refusing to run outside project root."
        cmd = _shell_command(command)
        try:
            completed = subprocess.run(
                cmd,
                cwd=self.work_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except FileNotFoundError as e:
            return 127, "", str(e)
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
        return completed.returncode, completed.stdout or "", completed.stderr or ""

    def install_via_npx(self, package_spec: str, timeout_s: int = 600) -> str:
        spec = (package_spec or "").strip()
        if not spec:
            return "Empty package_spec."
        before = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        cmd = f'npx skills add "{spec}" --dir "{self.project_skills_dir.as_posix()}"'
        code, out, err = self._run(cmd, timeout_s=timeout_s)
        after = {m.name for m in discover_skills_from_dirs(self.skills_dirs)}
        added = sorted(after - before)
        if code != 0:
            msg = "\n".join([f"exit_code={code}", out.strip(), err.strip()]).strip()
            return msg or f"Failed with exit_code={code}"
        for n in added:
            skills_state.record_installed(n, source=f"npx:{spec}")
        return json.dumps({"exit_code": code, "added": added, "stdout": out, "stderr": err}, ensure_ascii=False)

    def find_via_npx(self, query: str, timeout_s: int = 120) -> str:
        q = (query or "").strip()
        if not q:
            return "Empty query."
        cmd = f'npx skills find "{q}"'
        code, out, err = self._run(cmd, timeout_s=timeout_s)
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

