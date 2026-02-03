from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from . import skills_state


BASE_SYSTEM_PROMPT = "You are a single agent that can use skills via progressive disclosure."


@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    skill_md_path: Path


def _parse_skill_front_matter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}

    meta: dict[str, str] = {}
    for raw in lines[1:end_idx]:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta


def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return text
    return "\n".join(lines[end_idx + 1 :]).lstrip("\n")


def discover_skills(skills_dir: Path) -> list[SkillManifest]:
    manifests: list[SkillManifest] = []
    if not skills_dir.exists():
        return manifests

    for skill_md_path in skills_dir.rglob("SKILL.md"):
        try:
            text = skill_md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = skill_md_path.read_text(encoding="utf-8-sig")

        meta = _parse_skill_front_matter(text)
        name = (meta.get("name") or skill_md_path.parent.name).strip()
        description = (meta.get("description") or "").strip()
        if not description:
            description = f"Skill in {skill_md_path.parent.as_posix()}"

        manifests.append(
            SkillManifest(
                name=name,
                description=description,
                skill_md_path=skill_md_path,
            )
        )

    manifests.sort(key=lambda s: s.name.lower())
    return manifests


def discover_skills_from_dirs(skills_dirs: Sequence[Path]) -> list[SkillManifest]:
    seen: set[str] = set()
    unique: list[SkillManifest] = []
    for skills_dir in skills_dirs:
        for manifest in discover_skills(skills_dir):
            if manifest.name in seen:
                continue
            seen.add(manifest.name)
            unique.append(manifest)
    unique.sort(key=lambda s: s.name.lower())
    return unique


def discover_enabled_skills_from_dirs(skills_dirs: Sequence[Path]) -> list[SkillManifest]:
    return [s for s in discover_skills_from_dirs(skills_dirs) if not skills_state.is_disabled(s.name)]


def build_skill_catalog_text(manifests: list[SkillManifest]) -> str:
    if not manifests:
        return "No skills found."
    lines = ["Available skills (load on-demand):"]
    for s in manifests:
        lines.append(f"- {s.name}: {s.description}")
    return "\n".join(lines)


def _compute_skill_fingerprint(skills_dirs: Iterable[Path]) -> tuple[tuple[str, int, int], ...]:
    entries: list[tuple[str, int, int]] = []
    for skills_dir in skills_dirs:
        if not skills_dir.exists():
            continue
        for p in skills_dir.rglob("SKILL.md"):
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((p.as_posix(), st.st_mtime_ns, st.st_size))
    entries.sort()
    return tuple(entries)


class _SkillIndex:
    def __init__(self, skills_dirs: Sequence[Path]) -> None:
        self.skills_dirs = tuple(skills_dirs)
        self._fingerprint: tuple[tuple[str, int, int], ...] | None = None
        self._manifests: list[SkillManifest] = []
        self._skill_by_name: dict[str, SkillManifest] = {}
        self._catalog_text: str = "No skills found."
        self._loaded_skill_cache: dict[str, str] = {}
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> bool:
        fingerprint = _compute_skill_fingerprint(self.skills_dirs)
        if not force and self._fingerprint == fingerprint:
            return False

        manifests = discover_enabled_skills_from_dirs(self.skills_dirs)
        self._manifests = manifests
        self._skill_by_name = {s.name: s for s in manifests}
        self._catalog_text = build_skill_catalog_text(manifests)
        self._loaded_skill_cache = {}
        self._fingerprint = fingerprint
        return True

    def get_catalog_text(self) -> str:
        self.refresh()
        return self._catalog_text

    def load_skill(self, name: str) -> str:
        self.refresh()
        manifest = self._skill_by_name.get(name)
        if manifest is None:
            known = ", ".join(sorted(self._skill_by_name.keys()))
            return f"Unknown skill: {name}. Known skills: {known}"

        skills_state.record_skill_loaded(name)

        cached = self._loaded_skill_cache.get(name)
        if cached is not None:
            return cached

        try:
            text = manifest.skill_md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = manifest.skill_md_path.read_text(encoding="utf-8-sig")
        content = _strip_front_matter(text)
        self._loaded_skill_cache[name] = content
        return content


class SkillMiddleware(AgentMiddleware):
    def __init__(self, *, skills_prompt_supplier: Callable[[], str], tools: list) -> None:
        self.skills_prompt_supplier = skills_prompt_supplier
        self.tools = tools

    def wrap_model_call(self, request: ModelRequest, handler):
        base_text = request.system_message.text if request.system_message is not None else ""
        marker = "## Available Skills"

        skills_addendum = "\n".join(
            [
                "",
                marker,
                "",
                self.skills_prompt_supplier(),
                "",
                "Use the load_skill tool when you need detailed information about handling a specific type of request.",
            ]
        ).strip("\n")

        if marker in base_text:
            prefix = base_text.split(marker, 1)[0].rstrip()
            new_text = f"{prefix}\n\n{skills_addendum}".strip()
        else:
            base = base_text.strip()
            new_text = f"{base}\n\n{skills_addendum}".strip() if base else skills_addendum

        if new_text == base_text:
            return handler(request)
        modified_request = request.override(system_message=SystemMessage(content=new_text))
        return handler(modified_request)


def create_skill_middleware(skills_dir: Path | Sequence[Path]) -> tuple[str, SkillMiddleware]:
    skills_dirs = [skills_dir] if isinstance(skills_dir, Path) else list(skills_dir)
    index = _SkillIndex(skills_dirs)
    skill_catalog_text = index.get_catalog_text()

    @tool
    def list_skills() -> str:
        """List available skills with short descriptions."""
        return index.get_catalog_text()

    @tool
    def load_skill(name: str) -> str:
        """Load full SKILL.md content for a single skill by name."""
        return index.load_skill(name)

    return skill_catalog_text, SkillMiddleware(
        skills_prompt_supplier=index.get_catalog_text,
        tools=[list_skills, load_skill],
    )

