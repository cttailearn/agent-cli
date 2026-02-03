from __future__ import annotations

import json
import re
import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool

from skills.skills_manager import SkillManager
from skills import skills_state
from skills.skills_support import BASE_SYSTEM_PROMPT, SkillMiddleware

from memory import load_core_prompt

from .runtime import _format_tools, _init_model


def build_supervisor_agent(
    *,
    model_name: str,
    skill_middleware: SkillMiddleware,
    skills_dirs: list[Path],
    project_root: Path,
    output_dir: Path,
    work_dir: Path,
):
    model = _init_model(model_name)

    tasks: dict[str, dict[str, object]] = {}
    skill_manager = SkillManager(skills_dirs=skills_dirs, project_root=project_root, work_dir=work_dir)

    tools: list[object] = []

    @tool
    def list_tools() -> str:
        """List available tools with short descriptions."""
        return _format_tools(tools)

    @tool
    def start_task(task_id: str, description: str) -> str:
        """Create or reset a task record in memory."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        existing = tasks.get(tid)
        if isinstance(existing, dict) and not existing.get("completed"):
            if not existing.get("description") and (description or "").strip():
                existing["description"] = (description or "").strip()
            return "OK"
        tasks[tid] = {
            "task_id": tid,
            "description": (description or "").strip(),
            "observations": [],
            "executor_outputs": [],
            "judgements": [],
            "required_skills": [],
            "optimization_tasks": [],
            "skill_actions": [],
            "completed": False,
        }
        return "OK"

    @tool
    def add_observation(task_id: str, note: str) -> str:
        """Append an observation note to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        obs = t.get("observations")
        if isinstance(obs, list):
            obs.append((note or "").strip())
        return "OK"

    @tool
    def record_executor_output(task_id: str, output: str) -> str:
        """Append executor output to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        outs = t.get("executor_outputs")
        if isinstance(outs, list):
            outs.append((output or "").strip())
        raw = output or ""
        missing = re.findall(r"Unknown skill:\s*([A-Za-z0-9_.-]+)", raw)
        if missing:
            req = t.get("required_skills")
            if isinstance(req, list):
                now = time.time()
                existing = {item.get("name") for item in req if isinstance(item, dict)}
                for s in missing:
                    if s in existing:
                        continue
                    req.append({"name": s, "reason": "executor_output:Unknown skill", "ts": now})
        return "OK"

    @tool
    def add_required_skill(task_id: str, skill_name: str, reason: str = "") -> str:
        """Record a missing/needed skill for a task."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        n = (skill_name or "").strip()
        if not n:
            return "Missing skill_name."
        req = t.get("required_skills")
        if not isinstance(req, list):
            req = []
            t["required_skills"] = req
        existing = {item.get("name") for item in req if isinstance(item, dict)}
        if n not in existing:
            req.append({"name": n, "reason": (reason or "").strip(), "ts": time.time()})
        return "OK"

    @tool
    def get_required_skills(task_id: str) -> str:
        """Get required skills list as JSON."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        req = t.get("required_skills")
        if not isinstance(req, list):
            req = []
        return json.dumps(req, ensure_ascii=False, sort_keys=True)

    @tool
    def skills_scan() -> str:
        """Scan available skills (including disabled) with usage stats."""
        return skill_manager.scan_json()

    @tool
    def skills_disable(name: str, reason: str = "") -> str:
        """Disable a skill (it will be hidden from the catalog)."""
        return skill_manager.disable(name, reason=reason)

    @tool
    def skills_enable(name: str) -> str:
        """Enable a previously disabled skill."""
        return skill_manager.enable(name)

    @tool
    def skills_remove(name: str) -> str:
        """Delete a skill directory from project skills dir."""
        return skill_manager.remove_from_project(name)

    @tool
    def skills_create(name: str, description: str, body: str = "") -> str:
        """Create a new local skill under project skills dir."""
        return skill_manager.create(name, description, body=body)

    @tool
    def skills_find(query: str) -> str:
        """Find skills from the skills ecosystem via npx skills."""
        return skill_manager.find_via_npx(query)

    @tool
    def skills_install(package_spec: str) -> str:
        """Install a skill into project skills dir via npx skills add."""
        return skill_manager.install_via_npx(package_spec)

    @tool
    def skills_ensure(name_or_package_spec: str) -> str:
        """Ensure a skill is installed; if missing, install via npx."""
        return skill_manager.ensure_installed(name_or_package_spec)

    @tool
    def skills_prune_disabled() -> str:
        """Delete disabled skills from project skills dir."""
        return skill_manager.prune_disabled_from_project()

    def _default_optimizations(t: dict[str, object]) -> list[str]:
        req = t.get("required_skills")
        required: list[str] = []
        if isinstance(req, list):
            for item in req:
                if isinstance(item, dict):
                    n = item.get("name")
                    if isinstance(n, str) and n.strip():
                        required.append(n.strip())
        tasks_out: list[str] = []
        if required:
            tasks_out.append(f"补齐缺失技能：{', '.join(sorted(set(required)))}（先 find，再 install/或 create）")
        disabled = skills_state.load_state().get("disabled")
        if isinstance(disabled, dict) and disabled:
            tasks_out.append("清理或替换已禁用技能（评估原因，必要时 remove 或 enable 后修复）")
        tasks_out.append("把高频流程沉淀为技能：补充 SKILL.md 的步骤与示例")
        return tasks_out[:5]

    @tool
    def finalize_task(task_id: str, keep_record: bool = False) -> str:
        """Snapshot task record, attach optimization tasks, then mark completed and optionally forget."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        if not isinstance(t.get("optimization_tasks"), list) or not t.get("optimization_tasks"):
            t["optimization_tasks"] = _default_optimizations(t)
        snapshot_dir = (output_dir / "supervision").resolve()
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = (snapshot_dir / f"{tid}.json").resolve()
        snapshot_path.write_text(json.dumps(t, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        t["completed"] = True
        optimizations = t.get("optimization_tasks")
        if isinstance(optimizations, list):
            optimizations_out = optimizations
        else:
            optimizations_out = []
        if not keep_record:
            tasks.pop(tid, None)
        return json.dumps(
            {"snapshot": snapshot_path.as_posix(), "optimization_tasks": optimizations_out},
            ensure_ascii=False,
            sort_keys=True,
        )

    @tool
    def add_judgement(task_id: str, judgement: str) -> str:
        """Append a judgement item to a task record."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        js = t.get("judgements")
        if isinstance(js, list):
            js.append((judgement or "").strip())
        return "OK"

    @tool
    def get_task(task_id: str) -> str:
        """Get the current task record as JSON."""
        tid = (task_id or "").strip()
        if not tid:
            return ""
        t = tasks.get(tid)
        if not t:
            return ""
        return json.dumps(t, ensure_ascii=False, sort_keys=True)

    @tool
    def mark_completed(task_id: str) -> str:
        """Mark a task record as completed."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Task not found."
        t["completed"] = True
        return "OK"

    @tool
    def forget_task(task_id: str) -> str:
        """Forget a task record only if completed."""
        tid = (task_id or "").strip()
        if not tid:
            return "Missing task_id."
        t = tasks.get(tid)
        if not t:
            return "Not found."
        if not t.get("completed"):
            return "Task not completed."
        tasks.pop(tid, None)
        return "OK"

    tools.extend(
        [
            *skill_middleware.tools,
            list_tools,
            start_task,
            add_observation,
            record_executor_output,
            add_judgement,
            get_task,
            add_required_skill,
            get_required_skills,
            skills_scan,
            skills_disable,
            skills_enable,
            skills_remove,
            skills_create,
            skills_find,
            skills_install,
            skills_ensure,
            skills_prune_disabled,
            mark_completed,
            forget_task,
            finalize_task,
        ]
    )

    system_prompt = "\n\n".join(
        [
            BASE_SYSTEM_PROMPT,
            "你是一个监督者（Supervisor）。你不与用户直接对话，你的输出将交由观察者使用。",
            "你的职责是：判断任务是否被正确、完整地执行，并指出未完成项、风险与下一步。",
            "当任务被视为复杂任务时，观察者会把 task_id 和任务描述交给你。",
            "你必须把任务记忆写入工具：start_task/add_observation/record_executor_output/add_judgement。",
            "你还负责技能与流程优化：记录 required_skills；必要时调用 skills_find/skills_install/skills_create/skills_disable/skills_remove 管理技能。",
            "当你创建技能（skills_create）时，必须在 SKILL.md 写清楚：适用场景、步骤、示例、边界/风险、失败处理。",
            "当任务完成进入空闲时，调用 finalize_task(task_id) 生成优化任务并输出给观察者。",
            "在任务未完成前，严禁遗忘任务记忆；只有当任务完成并且 mark_completed 被调用后，才允许 forget_task。",
            "严禁虚构已执行的动作：你只能依据执行者提供的可验证结果判断完成情况。",
            "你的输出可能会被观察者直接转述给用户：不得提及“监督者/观察者/执行者/委派/转述/工具调用/内部流程”等字样，不得暴露多智能体结构。",
            "用用户可理解的方式给出 judgement：完成度、缺失项、风险、下一步；必要时包含可验证证据（路径/退出码/摘要）。",
            f"项目目录：{project_root.as_posix()}",
            f"输出目录：{output_dir.as_posix()}",
            f"工作目录：{work_dir.as_posix()}",
            load_core_prompt(project_root),
        ]
    )

    agent = create_agent(
        model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[skill_middleware],
    )
    return agent, tools
