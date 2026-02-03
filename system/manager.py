from __future__ import annotations

from pathlib import Path

from memory.manager import MemoryManager

from .scheduler import SystemScheduler
from .schedules import DailyAtSchedule, IntervalSchedule, ManualSchedule, Schedule
from .tasks import KnowledgeGraphBackfillTask, ObserverPromptTask, SystemContext, SystemTask


class SystemManager:
    def __init__(
        self,
        *,
        project_root: Path,
        output_dir: Path,
        work_dir: Path,
        model_name: str,
        observer_agent: object,
        memory_manager: MemoryManager,
    ) -> None:
        self._ctx = SystemContext(
            project_root=project_root.resolve(),
            output_dir=output_dir.resolve(),
            work_dir=work_dir.resolve(),
            model_name=model_name,
            observer_agent=observer_agent,
            memory_manager=memory_manager,
        )
        tasks = self._load_tasks()
        self._scheduler = SystemScheduler(ctx=self._ctx, tasks=tasks)

    def start(self) -> None:
        self._scheduler.start()

    def stop(self) -> None:
        self._scheduler.stop()

    def trigger(self, task_id: str) -> bool:
        return self._scheduler.trigger(task_id)

    def _load_tasks(self) -> list[SystemTask]:
        tasks: list[SystemTask] = [
            KnowledgeGraphBackfillTask(
                id="kg_daily_backfill",
                schedule=DailyAtSchedule(hour=0, minute=5, second=0),
            )
        ]
        tasks.extend(self._load_agent_tasks())
        return tasks

    def _load_agent_tasks(self) -> list[SystemTask]:
        try:
            from agents.system_tasks import AgentSystemTaskSpec, get_system_task_specs
        except Exception:
            return []

        def build_schedule(spec: AgentSystemTaskSpec) -> Schedule:
            t = (spec.schedule_type or "").strip().lower()
            args = spec.schedule_args or {}
            if t == "daily_at":
                return DailyAtSchedule(
                    hour=int(args.get("hour", 0)),
                    minute=int(args.get("minute", 0)),
                    second=int(args.get("second", 0)),
                )
            if t == "interval":
                return IntervalSchedule(seconds=float(args.get("seconds", 60.0)))
            return ManualSchedule()

        tasks: list[SystemTask] = []
        for spec in get_system_task_specs():
            kind = (spec.kind or "").strip().lower()
            if kind == "observer_prompt":
                prompt = spec.payload.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    continue
                tasks.append(
                    ObserverPromptTask(
                        id=spec.id,
                        schedule=build_schedule(spec),
                        prompt=prompt,
                    )
                )
        return tasks
