from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from memory.manager import MemoryManager

from .scheduler import SystemScheduler
from .schedules import DailyAtSchedule, IntervalSchedule, ManualSchedule, OneShotSchedule, Schedule
from .tasks import (
    AgentReminderTask,
    MemoryRollupTask,
    ObserverPromptTask,
    SystemContext,
    SystemTask,
)

_GLOBAL_MANAGER: "SystemManager | None" = None
_GLOBAL_MANAGER_LOCK = threading.Lock()


def set_global_system_manager(manager: "SystemManager | None") -> None:
    global _GLOBAL_MANAGER
    with _GLOBAL_MANAGER_LOCK:
        _GLOBAL_MANAGER = manager


def get_global_system_manager() -> "SystemManager | None":
    with _GLOBAL_MANAGER_LOCK:
        return _GLOBAL_MANAGER


def _local_tzinfo():
    return datetime.now().astimezone().tzinfo


def _ts_to_iso(ts: float) -> str:
    dt = datetime.fromtimestamp(float(ts), tz=_local_tzinfo())
    return dt.isoformat(sep=" ", timespec="seconds")


def _parse_datetime_to_ts(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        pass
    s2 = s.replace(" ", "T")
    if s2.endswith("Z"):
        s2 = s2[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s2)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_local_tzinfo())
    try:
        return float(dt.timestamp())
    except Exception:
        return None


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
        self._reminders_dir = (self._ctx.project_root / "schedules").resolve()
        self._reminders_path = (self._reminders_dir / "reminders.json").resolve()
        self._legacy_reminders_path = (self._ctx.project_root / ".agents" / "reminders.json").resolve()
        self._reminders_lock = threading.Lock()
        tasks = self._load_tasks()
        self._scheduler = SystemScheduler(ctx=self._ctx, tasks=tasks)
        set_global_system_manager(self)

    def start(self) -> None:
        self._migrate_legacy_reminders()
        self._scheduler.start()

    def stop(self) -> None:
        self._scheduler.stop()
        set_global_system_manager(None)

    def trigger(self, task_id: str) -> bool:
        return self._scheduler.trigger(task_id)

    def reminder_create_at(self, *, run_ts: float, message: str, reminder_id: str = "") -> str:
        msg = (message or "").strip()
        if not msg:
            raise ValueError("message is required")
        rid = (reminder_id or "").strip() or f"rem_{uuid.uuid4().hex[:12]}"
        now_ts = time.time()
        item = {
            "id": rid,
            "run_at": _ts_to_iso(float(run_ts)),
            "created_at": _ts_to_iso(float(now_ts)),
            "status": "scheduled",
            "message": msg,
        }
        with self._reminders_lock:
            state = self._reminders_load()
            items = state.get("items")
            if not isinstance(items, list):
                items = []
                state["items"] = items
            items = [it for it in items if not (isinstance(it, dict) and (it.get("id") == rid))]
            items.append(item)
            state["items"] = items
            self._reminders_save(state)

        task = AgentReminderTask(
            id=rid,
            schedule=OneShotSchedule(run_ts=float(run_ts)),
            message=msg,
            on_complete=lambda final_text, rid=rid: self._reminder_mark_done(rid, final_text),
        )
        self._scheduler.register_task(task)
        if float(run_ts) <= now_ts:
            self._scheduler.trigger(rid, when_ts=now_ts + 0.1)
        return rid

    def reminder_create_in(self, *, delay_s: float, message: str, reminder_id: str = "") -> str:
        s = float(delay_s)
        if s <= 0:
            raise ValueError("delay_s must be > 0")
        return self.reminder_create_at(run_ts=time.time() + s, message=message, reminder_id=reminder_id)

    def reminder_cancel(self, reminder_id: str) -> bool:
        rid = (reminder_id or "").strip()
        if not rid:
            return False
        changed = False
        with self._reminders_lock:
            state = self._reminders_load()
            items = state.get("items")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    if (it.get("id") or "") == rid and (it.get("status") or "") == "scheduled":
                        it["status"] = "canceled"
                        it["canceled_at"] = _ts_to_iso(time.time())
                        changed = True
            if changed:
                self._reminders_save(state)
        if changed:
            self._scheduler.unregister_task(rid)
        return changed

    def reminder_list(self) -> list[dict[str, object]]:
        with self._reminders_lock:
            state = self._reminders_load()
        items = state.get("items")
        if not isinstance(items, list):
            return []
        out: list[dict[str, object]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            rid = it.get("id")
            if not isinstance(rid, str) or not rid.strip():
                continue
            normalized = self._normalize_reminder_item(it)
            if normalized is None:
                continue
            out.append(normalized)
        out.sort(key=lambda x: float(_parse_datetime_to_ts(x.get("run_at")) or 0.0))
        return out

    def _load_tasks(self) -> list[SystemTask]:
        tasks: list[SystemTask] = []
        tasks.extend(self._load_agent_tasks())
        tasks.extend(self._load_reminder_tasks())
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
            if kind == "memory_rollup":
                tasks.append(
                    MemoryRollupTask(
                        id=spec.id,
                        schedule=build_schedule(spec),
                    )
                )
        return tasks

    def _load_reminder_tasks(self) -> list[SystemTask]:
        with self._reminders_lock:
            state = self._reminders_load()
        items = state.get("items")
        if not isinstance(items, list):
            return []

        tasks: list[SystemTask] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            if (it.get("status") or "") != "scheduled":
                continue
            rid = it.get("id")
            msg = it.get("message")
            run_ts = it.get("run_ts")
            if not isinstance(rid, str) or not rid.strip():
                continue
            if not isinstance(msg, str) or not msg.strip():
                continue
            t = _parse_datetime_to_ts(it.get("run_at"))
            if t is None:
                t = _parse_datetime_to_ts(run_ts)
            if t is None:
                continue
            tasks.append(
                AgentReminderTask(
                    id=rid.strip(),
                    schedule=OneShotSchedule(run_ts=t),
                    message=msg.strip(),
                    on_complete=lambda final_text, rid=rid.strip(): self._reminder_mark_done(rid, final_text),
                )
            )
        return tasks

    def _normalize_reminder_item(self, it: dict[str, object]) -> dict[str, object] | None:
        rid = it.get("id")
        if not isinstance(rid, str) or not rid.strip():
            return None
        msg = it.get("message")
        status = it.get("status")
        run_ts = _parse_datetime_to_ts(it.get("run_at"))
        if run_ts is None:
            run_ts = _parse_datetime_to_ts(it.get("run_ts"))
        if run_ts is None:
            return None

        created = it.get("created_at")
        if not isinstance(created, str) or not created.strip():
            created_ts = _parse_datetime_to_ts(it.get("created_ts"))
            created = _ts_to_iso(created_ts) if created_ts is not None else ""

        out: dict[str, object] = {
            "id": rid.strip(),
            "run_at": _ts_to_iso(run_ts),
            "created_at": created.strip() if isinstance(created, str) else "",
            "status": (status or "").strip() if isinstance(status, str) else str(status or "").strip(),
            "message": msg.strip() if isinstance(msg, str) else "",
        }

        canceled = it.get("canceled_at")
        if not isinstance(canceled, str) or not canceled.strip():
            canceled_ts = _parse_datetime_to_ts(it.get("canceled_ts"))
            canceled = _ts_to_iso(canceled_ts) if canceled_ts is not None else ""
        if isinstance(canceled, str) and canceled.strip():
            out["canceled_at"] = canceled.strip()

        triggered = it.get("triggered_at")
        if not isinstance(triggered, str) or not triggered.strip():
            triggered_ts = _parse_datetime_to_ts(it.get("triggered_ts"))
            triggered = _ts_to_iso(triggered_ts) if triggered_ts is not None else ""
        if isinstance(triggered, str) and triggered.strip():
            out["triggered_at"] = triggered.strip()

        last_output = it.get("last_output")
        if isinstance(last_output, str) and last_output.strip():
            out["last_output"] = last_output.strip()

        return out

    def _reminders_load(self) -> dict[str, object]:
        p = self._reminders_path if self._reminders_path.exists() else self._legacy_reminders_path
        if not p.exists():
            return {"version": 1, "items": []}
        try:
            raw = json.loads(p.read_text(encoding="utf-8", errors="replace") or "{}")
        except Exception:
            return {"version": 1, "items": []}
        if not isinstance(raw, dict):
            return {"version": 1, "items": []}
        raw.setdefault("version", 1)
        raw.setdefault("items", [])
        if not isinstance(raw.get("items"), list):
            raw["items"] = []
        return raw

    def _reminders_save(self, state: dict[str, object]) -> None:
        items_in = state.get("items")
        normalized_items: list[dict[str, object]] = []
        if isinstance(items_in, list):
            for it in items_in:
                if not isinstance(it, dict):
                    continue
                norm = self._normalize_reminder_item(it)
                if norm is None:
                    continue
                normalized_items.append(norm)
        state_out: dict[str, object] = {"version": 2, "items": normalized_items}
        p = self._reminders_path
        self._reminders_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state_out, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _reminder_mark_done(self, reminder_id: str, final_text: str) -> None:
        rid = (reminder_id or "").strip()
        if not rid:
            return
        with self._reminders_lock:
            state = self._reminders_load()
            items = state.get("items")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    if (it.get("id") or "") != rid:
                        continue
                    if (it.get("status") or "") != "scheduled":
                        continue
                    it["status"] = "done"
                    it["triggered_at"] = _ts_to_iso(time.time())
                    out = (final_text or "").strip()
                    if len(out) > 2000:
                        out = out[:2000]
                    it["last_output"] = out
            self._reminders_save(state)
        self._scheduler.unregister_task(rid)

    def _migrate_legacy_reminders(self) -> None:
        try:
            if self._reminders_path.exists() or self._legacy_reminders_path.exists():
                state = self._reminders_load()
                self._reminders_save(state)
        except Exception:
            return
