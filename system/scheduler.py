from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass

from .tasks import SystemContext, SystemTask


@dataclass(slots=True)
class _QueueItem:
    run_ts: float
    task_id: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _QueueItem):
            return NotImplemented
        if self.run_ts != other.run_ts:
            return self.run_ts < other.run_ts
        return self.task_id < other.task_id


class SystemScheduler:
    def __init__(self, *, ctx: SystemContext, tasks: list[SystemTask]) -> None:
        self._ctx = ctx
        self._tasks: dict[str, SystemTask] = {t.id: t for t in tasks}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._q: list[_QueueItem] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._seed()
        self._thread = threading.Thread(target=self._run, name="SystemScheduler", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def trigger(self, task_id: str, *, when_ts: float | None = None) -> bool:
        tid = (task_id or "").strip()
        if not tid:
            return False
        with self._lock:
            if tid not in self._tasks:
                return False
            run_ts = float(time.time() if when_ts is None else when_ts)
            heapq.heappush(self._q, _QueueItem(run_ts=run_ts, task_id=tid))
            self._wake.set()
        return True

    def register_task(self, task: SystemTask) -> bool:
        tid = (getattr(task, "id", "") or "").strip()
        if not tid:
            return False
        with self._lock:
            self._tasks[tid] = task
        now_ts = time.time()
        try:
            run_ts = task.schedule.next_run_ts(now_ts=now_ts)
        except Exception:
            run_ts = None
        if run_ts is not None:
            with self._lock:
                heapq.heappush(self._q, _QueueItem(run_ts=float(run_ts), task_id=tid))
                self._wake.set()
        else:
            self._wake.set()
        return True

    def unregister_task(self, task_id: str) -> bool:
        tid = (task_id or "").strip()
        if not tid:
            return False
        removed = False
        with self._lock:
            removed = self._tasks.pop(tid, None) is not None
            if removed and self._q:
                self._q = [it for it in self._q if it.task_id != tid]
                heapq.heapify(self._q)
            self._wake.set()
        return removed

    def list_task_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._tasks.keys())

    def _seed(self) -> None:
        now_ts = time.time()
        with self._lock:
            self._q.clear()
            for t in self._tasks.values():
                try:
                    run_ts = t.schedule.next_run_ts(now_ts=now_ts)
                except Exception:
                    run_ts = None
                if run_ts is None:
                    continue
                heapq.heappush(self._q, _QueueItem(run_ts=float(run_ts), task_id=t.id))

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._next_due_item()
            if item is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            try:
                task = self._tasks.get(item.task_id)
                if task is not None:
                    task.run(self._ctx)
            except Exception:
                pass

            self._reschedule(task_id=item.task_id)

    def _next_due_item(self) -> _QueueItem | None:
        while True:
            with self._lock:
                if not self._q:
                    return None
                item = self._q[0]
                now_ts = time.time()
                delay = item.run_ts - now_ts
                if delay <= 0:
                    return heapq.heappop(self._q)
            if delay > 0:
                self._wake.wait(timeout=min(1.0, delay))
                self._wake.clear()
                if self._stop.is_set():
                    return None

    def _reschedule(self, *, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return
        now_ts = time.time()
        try:
            next_ts = task.schedule.next_run_ts(now_ts=now_ts)
        except Exception:
            next_ts = None
        if next_ts is None:
            return
        with self._lock:
            heapq.heappush(self._q, _QueueItem(run_ts=float(next_ts), task_id=task_id))
            self._wake.set()
