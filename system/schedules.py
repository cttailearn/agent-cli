from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta


class Schedule:
    def next_run_ts(self, *, now_ts: float) -> float | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class IntervalSchedule(Schedule):
    seconds: float

    def next_run_ts(self, *, now_ts: float) -> float | None:
        s = float(self.seconds)
        if s <= 0:
            return None
        return now_ts + s


@dataclass(frozen=True, slots=True)
class DailyAtSchedule(Schedule):
    hour: int = 0
    minute: int = 0
    second: int = 0

    def next_run_ts(self, *, now_ts: float) -> float | None:
        now = datetime.fromtimestamp(now_ts)
        target_time = dtime(self.hour, self.minute, self.second)
        today_target = datetime.combine(now.date(), target_time)
        if today_target.timestamp() > now_ts:
            return today_target.timestamp()
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, target_time).timestamp()


@dataclass(frozen=True, slots=True)
class ManualSchedule(Schedule):
    def next_run_ts(self, *, now_ts: float) -> float | None:
        return None
