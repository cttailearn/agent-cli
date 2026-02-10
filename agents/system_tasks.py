from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentSystemTaskSpec:
    id: str
    schedule_type: str
    schedule_args: dict[str, object]
    kind: str
    payload: dict[str, object]


SYSTEM_TASK_SPECS: list[AgentSystemTaskSpec] = [
    AgentSystemTaskSpec(
        id="observer_daily_wrapup",
        schedule_type="daily_at",
        schedule_args={"hour": 23, "minute": 59, "second": 0},
        kind="observer_prompt",
        payload={
            "prompt": "\n".join(
                [
                    "请执行日终系统任务：",
                    "1) 调用 memory_kg_stats 查看 PageIndex 索引处理状态",
                    "2) 列出今天对未来有用的稳定事实/偏好（不要泛泛总结）",
                    "3) 如果发现需要写入长期记忆（core/user），再委派执行者写入",
                    "输出以要点形式给出。",
                ]
            )
        },
    ),
    AgentSystemTaskSpec(
        id="observer_daily_kg_check",
        schedule_type="daily_at",
        schedule_args={"hour": 0, "minute": 6, "second": 0},
        kind="observer_prompt",
        payload={
            "prompt": "\n".join(
                [
                    "系统已进入新的一天。",
                    "请检查昨天的聊天记录是否都已写入 PageIndex：调用 memory_kg_stats；如发现缺失，说明可能原因与补救步骤。",
                ]
            )
        },
    ),
]


def get_system_task_specs() -> list[AgentSystemTaskSpec]:
    return list(SYSTEM_TASK_SPECS)
