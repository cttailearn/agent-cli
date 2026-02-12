from __future__ import annotations

import json
import os
import queue
import re
import threading
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from . import paths
from .model import init_model


def ensure_memory_scaffold(project_root: Path) -> dict[str, str]:
    root = paths.memory_root(project_root)
    core = paths.core_dir(project_root)
    kg_dir = paths.langgraph_store_path(project_root).parent
    sessions_dir = (root / "sessions").resolve()
    extracted_dir = paths.episodic_dir(project_root)
    rollups_dir = paths.rollups_root(project_root)

    root.mkdir(parents=True, exist_ok=True)
    core.mkdir(parents=True, exist_ok=True)
    kg_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths.rollup_dir(project_root, "daily").mkdir(parents=True, exist_ok=True)
        paths.rollup_dir(project_root, "weekly").mkdir(parents=True, exist_ok=True)
        paths.rollup_dir(project_root, "monthly").mkdir(parents=True, exist_ok=True)
        paths.rollup_dir(project_root, "yearly").mkdir(parents=True, exist_ok=True)
    except Exception:
        rollups_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}
    defaults: dict[str, str] = {
        "soul.md": "\n".join(
            [
                "# 灵魂（Soul）",
                "",
                "这里记录不可妥协的核心价值、长期目标、底线与原则。",
                "这些内容应当在长期对话中逐步沉淀，并保持稳定性。",
                "",
            ]
        ),
        "traits.md": "\n".join(
            [
                "# 特性（Traits）",
                "",
                "这里记录稳定的行为风格、偏好、表达习惯与工作方式。",
                "这些内容应当在长期对话中逐步沉淀，并可缓慢演化。",
                "",
            ]
        ),
        "identity.md": "\n".join(
            [
                "# 身份（Identity）",
                "",
                "这里记录身份设定、职责边界、对外承诺与不做的事情。",
                "这些内容应当在长期对话中逐步沉淀，并保持一致性。",
                "",
            ]
        ),
        "user.md": "\n".join(
            [
                "# 用户记忆（User）",
                "",
                "这里记录与你对应“用户”的稳定信息，用于长期协作与决策一致性。内容允许更新与修订。",
                "",
                "## 偏好（Preferences）",
                "",
                "- ",
                "",
                "## 重要目标（Goals）",
                "",
                "- ",
                "",
                "## 关键决策（Decisions）",
                "",
                "- ",
                "",
                "## 关键数据（Data）",
                "",
                "- ",
                "",
            ]
        ),
    }
    for name, text in defaults.items():
        if name == "soul.md":
            p = paths.soul_path(project_root)
        elif name == "traits.md":
            p = paths.traits_path(project_root)
        elif name == "identity.md":
            p = paths.identity_path(project_root)
        elif name == "user.md":
            p = paths.user_path(project_root)
        else:
            continue
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", errors="replace")
            created[name] = p.as_posix()
    return created


def load_core_prompt(project_root: Path) -> str:
    parts: list[str] = []
    mapping = [
        ("灵魂（Soul）", paths.soul_path(project_root)),
        ("特性（Traits）", paths.traits_path(project_root)),
        ("身份（Identity）", paths.identity_path(project_root)),
        ("用户（User）", paths.user_path(project_root)),
    ]
    for title, p in mapping:
        try:
            text = p.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if not text:
            continue
        parts.append(f"## {title}\n\n{text}".strip())
    return "\n\n".join(parts).strip()

def _extract_system_text(msg) -> str:
    if msg is None:
        return ""
    text_attr = getattr(msg, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    content = getattr(msg, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _strip_block(text: str, *, start: str, end: str) -> str:
    t = text or ""
    if not t.strip():
        return ""
    i = t.find(start)
    if i < 0:
        return t
    j = t.find(end, i + len(start))
    if j < 0:
        return t[:i].rstrip()
    return f"{t[:i]}{t[j + len(end):]}".strip()


def _core_files_signature(project_root: Path) -> tuple[tuple[str, int, int], ...]:
    files = [
        paths.soul_path(project_root),
        paths.traits_path(project_root),
        paths.identity_path(project_root),
        paths.user_path(project_root),
    ]
    sig: list[tuple[str, int, int]] = []
    for p in files:
        try:
            st = p.stat()
            sig.append((p.as_posix(), int(st.st_mtime_ns), int(st.st_size)))
        except OSError:
            sig.append((p.as_posix(), 0, 0))
    return tuple(sig)


class CoreMemoryMiddleware(AgentMiddleware):
    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._core_sig: tuple[tuple[str, int, int], ...] | None = None
        self._core_text: str = ""

    def _load_core_prompt_cached(self) -> str:
        sig = _core_files_signature(self.project_root)
        if sig == self._core_sig:
            return self._core_text
        core_text = load_core_prompt(self.project_root).strip()
        self._core_sig = sig
        self._core_text = core_text
        return core_text

    def wrap_model_call(self, request: ModelRequest, handler):
        base_text = _extract_system_text(request.system_message)
        core_text = self._load_core_prompt_cached()
        if not core_text:
            return handler(request)

        start = "<CORE_MEMORIES priority=highest>"
        end = "</CORE_MEMORIES>"
        core_block = "\n".join([start, "# Core Memories（最高优先级）", "", core_text, end]).strip()
        rest = _strip_block(base_text, start=start, end=end)
        new_text = f"{core_block}\n\n{rest}".strip() if rest else core_block
        if new_text == base_text:
            return handler(request)
        return handler(request.override(system_message=SystemMessage(content=new_text)))


def create_memory_middleware(project_root: Path) -> CoreMemoryMiddleware:
    return CoreMemoryMiddleware(project_root=project_root)


class MemoryManager:
    def __init__(self, *, project_root: Path, model_name: str) -> None:
        self.project_root = project_root.resolve()
        self.model_name = model_name
        self.session_id = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._model = None
        self._queue: queue.Queue[object] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._closing = False

    def start(self) -> None:
        self._ensure_worker()

    def stop(self, *, flush: bool = True) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
            worker = self._worker
        if worker is None:
            return
        if flush:
            try:
                self._queue.join()
            except Exception:
                pass
            try:
                self._queue.put(None)
            except Exception:
                pass
            try:
                self._queue.join()
            except Exception:
                pass
            try:
                worker.join(timeout=30)
            except Exception:
                pass
        else:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
            try:
                worker.join(timeout=1)
            except Exception:
                pass
        with self._lock:
            self._worker = None

    def _session_input_tokens_threshold(self) -> int:
        raw = (os.environ.get("AGENT_SESSION_MEMORY_INPUT_TOKENS_THRESHOLD") or "").strip()
        if not raw:
            return 0
        try:
            v = int(raw)
        except ValueError:
            return 0
        return max(0, v)

    def _raw_md_path(self, *, dt: datetime) -> Path:
        root = paths.memory_root(self.project_root)
        d = (root / "sessions").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return (d / f"{dt.date().isoformat()}.md").resolve()

    def _episodic_md_path(self, *, dt: datetime) -> Path:
        d = paths.episodic_dir(self.project_root)
        d.mkdir(parents=True, exist_ok=True)
        return (d / f"{dt.date().isoformat()}.md").resolve()

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        self._model = init_model(self.model_name)
        return self._model

    def _time_bucket_tag(self, dt: datetime) -> str:
        h = int(getattr(dt, "hour", 0) or 0)
        if 0 <= h < 6:
            return "凌晨"
        if 6 <= h < 12:
            return "早上/上午"
        if 12 <= h < 14:
            return "中午"
        if 14 <= h < 18:
            return "下午"
        return "晚上"

    def _rollup_path_daily(self, d: date) -> Path:
        out_dir = paths.rollup_dir(self.project_root, "daily")
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{d.isoformat()}.md").resolve()

    def _rollup_path_weekly(self, iso_year: int, iso_week: int) -> Path:
        out_dir = paths.rollup_dir(self.project_root, "weekly")
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{iso_year}-W{iso_week:02d}.md").resolve()

    def _rollup_path_monthly(self, y: int, m: int) -> Path:
        out_dir = paths.rollup_dir(self.project_root, "monthly")
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{y:04d}-{m:02d}.md").resolve()

    def _rollup_path_yearly(self, y: int) -> Path:
        out_dir = paths.rollup_dir(self.project_root, "yearly")
        out_dir.mkdir(parents=True, exist_ok=True)
        return (out_dir / f"{y:04d}.md").resolve()

    def _parse_md_sections(self, text: str) -> dict[str, list[str]]:
        lines = (text or "").splitlines()
        out: dict[str, list[str]] = {}
        current: str | None = None
        for line in lines:
            if line.startswith("## "):
                current = line[3:].strip()
                if current:
                    out.setdefault(current, [])
                else:
                    current = None
                continue
            if current is None:
                continue
            out[current].append(line)
        return out

    def _parse_time_hhmm(self, s: str) -> tuple[int, int] | None:
        raw = (s or "").strip()
        if not raw:
            return None
        m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", raw)
        if not m:
            return None
        hh = int(m.group(1))
        mm = int(m.group(2))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return None
        return hh, mm

    def _bucket_from_time_str(self, time_str: str) -> str:
        hm = self._parse_time_hhmm(time_str)
        if hm is None:
            return ""
        hh, mm = hm
        return self._time_bucket_tag(datetime(2000, 1, 1, hh, mm))

    def _parse_structured_fields(self, body: str) -> dict[str, object]:
        s = (body or "").strip()
        if not s:
            return {}
        parts = [p.strip() for p in s.split("；") if p.strip()]
        out: dict[str, object] = {}
        keywords: list[str] = []
        for p in parts:
            if p.startswith("做了什么："):
                out["do"] = p[len("做了什么：") :].strip()
                continue
            if p.startswith("发生了什么："):
                out["happen"] = p[len("发生了什么：") :].strip()
                continue
            if p.startswith("完成了什么："):
                out["done"] = p[len("完成了什么：") :].strip()
                continue
            if p.startswith("关键词："):
                raw_kw = p[len("关键词：") :].strip()
                keywords.extend([x.strip() for x in raw_kw.split("/") if x.strip()])
                continue
        if keywords:
            out["keywords"] = keywords[:12]
        return out

    def _extract_events_from_raw_day(self, d: date) -> list[dict[str, object]]:
        p = (paths.memory_root(self.project_root) / "sessions" / f"{d.isoformat()}.md").resolve()
        if not p.exists() or not p.is_file():
            return []
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        sections = self._parse_md_sections(text)
        timeline = sections.get("时间线（Timeline）") or sections.get("时间线") or []
        out: list[dict[str, object]] = []
        if timeline:
            for line in timeline:
                ln = (line or "").strip()
                if not ln.startswith("- "):
                    continue
                m = re.match(r"^- \((\d{1,2}:\d{2})\)\s+\[([^\]]+)\]\s+\[scene=([^\]]+)\]\s*(?:\[(fact|constraint|decision|preference|plan|result)\]\s*)?(.*)$", ln)
                if not m:
                    continue
                t = m.group(1).strip()
                bucket = m.group(2).strip()
                scene = m.group(3).strip()
                kind = (m.group(4) or "").strip()
                rest = (m.group(5) or "").strip()
                fields = self._parse_structured_fields(rest)
                kws = fields.get("keywords") if isinstance(fields.get("keywords"), list) else []
                out.append(
                    {
                        "date": d.isoformat(),
                        "time": t,
                        "bucket": bucket or self._bucket_from_time_str(t),
                        "scene": scene,
                        "kind": kind,
                        "do": fields.get("do") or "",
                        "happen": fields.get("happen") or "",
                        "done": fields.get("done") or "",
                        "keywords": kws,
                        "raw": rest,
                    }
                )
            return out

        for scene, lines in sections.items():
            if scene.strip().lower() in {"时间线（timeline）", "时间线"}:
                continue
            for line in lines:
                ln = (line or "").strip()
                if not ln.startswith("- "):
                    continue
                m = re.match(r"^- \((\d{1,2}:\d{2})(?::\d{2})?\)\s*(?:\[(fact|constraint|decision|preference|plan|result)\]\s*)?(.*)$", ln)
                if not m:
                    continue
                t = m.group(1).strip()
                kind = (m.group(2) or "").strip()
                rest = (m.group(3) or "").strip()
                fields = self._parse_structured_fields(rest)
                kws = fields.get("keywords") if isinstance(fields.get("keywords"), list) else []
                out.append(
                    {
                        "date": d.isoformat(),
                        "time": t,
                        "bucket": self._bucket_from_time_str(t),
                        "scene": scene,
                        "kind": kind,
                        "do": fields.get("do") or "",
                        "happen": fields.get("happen") or "",
                        "done": fields.get("done") or "",
                        "keywords": kws,
                        "raw": rest,
                    }
                )
        return out

    def _event_score(self, ev: dict[str, object]) -> int:
        kind = str(ev.get("kind") or "").strip().lower()
        score = 0
        if kind in {"decision", "result"}:
            score += 8
        elif kind in {"constraint"}:
            score += 6
        elif kind in {"plan"}:
            score += 3
        elif kind in {"preference"}:
            score += 2
        elif kind in {"fact"}:
            score += 1
        if str(ev.get("done") or "").strip():
            score += 3
        kws = ev.get("keywords")
        if isinstance(kws, list):
            score += min(6, len([x for x in kws if isinstance(x, str) and x.strip()]))
        raw = str(ev.get("raw") or "")
        if any(tok in raw for tok in ["http://", "https://", "\\", "/", "Error", "Exception", "Traceback"]):
            score += 2
        return score

    def _scene_rollup_lines(self, events: list[dict[str, object]], *, max_scenes: int, max_lines_per_scene: int) -> list[str]:
        by_scene: dict[str, list[dict[str, object]]] = {}
        for ev in events:
            scene = str(ev.get("scene") or "").strip()
            if not scene:
                continue
            by_scene.setdefault(scene, []).append(ev)
        scored: list[tuple[int, str, list[dict[str, object]]]] = []
        for scene, evs in by_scene.items():
            total = sum(self._event_score(e) for e in evs)
            scored.append((total, scene, evs))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        scored = scored[: max(0, int(max_scenes))]
        out: list[str] = []
        for score, scene, evs in scored:
            evs2 = sorted(evs, key=lambda e: (str(e.get("date") or ""), str(e.get("time") or "")))
            kinds = sorted({str(e.get("kind") or "").strip() for e in evs2 if str(e.get("kind") or "").strip()})
            buckets = sorted({str(e.get("bucket") or "").strip() for e in evs2 if str(e.get("bucket") or "").strip()})
            out.append(f"- {scene}（重要度 {score}，事件 {len(evs2)}；类型：{'/'.join(kinds) if kinds else '-'}；时段：{'/'.join(buckets) if buckets else '-'}）")
            take = evs2[-max(1, int(max_lines_per_scene)) :]
            for e in take:
                t = str(e.get("time") or "").strip()
                k = str(e.get("kind") or "").strip()
                raw = str(e.get("raw") or "").strip()
                prefix = f"[{k}] " if k else ""
                out.append(f"  - ({t}) {prefix}{raw}".rstrip())
        return out

    def _write_rollup(self, *, p: Path, title: str, sections: list[tuple[str, list[str]]]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        out: list[str] = [f"# {title}".strip(), ""]
        for sec, lines in sections:
            s = (sec or "").strip()
            if not s:
                continue
            out.append(f"## {s}")
            out.append("")
            for ln in lines or []:
                out.append(str(ln))
            out.append("")
        while out and not out[-1].strip():
            out.pop()
        text = "\n".join(out).rstrip() + "\n"
        try:
            p.write_text(text, encoding="utf-8", errors="replace")
        except OSError:
            return

    def _week_range(self, iso_year: int, iso_week: int) -> tuple[date, date] | None:
        try:
            start = date.fromisocalendar(int(iso_year), int(iso_week), 1)
        except Exception:
            return None
        end = start + timedelta(days=6)
        return start, end

    def _path_mtime(self, p: Path) -> float:
        try:
            return float(p.stat().st_mtime)
        except OSError:
            return 0.0

    def _read_text(self, p: Path) -> str:
        if not p.exists() or not p.is_file():
            return ""
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _maybe_write_text(self, p: Path, text: str) -> bool:
        new_text = (text or "").rstrip() + ("\n" if (text or "").strip() else "")
        try:
            old = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        except OSError:
            old = ""
        if old == new_text:
            return False
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new_text, encoding="utf-8", errors="replace")
            return True
        except OSError:
            return False

    def _summarize_markdown(self, *, system_text: str, human_text: str) -> str:
        from langchain_core.messages import HumanMessage

        model = self._ensure_model()
        system = SystemMessage(content=(system_text or "").strip())
        human = HumanMessage(content=(human_text or "").strip())
        try:
            resp = model.invoke([system, human])
        except Exception:
            return ""
        text = getattr(resp, "content", "") if resp is not None else ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    def _raw_day_path(self, d: date) -> Path:
        root = paths.memory_root(self.project_root)
        ddir = (root / "sessions").resolve()
        return (ddir / f"{d.isoformat()}.md").resolve()

    def _build_daily_rollup(self, d: date) -> None:
        raw_p = self._raw_day_path(d)
        if not raw_p.exists() or not raw_p.is_file():
            return
        out_p = self._rollup_path_daily(d)
        if out_p.exists() and self._path_mtime(out_p) >= self._path_mtime(raw_p):
            return
        raw_text = self._read_text(raw_p)
        if not raw_text.strip():
            return
        sys = (
            "你是一个“日总结生成器”。输入是一整天的原始会话记录（Markdown，未提取）。"
            "只输出 Markdown，不要输出多余文本。不要编造输入中不存在的具体值（数字/路径/报错/URL/版本号等）。\n"
            f"固定标题：# 日总结 {d.isoformat()}\n"
            "固定小节（顺序保持）：\n"
            "## 摘要\n"
            "## 场景\n"
            "## 决策\n"
            "## 问题与阻塞\n"
            "## 下一步\n"
            "## 关键词\n"
            "格式要求：\n"
            "- 每个小节用无序列表（- ...）。\n"
            "- 场景小节每条尽量用：- [scene=<场景>] 做了什么：...；发生了什么：...；完成了什么：...；关键词：k1/k2；kind=<fact|constraint|decision|preference|plan|result>\n"
            "- 关键词小节每条只放一个关键词（不要整句）。\n"
        )
        out = self._summarize_markdown(system_text=sys, human_text=raw_text)
        if not out:
            return
        self._maybe_write_text(out_p, out)

    def _build_weekly_rollup(self, iso_year: int, iso_week: int) -> None:
        wr = self._week_range(iso_year, iso_week)
        if wr is None:
            return
        start, end = wr
        out_p = self._rollup_path_weekly(iso_year, iso_week)

        daily_paths: list[Path] = []
        latest = 0.0
        cur = start
        while cur <= end:
            self._build_daily_rollup(cur)
            dp = self._rollup_path_daily(cur)
            if dp.exists() and dp.is_file():
                daily_paths.append(dp)
                latest = max(latest, self._path_mtime(dp))
            cur = cur + timedelta(days=1)
        if not daily_paths:
            return
        if out_p.exists() and self._path_mtime(out_p) >= latest:
            return

        blocks: list[str] = []
        cur = start
        while cur <= end:
            dp = self._rollup_path_daily(cur)
            txt = self._read_text(dp)
            if txt.strip():
                blocks.append(f"# {cur.isoformat()}")
                blocks.append("")
                blocks.append(txt.strip())
                blocks.append("")
                blocks.append("---")
                blocks.append("")
            cur = cur + timedelta(days=1)
        inp = "\n".join(blocks).strip()
        if not inp:
            return
        sys = (
            "你是一个“周总结生成器”。输入是该周每天的日总结（Markdown）。"
            "只输出 Markdown，不要输出多余文本。不要编造输入中不存在的具体值。\n"
            f"固定标题：# 周总结 {iso_year}-W{iso_week:02d}（{start.isoformat()}..{end.isoformat()}）\n"
            "固定小节（顺序保持）：\n"
            "## 摘要\n"
            "## 主要场景\n"
            "## 关键决策\n"
            "## 风险与阻塞\n"
            "## 下周关注\n"
            "## 关键词\n"
            "格式要求：\n"
            "- 每个小节用无序列表（- ...）。\n"
            "- 主要场景每条尽量用：- [scene=<场景>] 本周做了什么：...；本周进展：...；本周完成：...；关键词：k1/k2\n"
            "- 关键词小节每条只放一个关键词。\n"
        )
        out = self._summarize_markdown(system_text=sys, human_text=inp)
        if not out:
            return
        self._maybe_write_text(out_p, out)

    def _build_monthly_rollup(self, y: int, m: int) -> None:
        first = date(y, m, 1)
        if m == 12:
            nxt = date(y + 1, 1, 1)
        else:
            nxt = date(y, m + 1, 1)
        last = nxt - timedelta(days=1)
        out_p = self._rollup_path_monthly(y, m)

        weeks: set[tuple[int, int]] = set()
        cur = first
        while cur <= last:
            iso_year, iso_week, _ = cur.isocalendar()
            weeks.add((int(iso_year), int(iso_week)))
            cur = cur + timedelta(days=1)

        week_paths: list[Path] = []
        latest = 0.0
        for iso_year, iso_week in sorted(weeks):
            self._build_weekly_rollup(iso_year, iso_week)
            wp = self._rollup_path_weekly(iso_year, iso_week)
            if wp.exists() and wp.is_file():
                week_paths.append(wp)
                latest = max(latest, self._path_mtime(wp))
        if not week_paths:
            return
        if out_p.exists() and self._path_mtime(out_p) >= latest:
            return

        blocks: list[str] = []
        for wp in week_paths:
            txt = self._read_text(wp)
            if not txt.strip():
                continue
            blocks.append(f"# {wp.stem}")
            blocks.append("")
            blocks.append(txt.strip())
            blocks.append("")
            blocks.append("---")
            blocks.append("")
        inp = "\n".join(blocks).strip()
        if not inp:
            return
        sys = (
            "你是一个“月总结生成器”。输入是该月相关的周总结（Markdown）。"
            "只输出 Markdown，不要输出多余文本。不要编造输入中不存在的具体值。\n"
            f"固定标题：# 月总结 {y:04d}-{m:02d}\n"
            "固定小节（顺序保持）：\n"
            "## 摘要\n"
            "## 主要场景\n"
            "## 关键决策\n"
            "## 风险与阻塞\n"
            "## 下月关注\n"
            "## 关键词\n"
            "格式要求：\n"
            "- 每个小节用无序列表（- ...）。\n"
            "- 关键词小节每条只放一个关键词。\n"
        )
        out = self._summarize_markdown(system_text=sys, human_text=inp)
        if not out:
            return
        self._maybe_write_text(out_p, out)

    def _build_yearly_rollup(self, y: int) -> None:
        out_p = self._rollup_path_yearly(y)
        month_paths: list[Path] = []
        latest = 0.0
        for m in range(1, 13):
            self._build_monthly_rollup(y, m)
            mp = self._rollup_path_monthly(y, m)
            if mp.exists() and mp.is_file():
                month_paths.append(mp)
                latest = max(latest, self._path_mtime(mp))
        if not month_paths:
            return
        if out_p.exists() and self._path_mtime(out_p) >= latest:
            return

        blocks: list[str] = []
        for mp in month_paths:
            txt = self._read_text(mp)
            if not txt.strip():
                continue
            blocks.append(f"# {mp.stem}")
            blocks.append("")
            blocks.append(txt.strip())
            blocks.append("")
            blocks.append("---")
            blocks.append("")
        inp = "\n".join(blocks).strip()
        if not inp:
            return
        sys = (
            "你是一个“年总结生成器”。输入是该年每个月的月总结（Markdown）。"
            "只输出 Markdown，不要输出多余文本。不要编造输入中不存在的具体值。\n"
            f"固定标题：# 年总结 {y:04d}\n"
            "固定小节（顺序保持）：\n"
            "## 年度摘要\n"
            "## 主要场景\n"
            "## 关键决策\n"
            "## 风险与阻塞\n"
            "## 明年关注\n"
            "## 关键词\n"
            "格式要求：\n"
            "- 每个小节用无序列表（- ...）。\n"
            "- 关键词小节每条只放一个关键词。\n"
        )
        out = self._summarize_markdown(system_text=sys, human_text=inp)
        if not out:
            return
        self._maybe_write_text(out_p, out)

    def _update_rollups_for_date(self, d: date) -> None:
        self._build_daily_rollup(d)
        iso_year, iso_week, _ = d.isocalendar()
        self._build_weekly_rollup(int(iso_year), int(iso_week))
        self._build_monthly_rollup(d.year, d.month)
        self._build_yearly_rollup(d.year)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._closing:
                return
            if self._worker is not None and self._worker.is_alive():
                return
            t = threading.Thread(target=self._worker_loop, name="MemoryManagerWorker", daemon=True)
            self._worker = t
            t.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if not isinstance(item, tuple) or len(item) != 4:
                    continue
                dt, user_text, assistant_text, token_usage = item
                if not isinstance(dt, datetime):
                    continue
                if not isinstance(user_text, str) or not isinstance(assistant_text, str):
                    continue
                usage = token_usage if isinstance(token_usage, dict) else None
                threshold = self._session_input_tokens_threshold()
                input_tokens = int((usage or {}).get("input_tokens") or 0)
                raw_p = self._append_raw_turn_md(dt=dt, user_text=user_text, assistant_text=assistant_text)
                items = self._extract_session_items(user_text=user_text, assistant_text=assistant_text)
                epi_p = self._append_episodic_md(dt=dt, items=items)
                if raw_p is not None and threshold > 0 and input_tokens > threshold:
                    self._rotate_thread_with_carryover(user_text=user_text, assistant_text=assistant_text, dt=dt)
                if raw_p is not None or epi_p is not None:
                    self._update_rollups_for_date(dt.date())
            except Exception:
                pass
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass

    def _extract_json_object(self, text: str) -> dict[str, object] | None:
        s = (text or "").strip()
        if not s:
            return None
        s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
        i = s.find("{")
        j = s.rfind("}")
        if i < 0 or j < 0 or j <= i:
            return None
        try:
            parsed = json.loads(s[i : j + 1])
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _extract_session_items(self, *, user_text: str, assistant_text: str) -> list[dict[str, str]]:
        from langchain_core.messages import HumanMessage

        model = self._ensure_model()
        system = SystemMessage(
            content=(
                "你是一个会话记忆提取器。给定一轮对话（用户+助手），提取可复用的信息用于“会话记忆（Episodic）”。"
                "会话记忆必须以“场景”组织，便于之后用自然语言回忆（例如：昨天你做了什么？上次我们讨论 XX 得出什么结论？）。"
                "仅输出 JSON，不要输出任何多余文本。\n"
                'JSON 格式：{"items":[{"scene":"<场景>","do":"<做了什么>","happen":"<发生了什么>","done":"<完成了什么>","keywords":["k1","k2"],"kind":"<类型>"}]}。\n'
                "要求：\n"
                "- scene：4~24 个字，场景标题，尽量具体（围绕任务/问题/决策/进展/协作），去重，不要包含日期时间。\n"
                "- do/happen/done：各 0~1 句，尽量精确，避免泛化。\n"
                "- keywords：0~8 个词，保留可检索的精确信息（数字/版本号/端口/路径/文件名/函数名/命令/参数/URL/错误码/配置键/阈值/ID 等）；不要把这些信息泛化成“若干/一些/大概”。\n"
                "- 若出现对比或变更：写清“从什么变成什么”。\n"
                "- 不要编造对话中不存在的具体值；若对话未给出具体数字/路径/报错信息，则不要补全。\n"
                "- kind：可选，取值之一：fact|constraint|decision|preference|plan|result。\n"
                "- items 数量 3~12，信息不足时可少于 3。\n"
            )
        )
        human = HumanMessage(content=f"用户：\n{(user_text or '').strip()}\n\n助手：\n{(assistant_text or '').strip()}\n")
        try:
            resp = model.invoke([system, human])
        except Exception:
            return []
        text = getattr(resp, "content", "") if resp is not None else ""
        if not isinstance(text, str):
            text = str(text)
        obj = self._extract_json_object(text)
        if obj is None:
            return []
        items = obj.get("items")
        if not isinstance(items, list):
            return []
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            k = it.get("scene")
            if not isinstance(k, str) or not k.strip():
                k = it.get("keyword")
            c = it.get("content")
            do = it.get("do")
            happen = it.get("happen")
            done = it.get("done")
            kws = it.get("keywords")
            kd = it.get("kind")
            if not isinstance(k, str):
                continue
            k = k.strip()
            if not k:
                continue
            lk = k.lower()
            if lk in seen:
                continue
            seen.add(lk)
            kind = kd.strip() if isinstance(kd, str) else ""
            if kind and kind not in {"fact", "constraint", "decision", "preference", "plan", "result"}:
                kind = ""
            kw_text = ""
            if isinstance(kws, list):
                toks = [str(x).strip() for x in kws if isinstance(x, (str, int, float)) and str(x).strip()]
                toks = toks[:8]
                kw_text = "/".join(toks)
            if not kw_text and isinstance(it.get("tags"), list):
                toks = [str(x).strip() for x in it.get("tags") if isinstance(x, (str, int, float)) and str(x).strip()]
                toks = toks[:8]
                kw_text = "/".join(toks)
            row: dict[str, str] = {"keyword": k, "kind": kind}
            if isinstance(do, str) and do.strip():
                row["do"] = do.strip()
            if isinstance(happen, str) and happen.strip():
                row["happen"] = happen.strip()
            if isinstance(done, str) and done.strip():
                row["done"] = done.strip()
            if kw_text:
                row["keywords"] = kw_text
            if isinstance(c, str) and c.strip():
                row["content"] = c.strip()
            out.append(row)
        return out

    def _append_raw_turn_md(self, *, dt: datetime, user_text: str, assistant_text: str) -> Path | None:
        ut = (user_text or "").strip()
        at = (assistant_text or "").strip()
        if not ut and not at:
            return None
        p = self._raw_md_path(dt=dt)
        time_str = dt.strftime("%H:%M")

        def _split_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
            lines = (text or "").splitlines()
            preamble: list[str] = []
            sections: list[tuple[str, list[str]]] = []
            key_to_index: dict[str, int] = {}
            current_index: int | None = None
            in_sections = False
            for line in lines:
                if line.startswith("## "):
                    in_sections = True
                    k = line[3:].strip()
                    lk = k.lower()
                    idx = key_to_index.get(lk)
                    if idx is None:
                        sections.append((k, []))
                        idx = len(sections) - 1
                        key_to_index[lk] = idx
                    current_index = idx
                    continue
                if not in_sections:
                    preamble.append(line)
                    continue
                if current_index is None:
                    continue
                sec_key, sec_lines = sections[current_index]
                sec_lines.append(line)
                sections[current_index] = (sec_key, sec_lines)
            return preamble, sections

        def _render(preamble: list[str], sections: list[tuple[str, list[str]]]) -> str:
            out: list[str] = []
            if preamble:
                while preamble and not preamble[-1].strip():
                    preamble.pop()
                out.extend(preamble)
                out.append("")
            for key, body_lines in sections:
                k = (key or "").strip()
                if not k:
                    continue
                out.append(f"## {k}")
                out.append("")
                cleaned = list(body_lines or [])
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                if cleaned:
                    out.extend(cleaned)
                    out.append("")
                out.append("")
            while out and not out[-1].strip():
                out.pop()
            return "\n".join(out).rstrip() + "\n"

        def _find_section_index(sections: list[tuple[str, list[str]]], key: str) -> int | None:
            lk = key.strip().lower()
            for i, (k, _) in enumerate(sections):
                if k.strip().lower() == lk:
                    return i
            return None

        def _as_subbullets(text: str) -> list[str]:
            lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
            out: list[str] = []
            for ln in lines:
                s = ln.rstrip()
                if not s.strip():
                    continue
                out.append(f"  - {s}")
            return out

        try:
            existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            preamble, sections = _split_sections(existing)
            if not existing.strip():
                preamble = [f"# 原始会话 {dt.date().isoformat()}"]
                sections = []
            else:
                if not any((ln or "").strip() for ln in preamble):
                    preamble = [f"# 原始会话 {dt.date().isoformat()}"]

            turns_key = "轮次（Turns）"
            turns_idx = _find_section_index(sections, turns_key)
            if turns_idx is None:
                sections.insert(0, (turns_key, []))
                turns_idx = 0

            tk, body = sections[turns_idx]
            body_lines = list(body or [])
            if ut:
                body_lines.append(f"- ({time_str}) user")
                body_lines.extend(_as_subbullets(ut))
            if at:
                body_lines.append(f"- ({time_str}) assistant")
                body_lines.extend(_as_subbullets(at))
            sections[turns_idx] = (tk, body_lines)

            new_text = _render(preamble, sections)
            if not new_text.strip():
                return None
            p.write_text(new_text, encoding="utf-8", errors="replace")
        except OSError:
            return None
        return p

    def _append_episodic_md(self, *, dt: datetime, items: list[dict[str, str]]) -> Path | None:
        if not items:
            return None
        p = self._episodic_md_path(dt=dt)
        time_str = dt.strftime("%H:%M")
        bucket = self._time_bucket_tag(dt)

        def _split_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
            lines = (text or "").splitlines()
            preamble: list[str] = []
            sections: list[tuple[str, list[str]]] = []
            key_to_index: dict[str, int] = {}
            current_index: int | None = None
            in_sections = False
            for line in lines:
                if line.startswith("## "):
                    in_sections = True
                    k = line[3:].strip()
                    lk = k.lower()
                    idx = key_to_index.get(lk)
                    if idx is None:
                        sections.append((k, []))
                        idx = len(sections) - 1
                        key_to_index[lk] = idx
                    current_index = idx
                    continue
                if not in_sections:
                    preamble.append(line)
                    continue
                if current_index is None:
                    continue
                sec_key, sec_lines = sections[current_index]
                sec_lines.append(line)
                sections[current_index] = (sec_key, sec_lines)
            return preamble, sections

        def _render(preamble: list[str], sections: list[tuple[str, list[str]]]) -> str:
            out: list[str] = []
            if preamble:
                while preamble and not preamble[-1].strip():
                    preamble.pop()
                out.extend(preamble)
                out.append("")
            for key, body_lines in sections:
                k = (key or "").strip()
                if not k:
                    continue
                out.append(f"## {k}")
                out.append("")
                cleaned = list(body_lines or [])
                while cleaned and not cleaned[0].strip():
                    cleaned.pop(0)
                while cleaned and not cleaned[-1].strip():
                    cleaned.pop()
                if cleaned:
                    out.extend(cleaned)
                    out.append("")
                out.append("")
            while out and not out[-1].strip():
                out.pop()
            return "\n".join(out).rstrip() + "\n"

        def _find_section_index(sections: list[tuple[str, list[str]]], key: str) -> int | None:
            lk = key.strip().lower()
            for i, (k, _) in enumerate(sections):
                if k.strip().lower() == lk:
                    return i
            return None

        def _build_structured_text(it: dict[str, str]) -> str:
            do = (it.get("do") or "").strip()
            happen = (it.get("happen") or "").strip()
            done = (it.get("done") or "").strip()
            keywords = (it.get("keywords") or "").strip()
            parts: list[str] = []
            if do:
                parts.append(f"做了什么：{do}")
            if happen:
                parts.append(f"发生了什么：{happen}")
            if done:
                parts.append(f"完成了什么：{done}")
            if not parts:
                content = (it.get("content") or "").strip()
                if content:
                    parts.append(content)
            if keywords:
                parts.append(f"关键词：{keywords}")
            return "；".join([p for p in parts if p]).strip()

        try:
            existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            preamble, sections = _split_sections(existing)
            if not existing.strip():
                preamble = [f"# 会话记忆（提取） {dt.date().isoformat()}"]
                sections = []
            else:
                if not any((ln or "").strip() for ln in preamble):
                    preamble = [f"# 会话记忆（提取） {dt.date().isoformat()}"]

            timeline_key = "时间线（Timeline）"
            timeline_idx = _find_section_index(sections, timeline_key)
            if timeline_idx is None:
                sections.insert(0, (timeline_key, []))
                timeline_idx = 0

            for it in items:
                k = (it.get("keyword") or "").strip()
                kind = (it.get("kind") or "").strip()
                body_text = _build_structured_text(it)
                if not k or not body_text:
                    continue
                prefix = f"[{kind}] " if kind else ""
                bullet = f"- ({time_str}) [{bucket}] {prefix}{body_text}".rstrip()

                idx = _find_section_index(sections, k)
                if idx is None:
                    sections.append((k, [bullet]))
                else:
                    sec_key, body = sections[idx]
                    body_lines = list(body or [])
                    if bullet not in body_lines:
                        body_lines.append(bullet)
                        sections[idx] = (sec_key, body_lines)

                tl_bullet = f"- ({time_str}) [{bucket}] [scene={k}] {prefix}{body_text}".rstrip()
                tl_key, tl_body = sections[timeline_idx]
                tl_lines = list(tl_body or [])
                if tl_bullet not in tl_lines:
                    tl_lines.append(tl_bullet)
                    sections[timeline_idx] = (tl_key, tl_lines)

            new_text = _render(preamble, sections)
            if not new_text.strip():
                return None
            p.write_text(new_text, encoding="utf-8", errors="replace")
        except OSError:
            return None
        return p

    def _truncate_carryover(self, text: str, *, max_chars: int) -> str:
        s = (text or "").strip()
        if not s:
            return ""
        if max_chars <= 0:
            return ""
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 1].rstrip() + "…"

    def _rotate_thread_with_carryover(self, *, user_text: str, assistant_text: str, dt: datetime) -> None:
        new_tid = uuid.uuid4().hex[:12]
        os.environ["AGENT_THREAD_ID"] = new_tid
        payload = {
            "ts": dt.isoformat(),
            "user": self._truncate_carryover(user_text, max_chars=4000),
            "assistant": self._truncate_carryover(assistant_text, max_chars=4000),
        }
        try:
            os.environ["AGENT_THREAD_CARRYOVER"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            os.environ["AGENT_THREAD_CARRYOVER"] = ""

    def record_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
        token_usage: dict[str, int] | None = None,
    ) -> Path | None:
        threshold = self._session_input_tokens_threshold()
        if threshold <= 0:
            return None
        dt = datetime.now().astimezone()
        self._ensure_worker()
        with self._lock:
            if self._closing:
                return None
        try:
            self._queue.put((dt, user_text or "", assistant_text or "", token_usage if token_usage else None))
        except Exception:
            return None
        return self._raw_md_path(dt=dt)
